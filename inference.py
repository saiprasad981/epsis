"""
inference.py — GEE-Only Pretrained Model Inference
==================================================
Handles TinyCD model loading, GEE satellite preprocessing, change prediction,
dynamic thresholding, and independent Reference Change Analysis.
"""

import os
import sys
import numpy as np
import torch
from torchvision.transforms import Normalize
from PIL import Image
import cv2

# --- Make Tiny_model_4_CD/models importable ---
_TINYCD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tiny_model_4_CD")
if _TINYCD_DIR not in sys.path:
    sys.path.insert(0, _TINYCD_DIR)

from models.change_classifier import ChangeClassifier  # noqa: E402


IMG_SIZE = 256  # Pretrained model input resolution
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _rekey_state_dict(state_dict: dict) -> dict:
    """Remaps state_dict keys so checkpoints load cleanly."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if "_mixing_mask.2._mixing." in k:
            new_k = k.replace("_mixing_mask.2._mixing.", "_mixing_mask.2.")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_model(model_path: str, device: str = None) -> torch.nn.Module:
    """Loads a pretrained TinyCD checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ChangeClassifier(pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    state_dict = _rekey_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print(f"[inference] TinyCD model loaded from '{model_path}' on device='{device}'")
    return model


def _load_and_preprocess(image_path: str) -> torch.Tensor:
    """Loads GEE satellite image and normalizes for TinyCD input."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
    tensor = _normalize(tensor)
    return tensor

def compute_reference_change_analysis(ref_path: str, comp_path: str, out_size: tuple = None) -> np.ndarray:
    """
    Computes an independent temporal difference reference change map directly from T1 and T2 GEE images.
    Combines per-channel color difference, normalized spectral difference, and structural difference.
    Returns a continuous 0–255 grayscale uint8 array (0 = black/low difference, 255 = white/strong difference).
    """
    img1 = cv2.imread(ref_path)
    img2 = cv2.imread(comp_path)
    if img1 is None or img2 is None:
        h, w = (out_size[1], out_size[0]) if out_size else (256, 256)
        return np.zeros((h, w), dtype=np.uint8)

    # Ensure identical dimensions between T1 and T2
    if out_size is not None and (img1.shape[1], img1.shape[0]) != out_size:
        img1 = cv2.resize(img1, out_size, interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, out_size, interpolation=cv2.INTER_CUBIC)
    elif (img1.shape[1], img1.shape[0]) != (img2.shape[1], img2.shape[0]):
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 1. Absolute RGB difference
    diff_rgb = cv2.absdiff(img1.astype(np.float32), img2.astype(np.float32))
    # Euclidean magnitude in 3D color space (normalized to 0..255)
    diff_mag = np.linalg.norm(diff_rgb, axis=-1) / np.sqrt(3.0)

    # 2. Spectral grayscale intensity difference
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff_gray = np.abs(g1 - g2)

    # 3. Structural / local contrast difference
    blur1 = cv2.GaussianBlur(g1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(g2, (5, 5), 0)
    diff_struct = np.abs(blur1 - blur2)

    # Combined physical difference score (0 .. 255)
    combined_score = (diff_mag * 0.5) + (diff_gray * 0.3) + (diff_struct * 0.2)

    # Zero out invalid/zero margin pixels or cloud-masked black background
    zero_mask = (g1 < 1.0) | (g2 < 1.0)
    combined_score[zero_mask] = 0.0

    # Continuous 0..255 grayscale change map (NO binary Otsu thresholding!)
    clean_ref = np.clip(combined_score, 0, 255).astype(np.uint8)

    return clean_ref


def generate_hirescam_attention_map(model: torch.nn.Module,
                                    ref_tensor: torch.Tensor,
                                    test_tensor: torch.Tensor,
                                    bg_rgb: np.ndarray = None,
                                    target_size: tuple = None,
                                    alpha: float = 0.55) -> tuple:
    """
    Computes Paper-Faithful HiResCAM Explainability for TinyCD Siamese U-Net.
    Derives feature attribution heatmap from forward activations & backward gradients.
    """
    device = next(model.parameters()).device
    model.eval()

    # Create T1/T2 tensor clones with autograd enabled
    ref = ref_tensor.clone().detach().requires_grad_(True)
    test = test_tensor.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        features = model._encode(ref, test)
        latents = model._decode(features)
        latents.retain_grad()

        output = model._classify(latents)
        prob_map = output.squeeze()

        # Define change prediction target for backward gradient pass
        # Differentiable total change score across model output
        target = prob_map.sum()

        model.zero_grad()
        target.backward()

        feat_act = latents.detach()
        feat_grad = latents.grad.detach() if latents.grad is not None else None

    out_w, out_h = target_size if target_size else (IMG_SIZE, IMG_SIZE)

    valid_cam = False
    act_mean = float(feat_act.mean().cpu()) if feat_act is not None else 0.0
    act_std = float(feat_act.std().cpu()) if feat_act is not None else 0.0
    grad_std = float(feat_grad.std().cpu()) if feat_grad is not None else 0.0
    grad_mean = float(feat_grad.mean().cpu()) if feat_grad is not None else 0.0

    if feat_act is not None and feat_grad is not None and grad_std > 1e-12:
        # HiResCAM elementwise activation * gradient (A * dY/dA) summed across channels
        raw_cam = (feat_act * feat_grad).sum(dim=1).squeeze().cpu().numpy()
        cam_std = float(raw_cam.std())

        if not (np.isnan(raw_cam).any() or np.isinf(raw_cam).any()) and cam_std > 1e-10:
            valid_cam = True

    if valid_cam:
        c_min, c_max = float(raw_cam.min()), float(raw_cam.max())
        denom = c_max - c_min
        norm_cam = (raw_cam - c_min) / denom if denom > 1e-6 else np.zeros_like(raw_cam)
    else:
        # Fallback to normalized probability map if gradients are flat
        raw_p = prob_map.detach().cpu().numpy()
        p_min, p_max = float(raw_p.min()), float(raw_p.max())
        denom = p_max - p_min
        norm_cam = (raw_p - p_min) / denom if denom > 1e-6 else np.zeros_like(raw_p)
        cam_std = float(norm_cam.std())

    # Resize normalized heatmap to target GEE ROI dimensions
    cam_u8 = (np.clip(norm_cam, 0, 1) * 255).astype(np.uint8)
    cam_resized = cv2.resize(cam_u8, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    norm_resized = cam_resized.astype(np.float32) / 255.0

    # Colorize using JET colormap (Blue -> Cyan -> Green -> Yellow -> Red)
    bgr_color = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    rgb_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2RGB)

    if bg_rgb is not None:
        bg_resized = cv2.resize(bg_rgb, (out_w, out_h))
        att_weight = np.clip(norm_resized, 0.15, 1.0)[:, :, np.newaxis] * alpha
        blended = (bg_resized.astype(float) * (1.0 - att_weight) + rgb_color.astype(float) * att_weight).astype(np.uint8)
        final_hirescam = blended
    else:
        final_hirescam = rgb_color

    stats = {
        "valid_cam": valid_cam,
        "act_shape": list(feat_act.shape),
        "act_mean": act_mean,
        "act_std": act_std,
        "grad_shape": list(feat_grad.shape) if feat_grad is not None else [],
        "grad_mean": grad_mean,
        "grad_std": grad_std,
        "cam_std": cam_std,
        "cam_min": float(norm_cam.min()),
        "cam_max": float(norm_cam.max()),
        "cam_mean": float(norm_cam.mean()),
        "nan_count": int(np.isnan(norm_cam).sum()),
        "inf_count": int(np.isinf(norm_cam).sum()),
        "target_layer": "Siamese Decoder Fused Latents (_decode)",
    }

    return final_hirescam, stats



def predict_change(model: torch.nn.Module, ref_path: str, comp_path: str, threshold: float = 0.45,
                   use_otsu: bool = True, apply_morph: bool = True) -> dict:
    """
    Runs TinyCD change detection on CURRENT GEE T1 and T2 images.
    Applies spatial mapping to match original ROI dimensions and dynamic thresholding.
    Computes independent Reference Change Analysis.
    """
    device = next(model.parameters()).device

    t1_orig = cv2.imread(ref_path)
    t2_orig = cv2.imread(comp_path)
    if t1_orig is None or t2_orig is None:
        raise ValueError(f"Failed to read image files: {ref_path} or {comp_path}")

    target_h, target_w = t1_orig.shape[0], t1_orig.shape[1]
    t1_rgb = cv2.cvtColor(t1_orig, cv2.COLOR_BGR2RGB)
    t2_rgb = cv2.cvtColor(t2_orig, cv2.COLOR_BGR2RGB)

    ref_tensor = _load_and_preprocess(ref_path).unsqueeze(0).to(device).float()
    test_tensor = _load_and_preprocess(comp_path).unsqueeze(0).to(device).float()

    with torch.no_grad():
        output = model(ref_tensor, test_tensor)

    prob_map_256 = output.squeeze().cpu().numpy().astype(np.float32)
    p_min, p_max, p_mean = float(prob_map_256.min()), float(prob_map_256.max()), float(prob_map_256.mean())

    # Map probability map back to original GEE ROI dimensions
    prob_map = cv2.resize(prob_map_256, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    prob_map = np.clip(prob_map, 0.0, 1.0)

    # Dynamic threshold determination
    if use_otsu or (threshold > 0.30 and p_max < threshold and p_max > 0.005):
        prob_u8 = (prob_map * 255).astype(np.uint8)
        val, _ = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        effective_thresh = max(0.015, float(val / 255.0))
    else:
        effective_thresh = threshold

    raw_mask = (prob_map >= effective_thresh).astype(np.uint8)

    if apply_morph and raw_mask.sum() > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    else:
        clean_mask = raw_mask

    # Binary mask format: 0 = black, 255 = white
    val_mask = (clean_mask > 0).astype(np.uint8) * 255
    binary_mask_rgb = np.stack([val_mask] * 3, axis=-1).astype(np.uint8)

    changed_pixels = int((clean_mask > 0).sum())
    total_pixels = clean_mask.size
    changed_fraction = float(changed_pixels / total_pixels)

    # Independent Reference Change Analysis directly from GEE T1/T2
    ref_change_mask = compute_reference_change_analysis(ref_path, comp_path, out_size=(target_w, target_h))
    ref_change_rgb = np.stack([ref_change_mask] * 3, axis=-1).astype(np.uint8)
    ref_changed_fraction = float((ref_change_mask > 30).sum() / ref_change_mask.size)

    # Overlay on T2 image
    overlay_rgb = create_change_overlay(t2_rgb, clean_mask)

    # Generate Paper-Faithful HiResCAM Attention Map (activations * gradients)
    hirescam_rgb, hirescam_stats = generate_hirescam_attention_map(
        model, ref_tensor, test_tensor,
        bg_rgb=t2_rgb, target_size=(target_w, target_h), alpha=0.55
    )

    # Diagnostic logging
    print("\n" + "=" * 70)
    print("      EPSIS MODEL INFERENCE AUDIT LOG")
    print("=" * 70)
    print(f"  GEE ROI Dimensions (W x H)    : {target_w}x{target_h} px")
    print(f"  Model Logits / Prob (Min/Max) : {p_min:.6f} / {p_max:.6f} (Mean: {p_mean:.6f})")
    print(f"  Effective Decision Threshold  : {effective_thresh:.4f}")
    print(f"  Predicted Change Percentage   : {changed_fraction * 100.0:.3f}% ({changed_pixels} px)")
    print(f"  Binary Mask Unique Values     : {np.unique(val_mask).tolist()}")
    print(f"  Reference Analysis Change %   : {ref_changed_fraction * 100.0:.3f}%")
    print("=" * 70 + "\n")

    print("=== EPSIS HIRESCAM DEBUG ===")
    print(f"  T1 Tensor Shape / Device     : {list(ref_tensor.shape)} / {ref_tensor.device}")
    print(f"  T2 Tensor Shape / Device     : {list(test_tensor.shape)} / {test_tensor.device}")
    print(f"  T1 Statistics (Min/Max/Mean) : [{ref_tensor.min():.4f}, {ref_tensor.max():.4f}, {ref_tensor.mean():.4f}, std={ref_tensor.std():.4f}]")
    print(f"  T2 Statistics (Min/Max/Mean) : [{test_tensor.min():.4f}, {test_tensor.max():.4f}, {test_tensor.mean():.4f}, std={test_tensor.std():.4f}]")
    print(f"  Target Layer / Valid CAM     : {hirescam_stats['target_layer']} / {hirescam_stats['valid_cam']}")
    print(f"  Activations (Shape/Mean/Std) : {hirescam_stats['act_shape']} / mean={hirescam_stats['act_mean']:.4f}, std={hirescam_stats['act_std']:.4f}")
    print(f"  Gradients (Shape/Mean/Std)   : {hirescam_stats['grad_shape']} / mean={hirescam_stats['grad_mean']:.6f}, std={hirescam_stats['grad_std']:.6f}")
    print(f"  HiResCAM Stats (Min/Max/Mean): [{hirescam_stats['cam_min']:.4f}, {hirescam_stats['cam_max']:.4f}, {hirescam_stats['cam_mean']:.4f}, std={hirescam_stats['cam_std']:.4f}]")
    print(f"  NaN Count / Inf Count        : {hirescam_stats['nan_count']} / {hirescam_stats['inf_count']}")
    print(f"  Display Dimensions (W x H)   : {target_w}x{target_h} px")
    print("=============================\n")

    return {
        "probability_map": prob_map,
        "binary_mask": clean_mask,
        "effective_threshold": effective_thresh,
        "changed_fraction": changed_fraction,
        "prob_map_rgb": hirescam_rgb,
        "binary_mask_rgb": binary_mask_rgb,
        "reference_change_analysis": ref_change_rgb,
        "ref_changed_fraction": ref_changed_fraction,
        "confidence_rgb": colorize_confidence_map(prob_map, effective_thresh),
        "overlay_rgb": overlay_rgb,
        "hirescam_rgb": hirescam_rgb,
        "hirescam_stats": hirescam_stats,
        "target_w": target_w,
        "target_h": target_h,
        "p_min": p_min,
        "p_max": p_max,
        "p_mean": p_mean,
    }


def create_change_overlay(background_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays red highlighting on detected change regions of the background image."""
    overlay = background_rgb.copy()
    red_mask = np.zeros_like(background_rgb)
    red_mask[mask > 0] = [255, 0, 0]

    blended = cv2.addWeighted(overlay, 1.0 - alpha, red_mask, alpha, 0)
    mask_3d = np.stack([mask > 0] * 3, axis=-1)
    overlay[mask_3d] = blended[mask_3d]
    return overlay


def colorize_probability_map(prob_map: np.ndarray) -> np.ndarray:
    """Renders probability activation map."""
    p_min, p_max = prob_map.min(), prob_map.max()
    denom = p_max - p_min
    norm_prob = (prob_map - p_min) / denom if denom > 1e-6 else np.zeros_like(prob_map)
    heat_u8 = (norm_prob * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)


def colorize_confidence_map(prob_map: np.ndarray, threshold: float = 0.45) -> np.ndarray:
    """Renders confidence map with VIRIDIS colormap."""
    denom = max(threshold, 1.0 - threshold, 1e-5)
    confidence = np.abs(prob_map - threshold) / denom
    c_min, c_max = confidence.min(), confidence.max()
    denom_c = c_max - c_min
    norm_conf = (confidence - c_min) / denom_c if denom_c > 1e-6 else np.ones_like(confidence)
    conf_u8 = (np.clip(norm_conf, 0, 1) * 255).astype(np.uint8)
    conf_color = cv2.applyColorMap(conf_u8, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(conf_color, cv2.COLOR_BGR2RGB)
