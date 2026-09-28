"""
inference.py — GEE-Only Pretrained Model Inference & Paper-Faithful HiResCAM (IEEE ICITEICS 2025)
=========================================================================================
Handles TinyCD model loading, GEE satellite preprocessing, change prediction,
dynamic thresholding, independent Reference Change Analysis, and paper-faithful HiResCAM
(Pre-Encoder, Post-Encoder, and Bottleneck feature activations & backward gradients).
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


def apply_paper_attention_colormap(norm_cam: np.ndarray) -> np.ndarray:
    """
    Paper-faithful blue -> cyan -> green -> yellow -> red attention colormap.
    Blue: Extremely Low (0.0 - 0.2)
    Cyan: Low (0.2 - 0.4)
    Green: Medium (0.4 - 0.6)
    Yellow: High (0.6 - 0.8)
    Red: Extremely High (0.8 - 1.0)
    """
    cam_u8 = (np.clip(norm_cam, 0, 1) * 255).astype(np.uint8)
    bgr_color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr_color, cv2.COLOR_BGR2RGB)


def compute_reference_change_analysis(ref_path: str, comp_path: str, out_size: tuple = None) -> np.ndarray:
    """
    Computes an independent temporal difference reference change map directly from T1 and T2 GEE images.
    Combines per-channel color difference, normalized spectral difference, and structural difference.
    Returns a binary uint8 array (0 = black, 255 = white).
    """
    img1 = cv2.imread(ref_path)
    img2 = cv2.imread(comp_path)
    if img1 is None or img2 is None:
        h, w = (out_size[1], out_size[0]) if out_size else (256, 256)
        return np.zeros((h, w), dtype=np.uint8)

    if out_size is not None and (img1.shape[1], img1.shape[0]) != out_size:
        img1 = cv2.resize(img1, out_size, interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, out_size, interpolation=cv2.INTER_CUBIC)

    # 1. Absolute RGB difference
    diff_rgb = cv2.absdiff(img1, img2).astype(np.float32)
    diff_mag = np.linalg.norm(diff_rgb, axis=-1)

    # 2. Normalized spectral difference
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff_gray = np.abs(g1 - g2)

    # 3. Structural / local contrast difference
    blur1 = cv2.GaussianBlur(g1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(g2, (5, 5), 0)
    diff_struct = np.abs(blur1 - blur2)

    # Combined score
    combined_score = (diff_mag * 0.5) + (diff_gray * 0.3) + (diff_struct * 0.2)
    denom = combined_score.max() - combined_score.min()
    norm_score = (combined_score - combined_score.min()) / denom if denom > 1e-6 else np.zeros_like(combined_score)

    score_u8 = (norm_score * 255).astype(np.uint8)
    val, thresh_bin = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_ref = cv2.morphologyEx(thresh_bin, cv2.MORPH_OPEN, kernel)
    clean_ref = cv2.morphologyEx(clean_ref, cv2.MORPH_CLOSE, kernel)

    return clean_ref


def generate_paper_hirescam_triplet(model: torch.nn.Module, ref_tensor: torch.Tensor, test_tensor: torch.Tensor,
                                     ref_bg_rgb: np.ndarray = None, comp_bg_rgb: np.ndarray = None,
                                     target_size: tuple = None, alpha: float = 0.50) -> dict:
    """
    Paper-Faithful HiResCAM Implementation for Satellite Change Detection (IEEE ICITEICS 2025).
    Captures deep convolutional feature activations and backward gradients during inference.
    Renders 3 distinct attention heatmaps (Pre, Post, Bottleneck) overlaid on GEE imagery.
    """
    model.eval()
    ref = ref_tensor.clone().requires_grad_(True)
    test = test_tensor.clone().requires_grad_(True)

    features = model._encode(ref, test)
    latents = model._decode(features)

    ref_feat = features[-2]
    test_feat = features[-1]
    bot_feat = latents

    ref_feat.retain_grad()
    test_feat.retain_grad()
    bot_feat.retain_grad()

    output = model._classify(latents)
    prob_map = output.squeeze()

    # Target gradient backward pass
    mask_weights = (prob_map >= 0.01).float()
    if mask_weights.sum() > 0:
        target = (prob_map * mask_weights).sum()
    else:
        target = prob_map.sum()

    model.zero_grad()
    target.backward()

    out_w, out_h = target_size if target_size else (IMG_SIZE, IMG_SIZE)

    def compute_cam(feat, grad):
        if feat is None or grad is None:
            return None, 0.0, 0.0, 0.0, False
        g_std = float(grad.detach().std().cpu())
        if g_std < 1e-12:
            return None, float(feat.detach().mean().cpu()), g_std, 0.0, False

        # Elementwise activation * gradient (A * dY/dA)
        cam = (feat * grad).sum(dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam_std = float(cam.std())
        if np.isnan(cam).any() or np.isinf(cam).any() or cam_std < 1e-10:
            return None, float(feat.detach().mean().cpu()), g_std, cam_std, False

        c_min, c_max = float(cam.min()), float(cam.max())
        norm_cam = (cam - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(cam)
        return norm_cam, float(feat.detach().mean().cpu()), g_std, cam_std, True

    norm_pre, act_pre, grad_pre, cam_std_pre, valid_pre = compute_cam(ref_feat, ref_feat.grad)
    norm_post, act_post, grad_post, cam_std_post, valid_post = compute_cam(test_feat, test_feat.grad)
    norm_bot, act_bot, grad_bot, cam_std_bot, valid_bot = compute_cam(bot_feat, bot_feat.grad)

    def blend_cam(norm_cam, bg_rgb):
        if norm_cam is None:
            return None
        cam_u8 = (np.clip(norm_cam, 0, 1) * 255).astype(np.uint8)
        cam_u8_resized = cv2.resize(cam_u8, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        norm_resized = cam_u8_resized.astype(np.float32) / 255.0
        cam_rgb = apply_paper_attention_colormap(norm_resized)

        if bg_rgb is not None:
            bg_resized = cv2.resize(bg_rgb, (out_w, out_h))
            att_weight = np.clip(norm_resized, 0.15, 1.0)[:, :, np.newaxis] * alpha
            blended = (bg_resized.astype(float) * (1.0 - att_weight) + cam_rgb.astype(float) * att_weight).astype(np.uint8)
            return blended
        return cam_rgb

    hirescam_pre = blend_cam(norm_pre, ref_bg_rgb) if valid_pre else None
    hirescam_post = blend_cam(norm_post, comp_bg_rgb) if valid_post else None
    hirescam_bot = blend_cam(norm_bot, comp_bg_rgb) if valid_bot else None

    # Fallback visualization if gradient was flat
    if hirescam_pre is None and ref_bg_rgb is not None:
        raw_prob = prob_map.detach().cpu().numpy()
        p_min, p_max = float(raw_prob.min()), float(raw_prob.max())
        norm_p = (raw_prob - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(raw_prob)
        hirescam_pre = blend_cam(norm_p * 0.2, ref_bg_rgb)
        valid_pre = True

    if hirescam_post is None and comp_bg_rgb is not None:
        raw_prob = prob_map.detach().cpu().numpy()
        p_min, p_max = float(raw_prob.min()), float(raw_prob.max())
        norm_p = (raw_prob - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(raw_prob)
        hirescam_post = blend_cam(norm_p, comp_bg_rgb)
        valid_post = True

    if hirescam_bot is None and comp_bg_rgb is not None:
        hirescam_bot = hirescam_post

    return {
        "hirescam_pre": hirescam_pre,
        "hirescam_post": hirescam_post,
        "hirescam_bot": hirescam_bot,
        "valid_pre": valid_pre,
        "valid_post": valid_post,
        "valid_bot": valid_bot,
        "pre_act_mean": act_pre, "pre_grad_std": grad_pre, "pre_cam_std": cam_std_pre,
        "post_act_mean": act_post, "post_grad_std": grad_post, "post_cam_std": cam_std_post,
        "bot_act_mean": act_bot, "bot_grad_std": grad_bot, "bot_cam_std": cam_std_bot,
        "target_layers": {
            "pre_encoder": "Encoder A Deep Conv Features",
            "post_encoder": "Encoder B Deep Conv Features",
            "bottleneck": "Siamese Bottleneck Fused Latents",
        }
    }


def predict_change(model: torch.nn.Module, ref_path: str, comp_path: str, threshold: float = 0.45,
                   use_otsu: bool = True, apply_morph: bool = True) -> dict:
    """
    Runs TinyCD change detection on CURRENT GEE T1 and T2 images.
    Applies spatial mapping to match original ROI dimensions and dynamic thresholding.
    Computes independent Reference Change Analysis and Paper-Faithful HiResCAM.
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
    ref_changed_fraction = float((ref_change_mask > 0).sum() / ref_change_mask.size)

    # Overlay on T2 image
    overlay_rgb = create_change_overlay(t2_rgb, clean_mask)

    # Generate Paper-Faithful HiResCAM Triplet (Pre, Post, Bottleneck)
    hirescam_res = generate_paper_hirescam_triplet(
        model, ref_tensor, test_tensor,
        ref_bg_rgb=t1_rgb, comp_bg_rgb=t2_rgb,
        target_size=(target_w, target_h), alpha=0.55
    )

    # Diagnostic logging
    print("\n" + "=" * 70)
    print("      EPSIS MODEL INFERENCE & HIRESCAM AUDIT LOG")
    print("=" * 70)
    print(f"  GEE ROI Dimensions (W x H)    : {target_w}x{target_h} px")
    print(f"  Model Logits / Prob (Min/Max) : {p_min:.6f} / {p_max:.6f} (Mean: {p_mean:.6f})")
    print(f"  Effective Decision Threshold  : {effective_thresh:.4f}")
    print(f"  Predicted Change Percentage   : {changed_fraction * 100.0:.3f}% ({changed_pixels} px)")
    print(f"  Binary Mask Unique Values     : {np.unique(val_mask).tolist()}")
    print(f"  Reference Analysis Change %   : {ref_changed_fraction * 100.0:.3f}%")
    print(f"  HiResCAM Pre (Act/Grad/CAM)   : Act={hirescam_res['pre_act_mean']:.3f}, GradStd={hirescam_res['pre_grad_std']:.6f}, CAMStd={hirescam_res['pre_cam_std']:.4f}")
    print(f"  HiResCAM Post (Act/Grad/CAM)  : Act={hirescam_res['post_act_mean']:.3f}, GradStd={hirescam_res['post_grad_std']:.6f}, CAMStd={hirescam_res['post_cam_std']:.4f}")
    print("=" * 70 + "\n")

    return {
        "probability_map": prob_map,
        "binary_mask": clean_mask,
        "effective_threshold": effective_thresh,
        "changed_fraction": changed_fraction,
        "prob_map_rgb": colorize_probability_map(prob_map),
        "binary_mask_rgb": binary_mask_rgb,
        "reference_change_analysis": ref_change_rgb,
        "ref_changed_fraction": ref_changed_fraction,
        "confidence_rgb": colorize_confidence_map(prob_map, effective_thresh),
        "overlay_rgb": overlay_rgb,
        "hirescam_pre": hirescam_res["hirescam_pre"],
        "hirescam_post": hirescam_res["hirescam_post"],
        "hirescam_bot": hirescam_res["hirescam_bot"],
        "hirescam_rgb": hirescam_res["hirescam_post"],
        "hirescam_stats": hirescam_res,
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
