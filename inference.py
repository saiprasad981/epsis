"""
inference.py — TinyCD pretrained inference & Explainable AI (HiResCAM)
========================================================================
Handles TinyCD model loading (LEVIR-CD & WHU-CD checkpoints),
preprocessing, change prediction, morphological cleanup, region analysis,
and HiResCAM model explainability.
"""

import os
import sys
import numpy as np
import torch
from torchvision.transforms import Normalize
from PIL import Image
import cv2

# --- Make Tiny_model_4_CD/models importable exactly as original repo expects ---
_TINYCD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tiny_model_4_CD")
if _TINYCD_DIR not in sys.path:
    sys.path.insert(0, _TINYCD_DIR)

from models.change_classifier import ChangeClassifier  # noqa: E402


IMG_SIZE = 256  # LEVIR-CD-256 / WHU-CD-256 training resolution
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _rekey_state_dict(state_dict: dict) -> dict:
    """Remaps state_dict keys so both WHU-CD and LEVIR-CD checkpoints load cleanly."""
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

    print(f"[inference] TinyCD loaded from '{model_path}' on device='{device}'")
    return model


def _load_and_preprocess(image_path: str) -> torch.Tensor:
    """Loads image and normalizes matching dataset.py."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
    tensor = _normalize(tensor)
    return tensor


def predict_change(model: torch.nn.Module, ref_path: str, comp_path: str, threshold: float = 0.45, use_otsu: bool = False, apply_morph: bool = True) -> dict:
    """
    Runs TinyCD change detection on T1 and T2 image paths.
    Supports dynamic Otsu thresholding or manual probability thresholding.
    """
    device = next(model.parameters()).device

    ref_tensor = _load_and_preprocess(ref_path).unsqueeze(0).to(device).float()
    test_tensor = _load_and_preprocess(comp_path).unsqueeze(0).to(device).float()

    with torch.no_grad():
        output = model(ref_tensor, test_tensor)

    prob_map = output.squeeze().cpu().numpy().astype(np.float32)

    # Determine effective threshold (Otsu or user manual)
    if use_otsu or (threshold > 0.30 and prob_map.max() < threshold and prob_map.max() > 0.01):
        prob_u8 = (np.clip(prob_map, 0, 1) * 255).astype(np.uint8)
        val, _ = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        effective_thresh = max(0.02, float(val / 255.0))
        print(f"[inference] Dynamic Otsu threshold applied: {effective_thresh:.4f} (raw max: {prob_map.max():.4f})")
    else:
        effective_thresh = threshold

    raw_mask = (prob_map >= effective_thresh).astype(np.uint8)

    if apply_morph and raw_mask.sum() > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    else:
        clean_mask = raw_mask

    changed_pixels = int(clean_mask.sum())
    total_pixels = clean_mask.size
    changed_fraction = float(changed_pixels / total_pixels)

    # Read original images as uint8 HWC for overlay visualization
    t2_img = cv2.imread(comp_path)
    if t2_img is not None:
        t2_img = cv2.cvtColor(cv2.resize(t2_img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    else:
        t2_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    overlay_rgb = create_change_overlay(t2_img, clean_mask)

    # Generate HiResCAM explainability map
    hirescam_rgb = generate_hirescam_map(model, ref_tensor, test_tensor)

    return {
        "probability_map": prob_map,
        "binary_mask": clean_mask,
        "effective_threshold": effective_thresh,
        "changed_fraction": changed_fraction,
        "prob_map_rgb": colorize_probability_map(prob_map),
        "binary_mask_rgb": colorize_binary_mask(clean_mask),
        "confidence_rgb": colorize_confidence_map(prob_map, effective_thresh),
        "overlay_rgb": overlay_rgb,
        "hirescam_rgb": hirescam_rgb,
    }


def create_change_overlay(background_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays red highlighting on detected change regions of the comparison image."""
    overlay = background_rgb.copy()
    red_mask = np.zeros_like(background_rgb)
    red_mask[mask > 0] = [255, 0, 0]  # Red for change

    blended = cv2.addWeighted(overlay, 1.0 - alpha, red_mask, alpha, 0)
    # Only update pixels where mask > 0
    mask_3d = np.stack([mask] * 3, axis=-1) > 0
    overlay[mask_3d] = blended[mask_3d]
    return overlay


def colorize_probability_map(prob_map: np.ndarray) -> np.ndarray:
    """
    Renders JET probability activation map with dynamic contrast stretching.
    Maps [p_min, p_max] across full 0-255 spectrum (Blue -> Cyan -> Green -> Yellow -> Red).
    """
    p_min, p_max = prob_map.min(), prob_map.max()
    if p_max > p_min:
        norm_prob = (prob_map - p_min) / (p_max - p_min)
    else:
        norm_prob = np.zeros_like(prob_map)

    heat_u8 = (norm_prob * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)


def colorize_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    """White = changed (255), Black = no change (0)."""
    val_mask = (binary_mask > 0).astype(np.uint8) * 255
    return np.stack([val_mask] * 3, axis=-1).astype(np.uint8)


def colorize_confidence_map(prob_map: np.ndarray, threshold: float = 0.45) -> np.ndarray:
    """
    Distance from effective decision threshold T:
    Confidence C = |p - T| / max(T, 1 - T).
    Rendered with VIRIDIS colormap (dark = uncertain boundary, bright = confident).
    """
    denom = max(threshold, 1.0 - threshold, 1e-5)
    confidence = np.abs(prob_map - threshold) / denom
    c_min, c_max = confidence.min(), confidence.max()
    if c_max > c_min:
        norm_conf = (confidence - c_min) / (c_max - c_min)
    else:
        norm_conf = np.ones_like(confidence)

    conf_u8 = (np.clip(norm_conf, 0, 1) * 255).astype(np.uint8)
    conf_color = cv2.applyColorMap(conf_u8, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(conf_color, cv2.COLOR_BGR2RGB)


def generate_hirescam_map(model: torch.nn.Module, ref_tensor: torch.Tensor, test_tensor: torch.Tensor) -> np.ndarray:
    """
    Computes HiResCAM (High-Resolution Class Activation Mapping) for model explainability.
    Directly uses gradients and feature map activations from TinyCD's decoder latents.
    """
    model.eval()
    ref = ref_tensor.clone().requires_grad_(True)
    test = test_tensor.clone().requires_grad_(True)

    features = model._encode(ref, test)
    latents = model._decode(features)

    # Retain gradient on latents
    latents.retain_grad()
    output = model._classify(latents)

    # Target: sum of predicted probability activations
    target = output.sum()
    model.zero_grad()
    target.backward()

    grads = latents.grad
    if grads is not None:
        # HiResCAM element-wise multiplication of activations and gradients
        cam = (latents * grads).sum(dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)  # ReLU positivity
    else:
        cam = output.squeeze().detach().cpu().numpy()

    c_min, c_max = cam.min(), cam.max()
    if c_max > c_min:
        norm_cam = (cam - c_min) / (c_max - c_min)
    else:
        norm_cam = np.zeros_like(cam)

    cam_u8 = (norm_cam * 255).astype(np.uint8)
    cam_color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
