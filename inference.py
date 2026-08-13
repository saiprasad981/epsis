"""
inference.py — TinyCD pretrained inference
=============================================
Preprocessing here matches Tiny_model_4_CD/dataset/dataset.py EXACTLY,
verified line-by-line from the actual file (not guessed):

  - matplotlib.image.imread() on a PNG returns float32 pixels already
    scaled to [0,1] — NOT a separate ToTensor()/255 step. We replicate
    that scaling manually since we load via PIL (from GEE thumbnails),
    which gives uint8 [0,255].
  - Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) — ImageNet
    stats, applied directly on the [0,1]-scaled tensor. Confirmed from
    dataset.py's _to_tensors(), not assumed.
  - No resize step in dataset.py because their dataset is pre-cropped to
    256x256 on disk. Our GEE thumbnails are NOT pre-cropped, so we add
    a resize to 256x256 ourselves before normalization — this is the one
    step dataset.py doesn't need but we do, since our image source differs.

Model I/O, verified from change_classifier.py and test_ondata.py:
  - model(reference_tensor, test_tensor) — two positional args, not a dict,
    not a stacked tensor.
  - Output already has Sigmoid applied inside the model's final
    PixelwiseLinear layer. DO NOT apply sigmoid again on the output.
  - Output shape (B, 1, H, W); test_ondata.py squeezes to (B, H, W).
  - Their own threshold: generated_mask > 0.5.
"""

import os
import sys

import numpy as np
import torch
from torchvision.transforms import Normalize
from PIL import Image
import cv2

# --- Make Tiny_model_4_CD/models importable exactly as the original repo
# expects (change_classifier.py itself does `from models.layers import ...`,
# so the Tiny_model_4_CD folder — not EPSIS root — must be on sys.path). ---
_TINYCD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tiny_model_4_CD")
if _TINYCD_DIR not in sys.path:
    sys.path.insert(0, _TINYCD_DIR)

from models.change_classifier import ChangeClassifier  # noqa: E402


IMG_SIZE = 256  # LEVIR-CD-256 / WHU-CD-256 training resolution
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str = None) -> torch.nn.Module:
    """
    Loads the pretrained TinyCD checkpoint.

    Note on `pretrained=False` below: the official test_ondata.py calls
    ChangeClassifier() with its default pretrained=True, which downloads
    ImageNet weights for the EfficientNet-B4 backbone from torchvision on
    every construction — before those weights get fully overwritten by
    load_state_dict() two lines later. Since state_dict() saves ALL
    parameters (backbone included), pretrained=False + load_state_dict()
    produces numerically identical weights, just without the redundant
    internet download every time the app restarts. This is not a change
    to model architecture or behavior — only to how the (immediately
    discarded) initial backbone weights are obtained.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = ChangeClassifier(pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print(f"[inference] TinyCD checkpoint loaded from '{model_path}' on device='{device}'")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[inference] Model parameters: {n_params:,}")

    return model


# ---------------------------------------------------------------------------
# Preprocessing — must match dataset.py's _to_tensors() exactly
# ---------------------------------------------------------------------------

def _load_and_preprocess(image_path: str) -> torch.Tensor:
    """
    Loads an image the way the model expects:
      1. Load as RGB (drop alpha if present)
      2. Resize to 256x256 (dataset.py skips this since their data is
         pre-cropped; our GEE thumbnails are not)
      3. Scale to [0,1] float32 — replicates matplotlib.image.imread's
         automatic PNG scaling, which dataset.py relies on implicitly
      4. HWC -> CHW
      5. ImageNet normalize (mean/std) — exactly as dataset.py does
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0  # matches imread's PNG scaling
    tensor = torch.from_numpy(arr).permute(2, 0, 1)     # HWC -> CHW
    tensor = _normalize(tensor)
    return tensor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_change(model: torch.nn.Module, ref_path: str, comp_path: str, threshold: float = 0.5) -> dict:
    """
    Runs TinyCD change detection on a T1/T2 image pair.

    Returns a dict with:
      probability_map : (256,256) float32, model's raw sigmoid output (0-1)
      binary_mask      : (256,256) uint8, thresholded change mask (0/1)
      changed_fraction : float, % of the tile flagged as changed
      prob_map_rgb     : (256,256,3) uint8, JET-colormapped probability heatmap
      binary_mask_rgb  : (256,256,3) uint8, white=changed / black=no-change
      confidence_rgb   : (256,256,3) uint8, VIRIDIS-colormapped confidence map
    """
    device = next(model.parameters()).device

    ref_tensor = _load_and_preprocess(ref_path).unsqueeze(0).to(device).float()
    test_tensor = _load_and_preprocess(comp_path).unsqueeze(0).to(device).float()

    print(f"[inference] ref_tensor shape={tuple(ref_tensor.shape)}  "
          f"test_tensor shape={tuple(test_tensor.shape)}")
    print(f"[inference] ref_tensor  min={ref_tensor.min():.3f} max={ref_tensor.max():.3f} mean={ref_tensor.mean():.3f}")
    print(f"[inference] test_tensor min={test_tensor.min():.3f} max={test_tensor.max():.3f} mean={test_tensor.mean():.3f}")

    with torch.no_grad():
        output = model(ref_tensor, test_tensor)  # (1, 1, 256, 256) — sigmoid ALREADY applied inside the model

    prob_map = output.squeeze().cpu().numpy().astype(np.float32)  # (256, 256), values already in [0,1]

    print(f"[inference] prob_map shape={prob_map.shape} "
          f"min={prob_map.min():.4f} max={prob_map.max():.4f} mean={prob_map.mean():.4f}")

    binary_mask = (prob_map > threshold).astype(np.uint8)
    changed_pixels = int(binary_mask.sum())
    total_pixels = binary_mask.size
    changed_fraction = changed_pixels / total_pixels

    print(f"[inference] threshold={threshold}  changed_pixels={changed_pixels}/{total_pixels} "
          f"({changed_fraction*100:.2f}%)")

    return {
        "probability_map": prob_map,
        "binary_mask": binary_mask,
        "changed_fraction": changed_fraction,
        "prob_map_rgb": colorize_probability_map(prob_map),
        "binary_mask_rgb": colorize_binary_mask(binary_mask),
        "confidence_rgb": colorize_confidence_map(prob_map),
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def colorize_probability_map(prob_map: np.ndarray) -> np.ndarray:
    """Raw model probability, 0 (no change, blue) -> 1 (change, red). JET colormap."""
    heat_u8 = (np.clip(prob_map, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)


def colorize_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Thresholded mask: white = changed, black = no change."""
    return np.stack([binary_mask * 255] * 3, axis=-1).astype(np.uint8)


def colorize_confidence_map(prob_map: np.ndarray) -> np.ndarray:
    """
    Confidence = how far the prediction sits from the decision boundary (0.5).
    0 = model is unsure (prob near 0.5), 1 = model is very sure either way.
    Different signal from the probability map — this shows WHERE the model
    is uncertain, regardless of which class it leans toward.
    """
    confidence = np.abs(prob_map - 0.5) * 2  # rescale [0, 0.5] -> [0, 1]
    conf_u8 = (np.clip(confidence, 0, 1) * 255).astype(np.uint8)
    conf_color = cv2.applyColorMap(conf_u8, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(conf_color, cv2.COLOR_BGR2RGB)
