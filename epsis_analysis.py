"""
epsis_analysis.py — EPSIS Intelligence Engine
===============================================
Provides Change Classification, Quantitative Severity Analysis,
Risk Assessment, and Spatial Region Extraction for detected satellite changes.
"""

import numpy as np
import cv2
from PIL import Image


def analyze_detected_changes(ref_img_path: str, comp_img_path: str,
                           prob_map: np.ndarray, binary_mask: np.ndarray,
                           lat: float, lon: float) -> dict:
    """
    Performs full EPSIS multi-stage analysis:
      - Change region extraction
      - Spectral / Spatial Change Classification
      - Severity Analysis (0-100 Score + Category)
      - Risk Assessment (Impact + Recommended Actions)
    """
    total_pixels = binary_mask.size
    changed_pixels = int(binary_mask.sum())
    changed_fraction = changed_pixels / total_pixels if total_pixels > 0 else 0.0
    changed_percent = changed_fraction * 100.0

    # 1. Connected Component Region Extraction
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8))

    regions = []
    # Index 0 is background
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 5:  # filter noise blobs < 5 pixels
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        # Estimate real-world area (assuming 1km radius tile = ~4 km² total tile)
        tile_area_m2 = 4000000.0  # 4 km²
        region_area_m2 = (area / total_pixels) * tile_area_m2
        region_hectares = region_area_m2 / 10000.0

        regions.append({
            "region_id": i,
            "pixel_area": area,
            "area_m2": region_area_m2,
            "hectares": region_hectares,
            "bbox": [x, y, w, h],
            "centroid": [round(cx, 1), round(cy, 1)],
            "mean_prob": float(prob_map[labels == i].mean()) if area > 0 else 0.0
        })

    # Sort regions by size descending
    regions.sort(key=lambda r: r["pixel_area"], reverse=True)

    # 2. Spectral Analysis for Change Classification
    classification = classify_change_type(ref_img_path, comp_img_path, binary_mask)

    # 3. Quantitative Severity Analysis
    severity = calculate_severity(changed_fraction, prob_map, binary_mask, len(regions))

    # 4. Risk Assessment
    risk = assess_risk(classification, severity, changed_percent, lat, lon)

    total_changed_m2 = (changed_fraction) * 4000000.0
    total_changed_ha = total_changed_m2 / 10000.0

    return {
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_percent": round(changed_percent, 2),
        "changed_area_m2": round(total_changed_m2, 1),
        "changed_area_ha": round(total_changed_ha, 2),
        "num_regions": len(regions),
        "regions": regions,
        "classification": classification,
        "severity": severity,
        "risk": risk
    }


def classify_change_type(ref_path: str, comp_path: str, mask: np.ndarray) -> dict:
    """
    Classifies detected change based on spectral channel shifts (RGB)
    between reference (T1) and comparison (T2) images within changed pixels.
    """
    if mask.sum() == 0:
        return {
            "primary_class": "No Significant Change",
            "confidence": 1.0,
            "breakdown": {"Urban/Construction": 0.0, "Vegetation Shift": 0.0, "Water Body Change": 0.0, "Land Surface Shift": 0.0},
            "description": "No measurable change detected across the selected temporal window."
        }

    try:
        ref_arr = np.array(Image.open(ref_path).convert("RGB").resize((256, 256))).astype(float)
        comp_arr = np.array(Image.open(comp_path).convert("RGB").resize((256, 256))).astype(float)

        mask_bool = mask > 0
        r1, g1, b1 = ref_arr[mask_bool, 0], ref_arr[mask_bool, 1], ref_arr[mask_bool, 2]
        r2, g2, b2 = comp_arr[mask_bool, 0], comp_arr[mask_bool, 1], comp_arr[mask_bool, 2]

        brightness1 = (r1 + g1 + b1) / 3.0
        brightness2 = (r2 + g2 + b2) / 3.0
        diff_bright = brightness2 - brightness1

        greenness1 = g1 - (r1 + b1) / 2.0
        greenness2 = g2 - (r2 + b2) / 2.0
        diff_green = greenness2 - greenness1

        blueness1 = b1 - (r1 + g1) / 2.0
        blueness2 = b2 - (r2 + g2) / 2.0
        diff_blue = blueness2 - blueness1

        # Spectral scoring heuristic
        scores = {
            "Urban / Construction": float(np.maximum(diff_bright.mean(), 0) * 1.5 + np.std(brightness2) * 0.5),
            "Vegetation Shift": float(np.abs(diff_green.mean()) * 2.0),
            "Water Body Change": float(np.abs(diff_blue.mean()) * 2.0 + (diff_bright.mean() < 0) * 20.0),
            "Land Surface / Soil": float(np.abs(diff_bright.mean()) * 0.8 + 10.0)
        }

        total_score = sum(scores.values()) + 1e-5
        probs = {k: round(v / total_score, 3) for k, v in scores.items()}
        primary = max(probs, key=probs.get)

        descriptions = {
            "Urban / Construction": "Increased high-reflectance structural coverage, typical of building development, concrete laying, or road expansion.",
            "Vegetation Shift": "Substantial gain or loss in vegetative canopy, green foliage density, or agricultural crop land.",
            "Water Body Change": "Shifts in water extent, reservoir levels, moisture content, or lake shoreline movement.",
            "Land Surface / Soil": "Clearing of topsoil, earthworks, grading, or ground surface texture alteration."
        }

        return {
            "primary_class": primary,
            "confidence": probs[primary],
            "breakdown": probs,
            "description": descriptions[primary]
        }
    except Exception as e:
        print(f"[analysis] Classification error: {e}")
        return {
            "primary_class": "Urban / Land Change",
            "confidence": 0.75,
            "breakdown": {"Urban / Land Change": 0.75, "Vegetation Shift": 0.25},
            "description": "Temporal surface modification identified across detected change bounds."
        }


def calculate_severity(changed_fraction: float, prob_map: np.ndarray, mask: np.ndarray, num_regions: int) -> dict:
    """
    Computes a quantitative 0-100 Severity Score based on area %,
    peak model confidence, and regional concentration.
    """
    if mask.sum() == 0:
        return {"score": 0.0, "level": "None", "color": "#22c55e", "description": "Baseline state. No active hazards or structural modifications."}

    changed_pct = changed_fraction * 100.0
    mean_prob = float(prob_map[mask > 0].mean()) if mask.sum() > 0 else 0.0
    max_prob = float(prob_map.max())

    # Score components (weighted)
    area_component = min(changed_pct * 4.0, 50.0)      # up to 50 pts for area
    prob_component = mean_prob * 30.0                  # up to 30 pts for probability confidence
    cluster_component = min(num_regions * 2.0, 20.0)   # up to 20 pts for multi-region impact

    score = round(min(area_component + prob_component + cluster_component, 100.0), 1)

    if score < 20.0:
        level = "Low"
        color = "#3b82f6"  # Blue
        desc = "Minor or localized surface modification. Low immediate impact."
    elif score < 45.0:
        level = "Moderate"
        color = "#eab308"  # Yellow
        desc = "Notable land conversion or structural change across multiple sub-regions."
    elif score < 75.0:
        level = "High"
        color = "#f97316"  # Orange
        desc = "Substantial geographic alteration. Requires active monitoring and planning oversight."
    else:
        level = "Critical"
        color = "#ef4444"  # Red
        desc = "Large-scale rapid territorial transformation or major environmental disturbance."

    return {
        "score": score,
        "level": level,
        "color": color,
        "description": desc,
        "metrics": {
            "area_percentage": round(changed_pct, 2),
            "confidence_mean": round(mean_prob, 3),
            "peak_probability": round(max_prob, 3),
            "region_count": num_regions
        }
    }


def assess_risk(classification: dict, severity: dict, changed_pct: float, lat: float, lon: float) -> dict:
    """
    Generates actionable risk assessment findings and recommended interventions.
    """
    primary_cls = classification.get("primary_class", "General Change")
    sev_level = severity.get("level", "Low")

    risk_ratings = {
        "Low": ("MINIMAL RISK", "#22c55e", "Low environmental or infrastructural vulnerability."),
        "Moderate": ("MODERATE RISK", "#eab308", "Moderate potential impact on local hydrology, traffic, or land cover."),
        "High": ("HIGH RISK", "#f97316", "Significant risk of habitat disruption, unpermitted construction, or runoff change."),
        "Critical": ("CRITICAL RISK", "#ef4444", "High threat level. Potential rapid environmental degradation or major structural encroachment.")
    }

    title, color, overview = risk_ratings.get(sev_level, risk_ratings["Low"])

    action_items = []
    if "Urban" in primary_cls:
        action_items.extend([
            "Cross-verify detected construction footprint against municipal master plans and zoning permits.",
            "Inspect site drainage and impervious surface ratio to mitigate urban heat and stormwater runoff.",
            "Schedule field survey for verification of newly constructed structures."
        ])
    elif "Vegetation" in primary_cls:
        action_items.extend([
            "Assess forest canopy loss or agricultural land conversion against regional environmental protection guidelines.",
            "Monitor soil erosion potential across newly exposed slope zones.",
            "Verify whether vegetation clearance was authorized under seasonal land clearance permits."
        ])
    elif "Water" in primary_cls:
        action_items.extend([
            "Evaluate hydrological impact on downstream water tables, wetlands, and reservoir storage capacity.",
            "Check for potential agricultural runoff or water body encroachment.",
            "Conduct water quality and turbidity testing across changed aquatic boundaries."
        ])
    else:
        action_items.extend([
            "Perform periodic high-resolution satellite monitoring to track change progression rate.",
            "Log change vector boundaries in GIS database for multi-temporal trend analysis."
        ])

    return {
        "risk_title": title,
        "risk_color": color,
        "overview": overview,
        "location_coords": f"{lat:.6f}, {lon:.6f}",
        "action_items": action_items
    }
