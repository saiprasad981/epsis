"""
test_real_changes.py — Verifies Real Satellite Image Change Detection Across Multiple Sites
=============================================================================================
Tests EPSIS pipeline on locations with verified temporal changes.
"""

import os
from satellite import fetch_satellite_pair
from inference import load_model, predict_change
from epsis_analysis import analyze_detected_changes


def test_site(name, lat, lon, r_start, r_end, c_start, c_end, thresh=0.35):
    print(f"\n" + "=" * 60)
    print(f" TESTING SITE: {name} ({lat}, {lon})")
    print(f" Window: {r_start}/{r_end} vs {c_start}/{c_end}")
    print("=" * 60)

    ref_path, comp_path, meta = fetch_satellite_pair(lat, lon, r_start, r_end, c_start, c_end, buffer_m=1000)
    model = load_model("Tiny_model_4_CD/pretrained_models/levir_best.pth")

    pred = predict_change(model, ref_path, comp_path, threshold=thresh, apply_morph=True)
    prob_map = pred["probability_map"]
    mask = pred["binary_mask"]

    analysis = analyze_detected_changes(ref_path, comp_path, prob_map, mask, lat, lon)

    print(f"  -> Provider: {meta.get('provider')}")
    print(f"  -> Raw Probability Max: {prob_map.max():.4f}, Mean: {prob_map.mean():.4f}")
    print(f"  -> Changed Area: {analysis['changed_percent']}% ({analysis['changed_pixels']} px, {analysis['changed_area_ha']} ha)")
    print(f"  -> Primary Class: {analysis['classification']['primary_class']}")
    print(f"  -> Severity: {analysis['severity']['score']} / 100 ({analysis['severity']['level']})")
    print(f"  -> Risk Level: {analysis['risk']['risk_title']}")
    print(f"  -> HiResCAM Map Generated: {pred['hirescam_rgb'].shape == (256, 256, 3)}")

    return analysis


if __name__ == "__main__":
    # Test 1: Kokapet Financial District Growth Corridor (Hyderabad)
    test_site("Kokapet Urban Development", 17.3950, 78.3300, "2021-01-01", "2021-06-30", "2023-06-01", "2023-12-31", thresh=0.35)

    # Test 2: Bengaluru Whitefield Tech Corridor
    test_site("Bengaluru Whitefield Corridor", 12.9800, 77.7400, "2020-01-01", "2020-12-31", "2023-01-01", "2023-12-31", thresh=0.35)
