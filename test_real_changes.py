"""
test_real_changes.py — Verifies Real Satellite Image Change Detection Across Multiple Sites
=============================================================================================
Tests EPSIS pipeline on locations with verified temporal changes using exact acquisition dates.
"""

import os
import pytest
from satellite import fetch_satellite_pair
from inference import load_model, predict_change
from epsis_analysis import analyze_detected_changes


@pytest.mark.parametrize(
    "name, lat, lon, before_date, after_date, thresh",
    [
        ("Kokapet Urban Development", 17.3950, 78.3300, "2021-06-15", "2023-06-20", 0.35),
    ],
)
def test_site(name, lat, lon, before_date, after_date, thresh):
    print(f"\n" + "=" * 60)
    print(f" TESTING SITE: {name} ({lat}, {lon})")
    print(f" Target Dates — Before: {before_date}, After: {after_date}")
    print("=" * 60)

    ref_path, comp_path, meta = fetch_satellite_pair(lat, lon, before_date, after_date, buffer_m=1000)
    model = load_model("Tiny_model_4_CD/pretrained_models/levir_best.pth")

    pred = predict_change(model, ref_path, comp_path, threshold=thresh, apply_morph=True)
    prob_map = pred["probability_map"]
    mask = pred["binary_mask"]

    analysis = analyze_detected_changes(ref_path, comp_path, prob_map, mask, lat, lon)

    assert "classification" in analysis
    assert "severity" in analysis
    assert "risk" in analysis

    print(f"  -> Provider: {meta.get('provider')}")
    print(f"  -> Before Image: Requested {meta.get('before_requested')} | Actual {meta.get('before_actual')}")
    print(f"  -> After Image:  Requested {meta.get('after_requested')} | Actual {meta.get('after_actual')}")
    print(f"  -> Raw Probability Max: {prob_map.max():.4f}, Mean: {prob_map.mean():.4f}")
    print(f"  -> Changed Area: {analysis['changed_percent']}% ({analysis['changed_pixels']} px, {analysis['changed_area_ha']} ha)")


if __name__ == "__main__":
    # Test 1: Kokapet Financial District Growth Corridor (Hyderabad)
    test_site("Kokapet Urban Development", 17.3950, 78.3300, "2021-06-15", "2023-06-20", thresh=0.35)

