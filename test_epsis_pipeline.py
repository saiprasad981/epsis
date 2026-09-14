"""
test_epsis_pipeline.py — Automated Verification Test Suite for EPSIS
===================================================================
Tests satellite pair fetching, TinyCD inference (LEVIR-CD & WHU-CD),
HiResCAM explainability, Change Classification, Severity Analysis,
and Risk Assessment.
"""

import sys
import os
import numpy as np

from satellite import fetch_satellite_pair
from inference import load_model, predict_change
from epsis_analysis import analyze_detected_changes


def test_epsis_end_to_end():
    print("=" * 70)
    print("       EPSIS END-TO-END SYSTEM INTEGRATION TEST")
    print("=" * 70)

    # Coordinates for active urban growth zone (Hitech City / Kokapet, Hyderabad)
    lat, lon = 17.3950, 78.3300
    ref_start, ref_end = "2021-01-01", "2021-12-31"
    comp_start, comp_end = "2023-01-01", "2023-12-31"

    print(f"\n[TEST 1] Acquiring satellite pair for ({lat}, {lon})...")
    ref_path, comp_path, sat_meta = fetch_satellite_pair(lat, lon, ref_start, ref_end, comp_start, comp_end, buffer_m=1000)
    assert os.path.exists(ref_path), "Reference T1 image file missing!"
    assert os.path.exists(comp_path), "Comparison T2 image file missing!"
    print(f"  -> Provider: {sat_meta.get('provider')}")
    print(f"  -> Reference Path: {ref_path}")
    print(f"  -> Comparison Path: {comp_path}")

    # 2. Test LEVIR-CD Checkpoint Inference
    print("\n[TEST 2] Testing TinyCD Model Inference (LEVIR-CD checkpoint)...")
    levir_model_path = "Tiny_model_4_CD/pretrained_models/levir_best.pth"
    levir_model = load_model(levir_model_path)
    res_levir = predict_change(levir_model, ref_path, comp_path, threshold=0.40)

    prob_map = res_levir["probability_map"]
    mask = res_levir["binary_mask"]
    print(f"  -> Probability Range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
    print(f"  -> Changed Pixels: {mask.sum()} / {mask.size} ({res_levir['changed_fraction']*100:.2f}%)")
    print(f"  -> Overlay Shape: {res_levir['overlay_rgb'].shape}")
    print(f"  -> HiResCAM Shape: {res_levir['hirescam_rgb'].shape}")

    assert prob_map.shape == (256, 256), "Probability map shape invalid!"
    assert res_levir["hirescam_rgb"].shape == (256, 256, 3), "HiResCAM shape invalid!"

    # 3. Test WHU-CD Checkpoint Inference
    print("\n[TEST 3] Testing TinyCD Model Inference (WHU-CD checkpoint key remapping)...")
    whu_model_path = "Tiny_model_4_CD/pretrained_models/whu_best.pth"
    whu_model = load_model(whu_model_path)
    res_whu = predict_change(whu_model, ref_path, comp_path, threshold=0.40)
    print(f"  -> WHU Changed Pixels: {res_whu['binary_mask'].sum()} ({res_whu['changed_fraction']*100:.2f}%)")

    # 4. Test Intelligence Analysis Engine
    print("\n[TEST 4] Testing EPSIS Classification, Severity, & Risk Engine...")
    analysis = analyze_detected_changes(ref_path, comp_path, prob_map, mask, lat, lon)

    cls_res = analysis["classification"]
    sev_res = analysis["severity"]
    risk_res = analysis["risk"]

    print(f"  -> Primary Class: {cls_res['primary_class']} (Confidence: {cls_res['confidence']*100:.1f}%)")
    print(f"  -> Severity Score: {sev_res['score']} / 100 ({sev_res['level']} Severity)")
    print(f"  -> Risk Rating: {risk_res['risk_title']}")
    print(f"  -> Region Count: {analysis['num_regions']}")

    assert "primary_class" in cls_res, "Classification result invalid!"
    assert 0.0 <= sev_res["score"] <= 100.0, "Severity score out of bounds!"

    print("\n" + "=" * 70)
    print("    [SUCCESS] ALL EPSIS SYSTEM INTEGRATION TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_epsis_end_to_end()
