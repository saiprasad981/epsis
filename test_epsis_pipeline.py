"""
test_epsis_pipeline.py — Automated Verification Test Suite for EPSIS
===================================================================
Tests exact Before/After satellite pair fetching, TinyCD inference (LEVIR-CD & WHU-CD),
Change Classification, Severity Analysis, and Risk Assessment.
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
    before_date = "2021-06-15"
    after_date = "2023-06-20"

    print(f"\n[TEST 1] Acquiring satellite pair for ({lat}, {lon}) on Before: {before_date}, After: {after_date}...")
    ref_path, comp_path, sat_meta = fetch_satellite_pair(lat, lon, before_date, after_date, buffer_m=1000)
    assert os.path.exists(ref_path), "Reference T1 image file missing!"
    assert os.path.exists(comp_path), "Comparison T2 image file missing!"
    print(f"  -> Provider: {sat_meta.get('provider')}")
    print(f"  -> Before Image (Requested: {sat_meta.get('before_requested')}, Acquired: {sat_meta.get('before_actual')})")
    print(f"  -> After Image  (Requested: {sat_meta.get('after_requested')}, Acquired: {sat_meta.get('after_actual')})")
    print(f"  -> Reference Path: {ref_path}")
    print(f"  -> Comparison Path: {comp_path}")

    assert sat_meta.get("before_requested") == before_date, "Metadata before_requested mismatch!"
    assert sat_meta.get("after_requested") == after_date, "Metadata after_requested mismatch!"
    assert sat_meta.get("before_actual") is not None, "Metadata before_actual missing!"
    assert sat_meta.get("after_actual") is not None, "Metadata after_actual missing!"

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

    assert prob_map.ndim == 2, "Probability map shape invalid!"

    # 3. Test WHU-CD Checkpoint Inference
    print("\n[TEST 3] Testing TinyCD Model Inference (WHU-CD checkpoint key remapping)...")
    whu_model_path = "Tiny_model_4_CD/pretrained_models/whu_best.pth"
    whu_model = load_model(whu_model_path)
    res_whu = predict_change(whu_model, ref_path, comp_path, threshold=0.40)
    print(f"  -> WHU Changed Pixels: {res_whu['binary_mask'].sum()} ({res_whu['changed_fraction']*100:.2f}%)")

    # 4. Test Reference Change Analysis Verification Suite
    print("\n[TEST 4] Testing Reference Change Analysis Verification Engine...")
    from inference import compute_reference_change_analysis

    # Identical Images Test (T1 == T2)
    ident_ref = compute_reference_change_analysis(ref_path, ref_path)
    print(f"  -> Identical Image Test (T1==T1) Max Value: {ident_ref.max()}, Mean Value: {ident_ref.mean():.4f}")
    assert ident_ref.max() == 0, "Reference Change Analysis on identical images must return zero / black output!"

    # Different Images Test (T1 != T2)
    diff_ref = res_levir["reference_change_analysis"]
    print(f"  -> Different Image Test (T1!=T2) Min/Max/Mean: [{diff_ref.min()}, {diff_ref.max()}, {diff_ref.mean():.2f}]")
    assert diff_ref.max() > 0, "Reference Change Analysis on different images must detect temporal change!"
    # Ensure continuous grayscale difference spectrum (not binary 0 and 255 only)
    unique_vals = np.unique(diff_ref)
    print(f"  -> Reference Analysis Unique Intensity Levels: {len(unique_vals)} distinct values")
    assert len(unique_vals) > 2, "Reference Change Analysis must produce continuous change gradients, not a binary mask!"

    # 5. Test HiResCAM Attention Color Scale Verification Suite
    print("\n[TEST 5] Testing HiResCAM Attention Color Scale Engine...")
    hirescam_rgb = res_levir["hirescam_rgb"]
    h_stats = res_levir["hirescam_stats"]

    print(f"  -> HiResCAM Image Shape: {hirescam_rgb.shape}")
    print(f"  -> HiResCAM Autograd Layer: {h_stats.get('target_layer')}")
    print(f"  -> Forward Activations Mean / Std: {h_stats.get('act_mean'):.4f} / {h_stats.get('act_std'):.4f}")
    print(f"  -> Backward Gradients Mean / Std : {h_stats.get('grad_mean'):.6f} / {h_stats.get('grad_std'):.6f}")
    print(f"  -> Heatmap Range [Min, Max, Mean]: [{h_stats.get('cam_min'):.4f}, {h_stats.get('cam_max'):.4f}, {h_stats.get('cam_mean'):.4f}]")
    print(f"  -> Heatmap NaN / Inf Count       : {h_stats.get('nan_count')} / {h_stats.get('inf_count')}")

    assert hirescam_rgb.ndim == 3 and hirescam_rgb.shape[2] == 3, "HiResCAM output must be a valid 3-channel RGB image!"
    assert h_stats["nan_count"] == 0 and h_stats["inf_count"] == 0, "HiResCAM heatmap contains invalid NaN/Inf values!"
    assert h_stats["grad_std"] > 0 or h_stats["cam_std"] > 0, "HiResCAM gradients and feature activations must be non-zero!"

    # 6. Test Intelligence Analysis Engine
    print("\n[TEST 6] Testing EPSIS Classification, Severity, & Risk Engine...")
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
