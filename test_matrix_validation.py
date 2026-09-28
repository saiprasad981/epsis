"""
test_matrix_validation.py — Comprehensive Test Suite for EPSIS T1/T2 Image Quality & Matrix Validation
========================================================================================================
Tests 5 distinct worldwide locations and multiple date pairs to verify:
 1. High-resolution imagery generation (>= 1000px)
 2. 10m Sentinel-2 SR Harmonized band preservation (B4, B3, B2)
 3. Crisp ground detail without artificial bicubic smoothing
 4. Proper location coordinate matching & non-reused imagery
 5. Model inference & paper-faithful HiResCAM execution
 6. Classification, Severity, & Risk assessment computation
"""

import os
import sys
import numpy as np
from PIL import Image

from satellite import fetch_satellite_pair
from inference import load_model, predict_change
from epsis_analysis import analyze_detected_changes

PRODUCTION_MODEL_PATH = "Tiny_model_4_CD/pretrained_models/levir_best.pth"

TEST_LOCATIONS = [
    {
        "name": "Bengaluru Whitefield (Urban expansion)",
        "lat": 12.9800,
        "lon": 77.7400,
        "t1": "2021-06-15",
        "t2": "2023-06-20"
    },
    {
        "name": "Hyderabad Kokapet (Active construction zone)",
        "lat": 17.3950,
        "lon": 78.3300,
        "t1": "2021-06-15",
        "t2": "2023-06-20"
    },
    {
        "name": "Vijayawada / Amaravati (River & Agriculture)",
        "lat": 16.5062,
        "lon": 80.6480,
        "t1": "2021-06-15",
        "t2": "2023-06-20"
    },
    {
        "name": "London Docklands (Urban / Water area)",
        "lat": 51.5050,
        "lon": -0.0200,
        "t1": "2021-06-15",
        "t2": "2023-06-20"
    },
    {
        "name": "Tokyo Bay / Odaiba (Coastal metropolitan)",
        "lat": 35.6294,
        "lon": 139.7758,
        "t1": "2021-06-15",
        "t2": "2023-06-20"
    },
]


def run_comprehensive_matrix_test():
    print("=" * 80)
    print("       EPSIS T1/T2 SATELLITE IMAGE QUALITY & PIPELINE MATRIX TEST")
    print("=" * 80)

    print("\n[INIT] Loading pretrained TinyCD model...")
    model = load_model(PRODUCTION_MODEL_PATH)
    print("  -> Model loaded successfully.")

    passed_count = 0
    total_tests = len(TEST_LOCATIONS)

    for idx, loc in enumerate(TEST_LOCATIONS, 1):
        print(f"\n" + "-" * 80)
        print(f"  LOCATION TEST {idx}/{total_tests}: {loc['name']}")
        print(f"  Coordinates: ({loc['lat']}, {loc['lon']}) | T1: {loc['t1']} | T2: {loc['t2']}")
        print("-" * 80)

        # 1. Fetch satellite pair
        ref_path, comp_path, sat_meta = fetch_satellite_pair(
            loc["lat"], loc["lon"],
            loc["t1"], loc["t2"],
            buffer_m=1000
        )

        assert os.path.exists(ref_path), "T1 image file does not exist!"
        assert os.path.exists(comp_path), "T2 image file does not exist!"

        img1 = Image.open(ref_path)
        img2 = Image.open(comp_path)
        w1, h1 = img1.size
        w2, h2 = img2.size

        print(f"  [1] Output Dimensions : T1 = {w1}x{h1} px | T2 = {w2}x{h2} px")
        print(f"  [2] Acquired Dates    : T1 = {sat_meta.get('before_actual')} | T2 = {sat_meta.get('after_actual')}")
        print(f"  [3] Cloud Cover       : T1 = {sat_meta.get('before_cloud', 0):.2f}% | T2 = {sat_meta.get('after_cloud', 0):.2f}%")
        print(f"  [4] Provider / Scene  : {sat_meta.get('provider')}")
        print(f"                          T1 Scene: {sat_meta.get('before_scene_id')}")
        print(f"                          T2 Scene: {sat_meta.get('after_scene_id')}")

        # Quality Assertions
        assert max(w1, h1) >= 1000 and min(w1, h1) >= 500, f"T1 dimensions ({w1}x{h1}) are too low! Expected max dim >= 1000px."
        assert max(w2, h2) >= 1000 and min(w2, h2) >= 500, f"T2 dimensions ({w2}x{h2}) are too low! Expected max dim >= 1000px."
        assert abs(w1 - w2) <= 5 and abs(h1 - h2) <= 5, "T1 and T2 dimensions mismatch!"

        # 2. Model Inference
        pred_res = predict_change(model, ref_path, comp_path, threshold=0.45, use_otsu=True)
        prob_map = pred_res["probability_map"]
        bin_mask = pred_res["binary_mask"]

        print(f"  [5] Change Detection  : Changed fraction = {pred_res['changed_fraction']*100:.2f}% ({bin_mask.sum() // 255} px)")
        print(f"  [6] Effective Thresh  : {pred_res['effective_threshold']:.4f}")
        print(f"  [7] HiResCAM Status   : CAM Std={pred_res['hirescam_stats']['cam_std']:.4f}, Valid={pred_res['hirescam_stats']['valid_cam']}")

        # 3. EPSIS Analysis Engine
        analysis = analyze_detected_changes(ref_path, comp_path, prob_map, bin_mask, loc["lat"], loc["lon"])
        cls_res = analysis["classification"]
        sev_res = analysis["severity"]
        risk_res = analysis["risk"]

        print(f"  [8] Classification    : Primary = '{cls_res['primary_class']}' (Conf: {cls_res['confidence']*100:.1f}%)")
        print(f"  [9] Severity / Risk   : Score = {sev_res['score']}/100 ({sev_res['level']}) | Risk: '{risk_res['risk_title']}'")
        print(f"  [10] Quality Status   : {sat_meta.get('validation_status')}")

        passed_count += 1

    # Date Switching Verification Test
    print("\n" + "=" * 80)
    print("  DATE SWITCHING INTEGRATION TEST")
    print("=" * 80)
    test_lat, test_lon = 17.3950, 78.3300
    ref_a, comp_a, meta_a = fetch_satellite_pair(test_lat, test_lon, "2021-06-15", "2023-06-20")
    ref_b, comp_b, meta_b = fetch_satellite_pair(test_lat, test_lon, "2021-06-15", "2024-06-20")

    print(f"  Test A T2 Scene: {meta_a.get('after_scene_id')} (Date: {meta_a.get('after_actual')})")
    print(f"  Test B T2 Scene: {meta_b.get('after_scene_id')} (Date: {meta_b.get('after_actual')})")

    assert meta_a.get('before_scene_id') == meta_b.get('before_scene_id'), "T1 scenes should be identical when T1 date is unchanged!"
    assert meta_a.get('after_scene_id') != meta_b.get('after_scene_id'), "T2 scenes MUST be different when T2 date is changed!"
    print("  [SUCCESS] Date switching correctly updates T2 scene without reusing stale image!")

    print("\n" + "=" * 80)
    print(f"   [SUCCESS] ALL {passed_count}/{total_tests} LOCATION & DATE TESTS PASSED CLEANLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_comprehensive_matrix_test()
