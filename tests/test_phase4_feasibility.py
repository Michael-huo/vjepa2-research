from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from research.scripts.common.phase4_feasibility import (
    ADAPTIVE_TARGET_OBSERVATION_RATE,
    EXPECTED_OUTPUT_FILES,
    FLOAT_TIE_TOLERANCE,
    MAX_PAIRS_PER_HORIZON,
    OBSERVATION_BUDGETS,
    SEQUENCE_NAME,
    AssociatedFrame,
    PoseEntry,
    PreparedTumSequence,
    RgbEntry,
    _time_pair_records,
    adjacent_reference_high_error_threshold,
    artifact_records,
    associate_rgb_to_poses,
    benchmark_latent_decision,
    benchmark_pixel_decision,
    break_even_refresh_rate,
    budget_curve_auc,
    build_feasibility_verdict,
    build_resolution_rgb,
    build_run_manifest,
    build_source_views,
    calibrate_causal_thresholds,
    causal_anchor_schedule,
    causal_score_function,
    cumulative_motion_prefix,
    deterministic_indices,
    estimate_policy_latency,
    experiment2_selection_summary,
    experiment3_selection_summary,
    find_time_horizon_pairs,
    frame_support_durations,
    global_latents,
    half_up_target_count,
    high_error_episode_metrics,
    hold_last_state_metrics,
    large_motion_coverage,
    latent_distance,
    motion_between,
    normalize_quaternion,
    offline_adjacent_scores,
    offline_topk_schedule,
    operation_dominates,
    parse_groundtruth_file,
    parse_rgb_file,
    pixel_distance,
    pose_staleness_metrics,
    prepare_tum_sequence,
    publish_staged_output,
    rotation_delta_degrees,
    run_adaptive_observation,
    run_ego_motion_consistency,
    run_sparse_state_refresh,
    scout_reference_consistency,
    source_crop_box,
    spearman_correlation,
    split_time_segments,
    timing_statistics,
    translation_delta,
    uniform_time_schedule,
    validate_associated_frames,
    validate_casefold_unique_paths,
    validate_expected_output,
    validate_manifest_paths,
)


def pose(timestamp: float, x: float = 0.0, yaw_degrees: float = 0.0) -> PoseEntry:
    radians = np.radians(yaw_degrees)
    return PoseEntry(timestamp, (x, 0.0, 0.0), (0.0, 0.0, float(np.sin(radians / 2)), float(np.cos(radians / 2))))


def frame(index: int, timestamp: float, *, x: float = 0.0, yaw_degrees: float = 0.0) -> AssociatedFrame:
    item = pose(timestamp, x=x, yaw_degrees=yaw_degrees)
    return AssociatedFrame(index, timestamp, timestamp, 0.0, f"rgb/{index}.png", Path(f"{index}.png"), item.translation, item.quaternion_xyzw)


class TumPreparationTests(unittest.TestCase):
    def test_parsing_comments_sorting_and_quaternion_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rgb").mkdir()
            (root / "rgb.txt").write_text("# c\n\n2 rgb/b.png\n1 rgb/a.png\n", encoding="utf-8")
            (root / "groundtruth.txt").write_text("2 0 0 0 0 0 0 2\n1 0 0 0 0 0 0 1\n", encoding="utf-8")
            self.assertEqual([item.timestamp for item in parse_rgb_file(root / "rgb.txt", dataset_root=root)], [1.0, 2.0])
            parsed = parse_groundtruth_file(root / "groundtruth.txt")
            self.assertEqual(parsed[-1].quaternion_xyzw, (0.0, 0.0, 0.0, 1.0))

    def test_duplicate_timestamps_and_unsafe_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rgb").mkdir()
            (root / "rgb.txt").write_text("1 rgb/a.png\n1 rgb/b.png\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate RGB"):
                parse_rgb_file(root / "rgb.txt", dataset_root=root)
            (root / "groundtruth.txt").write_text("1 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate GT"):
                parse_groundtruth_file(root / "groundtruth.txt")
            for unsafe in ("/tmp/a.png", "rgb/../../a.png", "other/a.png"):
                (root / "rgb.txt").write_text(f"1 {unsafe}\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Unsafe|inside rgb"):
                    parse_rgb_file(root / "rgb.txt", dataset_root=root)

    def test_association_boundary_tie_and_monotonic_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rgb").mkdir()
            for name in ("a.png", "b.png", "c.png"):
                (root / "rgb" / name).write_bytes(b"x")
            rgb = [RgbEntry(1.0, "rgb/a.png"), RgbEntry(2.0, "rgb/b.png"), RgbEntry(3.0, "rgb/c.png")]
            associated, valid = associate_rgb_to_poses(rgb, [pose(0.99), pose(1.01), pose(2.02), pose(3.021)], dataset_root=root)
            self.assertEqual((valid, len(associated), associated[0].gt_timestamp), (3, 2, 0.99))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                validate_associated_frames([associated[0], AssociatedFrame(1, associated[0].rgb_timestamp, 1, 0, "rgb/b.png", Path("b"), (0, 0, 0), (0, 0, 0, 1))])

    def test_zero_quaternion_and_portable_reused_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            normalize_quaternion((0, 0, 0, 0))
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / SEQUENCE_NAME
            manifest = parent / "prepared" / "manifest.json"
            (root / "rgb").mkdir(parents=True)
            for index, timestamp in enumerate((1.0, 1.1)):
                Image.new("RGB", (8, 6), (index * 20, 0, 0)).save(root / "rgb" / f"{index}.png")
            (root / "rgb.txt").write_text("1.0 rgb/0.png\n1.1 rgb/1.png\n", encoding="utf-8")
            (root / "groundtruth.txt").write_text("1.0 0 0 0 0 0 0 1\n1.1 1 0 0 0 0 0 1\n", encoding="utf-8")
            first = prepare_tum_sequence(dataset_root=root, manifest_path=manifest)
            mtime = manifest.stat().st_mtime_ns
            time.sleep(0.001)
            second = prepare_tum_sequence(dataset_root=root, manifest_path=manifest)
            self.assertFalse(first.skipped)
            self.assertTrue(second.skipped)
            self.assertEqual(mtime, manifest.stat().st_mtime_ns)
            self.assertNotIn(temporary, manifest.read_text(encoding="utf-8"))


class RepresentationAndMotionTests(unittest.TestCase):
    def test_one_source_crop_serves_384_512_and_gray(self) -> None:
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(8, dtype=np.uint8)[None, :]
        views = build_source_views(image)
        rgb384, crop384 = build_resolution_rgb(image, 384)
        rgb512, crop512 = build_resolution_rgb(image, 512)
        self.assertEqual(source_crop_box(8, 4), (2, 0, 6, 4))
        self.assertEqual(crop384, crop512)
        self.assertEqual(crop384, views.source_crop_box_xyxy)
        np.testing.assert_array_equal(rgb384, views.rgb384)
        np.testing.assert_array_equal(rgb512, views.rgb512)
        np.testing.assert_array_equal(views.gray384, cv2.cvtColor(views.rgb384, cv2.COLOR_RGB2GRAY))

    def test_global_latent_and_scout_reference_consistency(self) -> None:
        dense = np.array([[[3.0, 4.0]], [[0.0, 5.0]]], dtype=np.float32)
        latents = global_latents(dense)
        np.testing.assert_allclose(latents, [[0.6, 0.8], [0.0, 1.0]], atol=1e-6)
        self.assertAlmostEqual(latent_distance(latents[0], latents[1]), 0.2)
        records = [{"jepa384_distance": value, "jepa512_distance": value * 2} for value in (0.1, 0.2, 0.3)]
        result = scout_reference_consistency(records)
        self.assertAlmostEqual(result["distance_spearman"]["value"], 1.0)
        self.assertAlmostEqual(result["mean_absolute_distance_difference"], 0.2)

    def test_net_and_cumulative_motion(self) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        quaternions = np.array([pose(0, yaw_degrees=value).quaternion_xyzw for value in (0, 180, 360)])
        translation_prefix, rotation_prefix = cumulative_motion_prefix(positions, quaternions)
        motion = motion_between(0, 2, positions, quaternions, translation_prefix, rotation_prefix)
        self.assertAlmostEqual(motion["net_translation_m"], 0.0)
        self.assertAlmostEqual(motion["cumulative_translation_m"], 2.0)
        self.assertAlmostEqual(motion["net_rotation_deg"], 0.0, places=6)
        self.assertAlmostEqual(motion["cumulative_rotation_deg"], 360.0, places=6)
        self.assertAlmostEqual(translation_delta((0, 0, 0), (3, 4, 0)), 5.0)
        self.assertAlmostEqual(rotation_delta_degrees(pose(0).quaternion_xyzw, pose(1, yaw_degrees=90).quaternion_xyzw), 90.0)

    def test_pose_staleness_and_large_motion(self) -> None:
        positions = np.arange(12, dtype=float).reshape(4, 3)
        quaternions = np.array([[0, 0, 0, 1]] * 4, dtype=float)
        observed = np.array([True, False, True, False])
        metrics = pose_staleness_metrics(observed, positions, quaternions)
        self.assertIn("cumulative_translation_m", metrics["staleness_summary"])
        self.assertEqual(len(metrics["coverage_gaps"]), 1)
        self.assertEqual(large_motion_coverage(observed, positions, quaternions)["translation"]["recall"], 1.0)


class TimeHorizonTests(unittest.TestCase):
    def test_nearest_strictly_future_pair_and_tolerance(self) -> None:
        timestamps = [0.0, 0.08, 0.21, 0.31, 0.55]
        result = find_time_horizon_pairs(timestamps, 0.25, tolerance_seconds=0.04)
        self.assertIn((0, 2), result["pairs"])
        self.assertTrue(all(right > left for left, right in result["pairs"]))
        rejected = find_time_horizon_pairs(timestamps, 0.40, tolerance_seconds=0.001)
        self.assertGreater(rejected["tolerance_discarded_count"], 0)

    def test_pairs_are_unique_and_deterministically_capped(self) -> None:
        timestamps = np.arange(700, dtype=float) * 0.1
        first = find_time_horizon_pairs(timestamps, 0.1)
        second = find_time_horizon_pairs(timestamps, 0.1)
        self.assertEqual(first["pairs"], second["pairs"])
        self.assertEqual(len(first["pairs"]), MAX_PAIRS_PER_HORIZON)
        self.assertEqual(len(first["pairs"]), len(set(first["pairs"])))

    def test_pair_records_contain_actual_time_and_both_state_scales(self) -> None:
        frames = [frame(index, value, x=value) for index, value in enumerate((0.0, 0.1, 0.26, 0.5, 1.0, 2.0, 4.0, 8.0, 8.1))]
        gray = np.stack([np.full((4, 4), index, np.uint8) for index in range(len(frames))])
        angles = np.linspace(0, 0.8, len(frames))
        z384 = np.stack([np.array([np.cos(value), np.sin(value)]) for value in angles])
        z512 = np.stack([np.array([np.cos(value * 1.1), np.sin(value * 1.1)]) for value in angles])
        records, counts = _time_pair_records(frames, gray, z384, z512)
        self.assertTrue(records)
        row = records[0]
        self.assertAlmostEqual(row["actual_time_delta_seconds"], row["timestamp_j"] - row["timestamp_i"])
        self.assertAlmostEqual(row["jepa384_similarity"], 1 - row["jepa384_distance"])
        self.assertAlmostEqual(row["jepa512_similarity"], 1 - row["jepa512_distance"])
        self.assertIn("0.1", counts)


class FixedBudgetTests(unittest.TestCase):
    def test_half_up_count_rounding(self) -> None:
        self.assertEqual(half_up_target_count(0.24, 10), 2)
        self.assertEqual(half_up_target_count(0.25, 10), 3)
        self.assertEqual(half_up_target_count(0.26, 10), 3)
        self.assertEqual(half_up_target_count(0.01, 3), 1)
        self.assertEqual(OBSERVATION_BUDGETS, (0.05, 0.10, 0.20, 0.30, 0.40))

    def test_uniform_time_exact_count_with_duplicate_nearest_targets(self) -> None:
        timestamps = [0.0, 0.01, 0.02, 9.9, 10.0]
        observed = uniform_time_schedule(timestamps, 4)
        self.assertTrue(observed[0])
        self.assertEqual(int(observed.sum()), 4)
        np.testing.assert_array_equal(observed, uniform_time_schedule(timestamps, 4))

    def test_adjacent_score_interfaces_and_topk_ties(self) -> None:
        gray = np.stack([np.full((3, 3), value, np.uint8) for value in (0, 5, 5, 20)])
        latents = np.array([[1, 0], [0.9, 0.1], [0.9, 0.1], [0, 1]], dtype=float)
        pixel = offline_adjacent_scores("pixel", gray, latents)
        flow = offline_adjacent_scores("flow", gray, latents)
        jepa = offline_adjacent_scores("jepa", gray, latents)
        self.assertEqual(len(pixel), len(flow))
        self.assertEqual(len(jepa), 3)
        observed = offline_topk_schedule([1.0, 1.0, 0.5], [0.0, 1.0, 2.0, 3.0], 2)
        self.assertEqual(np.flatnonzero(observed).tolist(), [0, 1])

    def test_hold_last_intervals_and_high_error_threshold(self) -> None:
        latents = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
        metrics = hold_last_state_metrics(np.array([True, False, True]), latents, [0.0, 0.2, 0.7])
        self.assertAlmostEqual(metrics["mean_state_error"], 1 / 3)
        self.assertAlmostEqual(metrics["refresh_interval_seconds"]["mean"], 0.7)
        self.assertAlmostEqual(adjacent_reference_high_error_threshold(latents), 1.0)

    def test_episode_time_support_irregular_single_multi_separated_and_final(self) -> None:
        timestamps = [0.0, 0.1, 0.4, 1.0]
        support = frame_support_durations(timestamps)
        np.testing.assert_allclose(support["durations_seconds"], [0.1, 0.3, 0.6, 0.3])
        single = high_error_episode_metrics([0, 2, 0, 0], timestamps, 1)
        self.assertAlmostEqual(single["episodes"][0]["duration_seconds"], 0.3)
        multi = high_error_episode_metrics([2, 2, 0, 2], timestamps, 1)
        self.assertEqual(multi["episode_count"], 2)
        self.assertAlmostEqual(multi["episodes"][0]["duration_seconds"], 0.4)
        self.assertAlmostEqual(multi["episodes"][1]["duration_seconds"], 0.3)
        degenerate = frame_support_durations([2.0])
        self.assertEqual(degenerate["status"], "degenerate")

    def test_error_and_similarity_auc_use_trapezoid(self) -> None:
        with mock.patch.object(np, "trapezoid", wraps=np.trapezoid) as trapezoid:
            lower_error = budget_curve_auc(OBSERVATION_BUDGETS, [0.3, 0.2, 0.1, 0.08, 0.05])
            higher_error = budget_curve_auc(OBSERVATION_BUDGETS, [0.4, 0.3, 0.2, 0.18, 0.15])
            similarity = budget_curve_auc(OBSERVATION_BUDGETS, [0.7, 0.8, 0.9, 0.92, 0.95])
        self.assertEqual(trapezoid.call_count, 3)
        self.assertLess(lower_error, higher_error)
        self.assertGreater(similarity, 0.8)

    def test_winner_sets_tolerance_sparse_points_and_pass(self) -> None:
        operations = []
        for budget in OBSERVATION_BUDGETS:
            for method, error in (("uniform", 0.2), ("pixel", 0.15), ("flow", 0.16), ("jepa", 0.15 + FLOAT_TIE_TOLERANCE / 2)):
                operations.append({"target_observation_budget": budget, "method": method, "mean_state_error": error, "p95_state_error": error})
        aucs = {method: {"budget_error_auc": value} for method, value in (("uniform", 0.2), ("pixel", 0.15), ("flow", 0.16), ("jepa", 0.15 + FLOAT_TIE_TOLERANCE / 2))}
        result = experiment2_selection_summary(operations, aucs)
        self.assertTrue(result["experiment2_pass"])
        self.assertTrue(result["win_at_10_percent"])
        self.assertTrue(result["win_at_20_percent"])
        self.assertIn("jepa", result["budget_winners"][0]["mean_error_winner_set"])


class CausalCalibrationTests(unittest.TestCase):
    def test_time_split_boundary_and_independent_evaluation(self) -> None:
        split = split_time_segments([0.0, 1.0, 2.0, 3.0, 10.0])
        self.assertEqual(split["split_timestamp"], 2.0)
        self.assertEqual(split["calibration_indices"].tolist(), [0, 1, 2])
        self.assertEqual(split["evaluation_indices"].tolist(), [3, 4])
        observed, reasons, _ = causal_anchor_schedule([3.0, 4.0], lambda index, anchor: 0.0, 0.1, 0.2)
        self.assertTrue(observed[0])
        self.assertEqual(reasons[0], "initial")

    def test_second_based_stable_normal_dynamic_scheduler(self) -> None:
        scores = {(1, 0): 0.1, (2, 0): 0.1, (3, 2): 0.5, (4, 3): 0.9}
        observed, reasons, _ = causal_anchor_schedule([0.0, 1.9, 2.0, 2.5, 2.6], lambda index, anchor: scores[(index, anchor)], 0.2, 0.8)
        self.assertEqual(reasons[1], "skip")
        self.assertEqual(reasons[2], "stable_timeout")
        self.assertEqual(reasons[3], "normal_timeout")
        self.assertEqual(reasons[4], "dynamic")
        self.assertTrue(observed[4])

    def test_calibration_degenerate_scores_and_duplicate_threshold_pairs(self) -> None:
        unavailable = calibrate_causal_thresholds([0.0, 0.1], [1.0], lambda index, anchor: 1.0)
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertEqual(unavailable["unique_score_count"], 1)
        scorer = lambda index, anchor: float(index - anchor)
        result = calibrate_causal_thresholds([0, 0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1], scorer)
        self.assertEqual(result["status"], "available")
        self.assertEqual(len(result["candidates"]), 20)
        self.assertLess(result["unique_threshold_pair_count"], len(result["candidates"]))

    def test_candidate_tie_breaking_is_six_level_deterministic(self) -> None:
        result = calibrate_causal_thresholds([0, 0.5, 1.0, 1.5], [0.1, 0.2, 0.9], lambda index, anchor: [0, 0.1, 0.2, 0.9][index])
        selected = result["selected"]
        expected = min(result["candidates"], key=lambda row: (row["absolute_rate_error"], 0 if row["calibration_observation_rate"] <= ADAPTIVE_TARGET_OBSERVATION_RATE else 1, row["calibration_observation_rate"], -row["high_quantile"], -row["low_quantile"], row["candidate_order"]))
        self.assertEqual(selected, expected)

    def test_offline_adjacent_and_causal_anchor_scores_are_separate(self) -> None:
        gray = np.stack([np.full((2, 2), value, np.uint8) for value in (0, 10, 20)])
        latents = np.array([[1, 0], [0.8, 0.2], [0, 1]], dtype=float)
        adjacent = offline_adjacent_scores("jepa", gray, latents)
        causal = causal_score_function("jepa", gray, latents)(2, 0)
        self.assertEqual(len(adjacent), 2)
        self.assertNotEqual(causal, adjacent[-1])

    def test_matched_uniform_exact_count_and_quality_rule(self) -> None:
        times = np.linspace(0, 10, 20)
        jepa_count = 3
        self.assertEqual(int(uniform_time_schedule(times, jepa_count).sum()), jepa_count)
        jepa = {"observation_rate": 0.10, "mean_state_error": 0.1, "p95_state_error": 0.2}
        matched = {"observation_rate": 0.10, "mean_state_error": 0.11, "p95_state_error": 0.21}
        target = {"observation_rate": 0.10, "mean_state_error": 0.01, "p95_state_error": 0.01}
        pixel = {"observation_rate": 0.11, "mean_state_error": 0.09, "p95_state_error": 0.19}
        result = experiment3_selection_summary(jepa, matched, pixel)
        self.assertTrue(result["conditions"]["matched_uniform_quality"]["passed"])
        self.assertTrue(result["experiment3_pass"])
        target["mean_state_error"] = 0.9
        self.assertTrue(result["conditions"]["matched_uniform_quality"]["passed"])

    def test_pixel_dominance_and_unavailable_fail(self) -> None:
        jepa = {"observation_rate": 0.10, "mean_state_error": 0.1, "p95_state_error": 0.2}
        pixel = {"observation_rate": 0.09, "mean_state_error": 0.1, "p95_state_error": 0.2}
        self.assertTrue(operation_dominates(pixel, jepa))
        self.assertFalse(experiment3_selection_summary(None, None, None)["experiment3_pass"])


class TimingVerdictAndArtifactTests(unittest.TestCase):
    def test_three_experiments_generate_new_named_artifacts_from_synthetic_data(self) -> None:
        count = 60
        timestamps = np.cumsum(np.linspace(0.08, 0.12, count))
        frames = [frame(index, float(timestamp), x=index * 0.01, yaw_degrees=index * 2) for index, timestamp in enumerate(timestamps)]
        gray = np.stack([np.full((12, 12), (index * index + index) % 220, np.uint8) for index in range(count)])
        scout_angles = np.asarray([0.002 * index * index for index in range(count)])
        reference_angles = scout_angles * 1.05
        z384 = np.stack([np.cos(scout_angles), np.sin(scout_angles)], axis=1)
        z512 = np.stack([np.cos(reference_angles), np.sin(reference_angles)], axis=1)
        positions = np.asarray([item.translation for item in frames])
        quaternions = np.asarray([item.quaternion_xyzw for item in frames])
        timing = {
            "pipeline_384": {"median_ms": 4.0},
            "pipeline_512": {"median_ms": 10.0},
            "pixel": {"decision": {"median_ms": 0.2}, "scheduler_fusion": {"median_ms": 0.1}},
            "jepa": {"decision": {"median_ms": 0.05}, "scheduler_fusion": {"median_ms": 0.1}},
            "uniform_scheduler": {"median_ms": 0.05},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("ego", "sparse", "adaptive"):
                (root / directory).mkdir()
            ego = run_ego_motion_consistency(frames=frames, gray_frames=gray, scout_latents=z384, reference_latents=z512, output_dir=root / "ego")
            sparse = run_sparse_state_refresh(timestamps=timestamps, gray_frames=gray, scout_latents=z384, reference_latents=z512, positions=positions, quaternions=quaternions, output_dir=root / "sparse")
            adaptive = run_adaptive_observation(timestamps=timestamps, gray_frames=gray, scout_latents=z384, reference_latents=z512, positions=positions, quaternions=quaternions, timing=timing, output_dir=root / "adaptive")
            self.assertIn("jepa384_distance", ego["horizon_statistics"]["0.1"])
            self.assertIn("budget_error_auc", sparse["auc_by_method"]["jepa"])
            self.assertEqual([row["method"] for row in adaptive["operations"]], ["uniform_target_10", "uniform_matched_jepa", "pixel_adaptive", "jepa_adaptive"])
            self.assertTrue((root / "ego" / "time_horizon_curves.png").is_file())
            self.assertTrue((root / "sparse" / "budget_quality_curves.png").is_file())
            self.assertTrue((root / "adaptive" / "quality_compute_pareto.png").is_file())

    def test_decision_benchmarks_cache_anchors_and_cost_formula(self) -> None:
        frames = [np.zeros((6, 8, 3), dtype=np.uint8), np.full((6, 8, 3), 10, dtype=np.uint8)]
        pixel = benchmark_pixel_decision(frames, low_threshold=0.01, high_threshold=0.1, timestamps=[0.0, 2.1], warmups=0, repeats=1)
        latent = benchmark_latent_decision(np.array([[1, 0], [0.9, 0.1]], dtype=float), low_threshold=0.01, high_threshold=0.2, timestamps=[0.0, 2.1], warmups=0, repeats=1)
        self.assertTrue(pixel["anchor_gray_cached"])
        self.assertTrue(latent["anchor_latent_cached"])
        self.assertEqual(timing_statistics([1, 2, 3])["median_ms"], 2.0)
        estimate = estimate_policy_latency(observation_rate=0.2, pipeline_512_ms=10, scheduler_ms=1, pipeline_384_ms=4, decision_ms=0.5)
        self.assertAlmostEqual(estimate["estimated_latency_ms_per_frame"], 7.5)
        self.assertAlmostEqual(break_even_refresh_rate(4, 10, 1), 0.5)

    def test_go_conditional_go_and_no_go(self) -> None:
        sparse = {"selection_result": {"experiment2_pass": True}}
        adaptive = {"experiment3_pass": True, "selection_conditions": {}, "operations": [{"status": "available", "method": "jepa_adaptive", "estimated_latency_ms_per_frame": 8.0}], "timing": {"pipeline_512": {"median_ms": 10.0}}}
        self.assertEqual(build_feasibility_verdict(sparse, adaptive)["label"], "Go")
        adaptive["operations"][0]["estimated_latency_ms_per_frame"] = 12.0
        self.assertEqual(build_feasibility_verdict(sparse, adaptive)["label"], "Conditional Go")
        sparse["selection_result"]["experiment2_pass"] = False
        self.assertEqual(build_feasibility_verdict(sparse, adaptive)["label"], "No-Go")

    def test_exact_15_file_publication_casefold_and_integrity(self) -> None:
        self.assertEqual(len(EXPECTED_OUTPUT_FILES), 15)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage, final = root / "stage", root / "final"
            stage.mkdir()
            for relative in EXPECTED_OUTPUT_FILES:
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            validate_expected_output(stage)
            entries = artifact_records(stage)
            validate_manifest_paths(stage, entries)
            publish_staged_output(stage, final)
            self.assertEqual({path.relative_to(final).as_posix() for path in final.rglob("*") if path.is_file()}, EXPECTED_OUTPUT_FILES)
        with self.assertRaisesRegex(RuntimeError, "case-insensitively"):
            validate_casefold_unique_paths(["Result.json", "result.JSON"])

    def test_manifest_contains_new_protocol_and_no_legacy_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path = root / "prepared.json"
            prepared_path.write_text("{}", encoding="utf-8")
            prepared = PreparedTumSequence(root, prepared_path, (frame(0, 0.0),), {"dataset_root": "portable", "manifest_path": "prepared.json", "structure_fingerprint": "abc"}, True)
            summary = {"experiment_3": {"policies": {"uniform_target_10": {"observation_count": 3}, "uniform_matched_jepa": {"observation_count": 4}}, "calibrations": {"pixel": {"unique_score_count": 5, "unique_threshold_pair_count": 2}, "jepa": {"unique_score_count": 6, "unique_threshold_pair_count": 3}}}}
            manifest = build_run_manifest(output_dir=root, prepared=prepared, model_metadata={}, environment={}, stage_runtime={}, summary=summary)
            encoded = json.dumps(manifest)
            self.assertIn("half-up", encoded)
            self.assertIn("matched_uniform_rule", encoded)
            self.assertIn("high_error_episode_support", encoded)
            for legacy in ("frame" + "_gaps", "common" + "_range", "matched" + "_rate"):
                self.assertNotIn(legacy, encoded)

    def test_import_and_prepare_help_path_do_not_load_torch(self) -> None:
        code = "import sys; import research.scripts.common.phase4_feasibility; assert 'torch' not in sys.modules"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        subprocess.run([sys.executable, "-c", code], check=True, env=environment)


if __name__ == "__main__":
    unittest.main()
