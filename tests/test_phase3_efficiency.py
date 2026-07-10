from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path

from research.scripts.common.phase3_efficiency import (
    MODEL_REGISTRY,
    build_artifact_index_entries,
    make_config_slug,
    select_quality_efficiency_recommendations,
    validate_casefold_unique_relative_paths,
    validate_casefold_unique_paths,
    validate_manifest_output_paths,
    validate_model_registry,
    validate_successful_quality_artifacts,
)


def quality_row(
    model: str,
    resolution: int,
    quality: object,
    latency: object,
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "status": "success",
        "model": model,
        "resolution": resolution,
        "sampled_crop_j_mean": quality,
        "pipeline_ms_per_frame_median": latency,
    }
    row.update(overrides)
    return row


class Phase3RegistryTests(unittest.TestCase):
    def test_artifact_slugs_are_casefold_unique(self) -> None:
        validate_model_registry()
        slugs = [entry.artifact_slug for entry in MODEL_REGISTRY.values()]
        self.assertEqual(len(slugs), len({slug.casefold() for slug in slugs}))

    def test_giant_artifact_slugs_preserve_logical_ids(self) -> None:
        self.assertEqual(MODEL_REGISTRY["vit_g"].artifact_slug, "vit_g_lc")
        self.assertEqual(MODEL_REGISTRY["vit_G"].artifact_slug, "vit_G_uc")
        self.assertEqual(MODEL_REGISTRY["vit_g"].model_id, "vit_g")
        self.assertEqual(MODEL_REGISTRY["vit_G"].model_id, "vit_G")

    def test_config_slug_generation(self) -> None:
        self.assertEqual(make_config_slug("vit_g", 512), "vit_g_lc_512")
        self.assertEqual(make_config_slug("vit_G", 512), "vit_G_uc_512")


class Phase3ArtifactIntegrityTests(unittest.TestCase):
    def test_case_insensitive_file_collision_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "case-insensitive"):
            validate_casefold_unique_relative_paths(["Result.txt", "result.TXT"])

    def test_case_insensitive_directory_collision_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "case-insensitive"):
            validate_casefold_unique_relative_paths(
                ["selected_outputs/ViT-g", "selected_outputs/vit-G"]
            )

    def test_real_directory_scan_accepts_unique_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "selected_outputs").mkdir()
            (root / "selected_outputs" / "vit_g_lc_512").mkdir()
            validate_casefold_unique_paths(root)

    def test_artifact_index_hashes_actual_small_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"phase-3"
            (root / "metrics.json").write_bytes(payload)
            (root / "manifest.json").write_text("excluded", encoding="utf-8")
            entries = build_artifact_index_entries(root)
            self.assertEqual(
                entries,
                [
                    {
                        "relative_path": "metrics.json",
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            )

    def test_manifest_output_paths_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "present.txt").write_text("ok", encoding="utf-8")
            validate_manifest_output_paths(root, ["present.txt"])
            with self.assertRaisesRegex(RuntimeError, "nonexistent"):
                validate_manifest_output_paths(root, ["missing.txt"])

    def test_no_quality_integrity_accepts_zero_selected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validate_successful_quality_artifacts(root, [])
            self.assertFalse((root / "selected_outputs").exists())


class Phase3RecommendationTests(unittest.TestCase):
    def test_current_like_resolution_curve(self) -> None:
        rows = [
            quality_row("vit_b", 384, 0.7488, 34.7),
            quality_row("vit_b", 512, 0.8206, 38.6),
            quality_row("vit_b", 768, 0.8645, 51.2),
            quality_row("vit_b", 1024, 0.8868, 72.7),
        ]
        result = select_quality_efficiency_recommendations(rows)
        self.assertEqual(result["low_latency"]["config_id"], "vit_b@384")
        self.assertEqual(result["balanced"]["config_id"], "vit_b@512")
        self.assertEqual(result["high_quality"]["config_id"], "vit_b@1024")
        self.assertFalse(result["balanced"]["fallback_used"])
        self.assertEqual(
            result["pareto_configuration_ids"],
            ["vit_b@384", "vit_b@512", "vit_b@768", "vit_b@1024"],
        )

    def test_dominated_configuration_is_not_recommended(self) -> None:
        rows = [
            quality_row("vit_b", 384, 0.75, 30.0),
            quality_row("vit_b", 512, 0.82, 38.0),
            quality_row("vit_l", 512, 0.80, 50.0),
            quality_row("vit_b", 1024, 0.88, 70.0),
        ]
        result = select_quality_efficiency_recommendations(rows)
        self.assertNotIn("vit_l@512", result["pareto_configuration_ids"])
        selected = {
            result[role]["config_id"]
            for role in ("low_latency", "balanced", "high_quality")
        }
        self.assertNotIn("vit_l@512", selected)

    def test_balanced_falls_back_with_explicit_reason(self) -> None:
        rows = [
            quality_row("vit_b", 384, 0.700, 20.0),
            quality_row("vit_b", 512, 0.709, 24.0),
        ]
        result = select_quality_efficiency_recommendations(rows)
        self.assertEqual(
            result["balanced"]["config_id"], result["low_latency"]["config_id"]
        )
        self.assertTrue(result["balanced"]["fallback_used"])
        self.assertIsNone(result["balanced"]["quality_gain_over_low_latency"])
        self.assertIsNone(result["balanced"]["additional_latency_ms_over_low_latency"])
        self.assertIsNone(result["balanced"]["marginal_quality_gain_per_ms"])
        self.assertIn("回退", result["balanced"]["selection_reason"])

    def test_tie_breaking_is_stable_across_input_order(self) -> None:
        rows = [
            quality_row("vit_b", 384, 0.50, 10.0),
            quality_row("vit_l", 768, 0.60, 20.0),
            quality_row("vit_g", 512, 0.60, 20.0),
            quality_row("vit_l", 512, 0.60, 20.0),
        ]
        forward = select_quality_efficiency_recommendations(rows)
        reverse = select_quality_efficiency_recommendations(list(reversed(rows)))
        self.assertEqual(forward, reverse)
        self.assertEqual(forward["low_latency"]["config_id"], "vit_b@384")
        self.assertEqual(forward["balanced"]["config_id"], "vit_l@512")
        self.assertEqual(forward["high_quality"]["config_id"], "vit_l@512")

    def test_large_model_candidate_uses_deterministic_ties(self) -> None:
        rows = [
            quality_row("vit_b", 384, 0.95, 20.0),
            quality_row("vit_G", 512, 0.85, 45.0),
            quality_row("vit_g", 512, 0.85, 40.0),
            quality_row("vit_l", 768, 0.85, 40.0),
            quality_row("vit_l", 512, 0.85, 40.0),
        ]
        result = select_quality_efficiency_recommendations(rows)
        self.assertEqual(result["large_model"]["config_id"], "vit_l@512")
        self.assertNotEqual(result["large_model"]["model"], "vit_b")

    def test_invalid_rows_are_rejected_safely(self) -> None:
        valid = quality_row("vit_b", 384, 0.75, 30.0)
        invalid = [
            quality_row("vit_b", 512, 0.8, 35.0, status="error"),
            quality_row("vit_b", 512, None, 35.0),
            quality_row("vit_b", 512, "", 35.0),
            quality_row("vit_b", 512, "not-a-number", 35.0),
            quality_row("vit_b", 512, math.nan, 35.0),
            quality_row("vit_b", 512, math.inf, 35.0),
            quality_row("vit_b", 512, 0.8, None),
            quality_row("vit_b", 512, 0.8, 0.0),
            quality_row("vit_b", 512, 0.8, -1.0),
            quality_row("vit_b", 512, 0.8, math.nan),
            quality_row("vit_b", 512, 0.8, math.inf),
            quality_row("unknown", 512, 0.8, 35.0),
            quality_row("vit_b", 0, 0.8, 35.0),
            quality_row("vit_b", 512.5, 0.8, 35.0),
            quality_row("vit_b", 512, 0.8, 35.0, batch_size=4),
        ]
        result = select_quality_efficiency_recommendations([valid, *invalid])
        self.assertEqual(result["valid_configuration_ids"], ["vit_b@384"])

    def test_no_valid_rows_returns_structured_unavailable_result(self) -> None:
        result = select_quality_efficiency_recommendations([])
        self.assertTrue(result["method"]["pareto_filter"])
        self.assertEqual(result["valid_configuration_ids"], [])
        self.assertEqual(result["pareto_configuration_ids"], [])
        for key in ("low_latency", "balanced", "high_quality", "large_model"):
            self.assertIsNone(result[key])
        self.assertTrue(result["unavailable_reason"])

    def test_duplicate_rows_are_collapsed_or_rejected(self) -> None:
        row = quality_row("vit_b", 384, 0.75, 30.0)
        result = select_quality_efficiency_recommendations([row, dict(row)])
        self.assertEqual(result["valid_configuration_ids"], ["vit_b@384"])

        for conflicting in (
            quality_row("vit_b", 384, 0.76, 30.0),
            quality_row("vit_b", 384, 0.75, 31.0),
        ):
            with self.subTest(conflicting=conflicting):
                with self.assertRaisesRegex(ValueError, "vit_b@384"):
                    select_quality_efficiency_recommendations([row, conflicting])

    def test_pipeline_fps_falls_back_to_latency_inverse(self) -> None:
        result = select_quality_efficiency_recommendations(
            [quality_row("vit_b", 384, 0.75, 40.0, pipeline_fps="")]
        )
        self.assertAlmostEqual(result["low_latency"]["pipeline_fps"], 25.0)


if __name__ == "__main__":
    unittest.main()
