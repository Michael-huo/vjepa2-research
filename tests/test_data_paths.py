from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research.scripts.common import data_paths
from research.scripts.common.phase2_data import (
    DEFAULT_SEQUENCES,
    inspect_davis_dataset,
    prepare_davis_dataset,
)
from research.scripts.common.phase3_efficiency import DAVIS_ROOT as PHASE3_DAVIS_ROOT
from research.scripts.common.runtime import require_sample_video


class DataPathsTests(unittest.TestCase):
    def test_roots_and_fixed_paths_are_repo_relative(self) -> None:
        self.assertEqual(data_paths.RESEARCH_ROOT, data_paths.REPO_ROOT / "research")
        self.assertEqual(data_paths.REPO_ROOT, Path(__file__).resolve().parents[1])
        self.assertEqual(
            data_paths.BOWLING_VIDEO_PATH.relative_to(data_paths.REPO_ROOT).as_posix(),
            "research/assets/datasets/bowling/sample_bowling.mp4",
        )
        self.assertEqual(
            data_paths.DAVIS_ROOT.relative_to(data_paths.REPO_ROOT).as_posix(),
            "research/assets/datasets/davis2017/DAVIS",
        )
        self.assertEqual(
            data_paths.DAVIS_MANIFEST_PATH.relative_to(data_paths.REPO_ROOT).as_posix(),
            "research/assets/prepared/davis2017/manifest.json",
        )

    def test_phase2_and_phase3_share_the_canonical_davis_path(self) -> None:
        from research.scripts.common.phase2_data import DAVIS_ROOT as phase2_davis_root

        self.assertIs(phase2_davis_root, data_paths.DAVIS_ROOT)
        self.assertIs(PHASE3_DAVIS_ROOT, data_paths.DAVIS_ROOT)

    def test_paths_do_not_depend_on_current_working_directory(self) -> None:
        expected = data_paths.DAVIS_ROOT
        with tempfile.TemporaryDirectory() as temporary:
            previous_directory = Path.cwd()
            try:
                os.chdir(temporary)
                self.assertEqual(data_paths.DAVIS_ROOT, expected)
                self.assertEqual(
                    data_paths.BOWLING_VIDEO_PATH,
                    data_paths.DATASETS_ROOT / "bowling" / "sample_bowling.mp4",
                )
            finally:
                os.chdir(previous_directory)

    def test_import_is_side_effect_free_when_dataset_directories_are_absent_or_present(self) -> None:
        datasets_dir = data_paths.DATASETS_ROOT
        prepared_dir = data_paths.PREPARED_ROOT
        before = (datasets_dir.exists(), prepared_dir.exists())
        code = (
            "from pathlib import Path\n"
            f"datasets = Path({str(datasets_dir)!r})\n"
            f"prepared = Path({str(prepared_dir)!r})\n"
            "before = (datasets.exists(), prepared.exists())\n"
            "import research.scripts.common.data_paths\n"
            "assert before == (datasets.exists(), prepared.exists())\n"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(data_paths.REPO_ROOT) + os.pathsep + environment.get(
            "PYTHONPATH", ""
        )
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run(
                [sys.executable, "-c", code],
                cwd=temporary,
                env=environment,
                check=True,
            )
        self.assertEqual(before, (datasets_dir.exists(), prepared_dir.exists()))

    def test_prepare_manifest_is_portable_atomic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as first_temporary, tempfile.TemporaryDirectory() as second_temporary:
            first_root, first_manifest = self._build_minimal_davis_fixture(Path(first_temporary))
            second_root, second_manifest = self._build_minimal_davis_fixture(Path(second_temporary))

            first = prepare_davis_dataset(davis_root=first_root, manifest_path=first_manifest)
            self.assertFalse(first.skipped)
            self.assertTrue(first_manifest.is_file())
            serialized = json.dumps(first.manifest, sort_keys=True)
            self.assertNotIn(first_temporary, serialized)
            self.assertEqual(first.manifest["davis_root"], "DAVIS")
            self.assertTrue(first.manifest["jpeg_root"].startswith("DAVIS/"))
            self.assertEqual(list(first_manifest.parent.glob(".*.tmp")), [])

            reused = prepare_davis_dataset(davis_root=first_root, manifest_path=first_manifest)
            self.assertTrue(reused.skipped)
            self.assertEqual(first_manifest.read_text(encoding="utf-8"), json.dumps(first.manifest, ensure_ascii=False, indent=2))

            second = prepare_davis_dataset(davis_root=second_root, manifest_path=second_manifest)
            self.assertEqual(first.manifest["structure_fingerprint"], second.manifest["structure_fingerprint"])
            self.assertNotIn(second_temporary, json.dumps(second.manifest, sort_keys=True))

    def test_missing_inputs_explain_the_standard_preparation_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-davis"
            with self.assertRaises(FileNotFoundError) as davis_error:
                inspect_davis_dataset(davis_root=missing)
            self.assertIn("research/assets/datasets/davis2017", str(davis_error.exception))

            with self.assertRaises(FileNotFoundError) as bowling_error:
                require_sample_video(Path(temporary) / "missing-bowling.mp4")
            self.assertIn("research/assets/datasets/bowling/sample_bowling.mp4", str(bowling_error.exception))

    @staticmethod
    def _build_minimal_davis_fixture(parent: Path) -> tuple[Path, Path]:
        davis_root = parent / "fixture" / "DAVIS"
        (davis_root / "ImageSets" / "2017").mkdir(parents=True)
        for sequence in DEFAULT_SEQUENCES:
            image = davis_root / "JPEGImages" / "480p" / sequence / "00000.jpg"
            mask = davis_root / "Annotations" / "480p" / sequence / "00000.png"
            image.parent.mkdir(parents=True)
            mask.parent.mkdir(parents=True)
            image.write_bytes(b"jpeg-fixture")
            mask.write_bytes(b"png-fixture")
            os.utime(image, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
            os.utime(mask, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        return davis_root, parent / "prepared" / "manifest.json"


if __name__ == "__main__":
    unittest.main()
