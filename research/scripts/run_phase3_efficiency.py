from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.scripts.common.phase2_data import (
    DEFAULT_SEQUENCES,
    prepare_davis_dataset,
    repo_relative,
)
from research.scripts.common.data_paths import DAVIS_MANIFEST_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 efficiency and compute-demand evaluation for V-JEPA 2.1 dense DAVIS features."
    )
    parser.add_argument("--full", action="store_true", help="Run the full model/resolution/batch matrix.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only validate/reuse the Phase 2 DAVIS prepared manifest. Does not require CUDA.",
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip quality-efficiency evaluation and write empty/readable quality outputs.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Rewrite the prepared DAVIS manifest even if the structure fingerprint matches.",
    )
    return parser


def run_prepare_only(*, force_prepare: bool) -> None:
    print("[Phase 3 / Efficiency] V-JEPA 2.1 dense feature compute-demand evaluation")
    print("[0/N] Preparing DAVIS dataset")
    prepare_start = time.perf_counter()
    prepared = prepare_davis_dataset(
        manifest_path=DAVIS_MANIFEST_PATH,
        sequences=DEFAULT_SEQUENCES,
        force=force_prepare,
    )
    prepare_seconds = time.perf_counter() - prepare_start
    status = "reused" if prepared.skipped else "wrote"
    print(f"      prepared manifest {status}: {repo_relative(prepared.manifest_path)}")
    for sequence in DEFAULT_SEQUENCES:
        record = prepared.sequences[sequence]
        print(f"      {sequence}: {record.frame_count} frames, {record.mask_count} masks")
    print(f"      elapsed: {prepare_seconds:.2f}s")
    print("[Done] Prepare-only check complete.")
    print("       Full Phase 3 run command: python research/scripts/run_phase3_efficiency.py")


def main() -> None:
    args = build_parser().parse_args()
    if args.prepare_only:
        run_prepare_only(force_prepare=bool(args.force_prepare))
        return

    from research.scripts.common.phase3_efficiency import run_from_args

    run_from_args(args)


if __name__ == "__main__":
    main()
