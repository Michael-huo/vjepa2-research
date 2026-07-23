"""Single entry point for the Phase 4 TUM RGB-D feasibility validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed Phase 4 time-horizon, budgeted, and causal TUM RGB-D feasibility validation."
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only validate/reuse the fixed TUM RGB-D prepared manifest. Does not require CUDA.",
    )
    return parser


def run_prepare_only() -> None:
    from research.scripts.common.phase4_feasibility import prepare_tum_sequence, print_prepare_summary

    print("[Phase 4 / Feasibility] Preparing fixed TUM RGB-D inputs")
    prepared = prepare_tum_sequence()
    status = "reused" if prepared.skipped else "wrote"
    print(f"      prepared manifest {status}: {prepared.manifest['manifest_path']}")
    print_prepare_summary(prepared)
    print("[Done] Prepare-only check complete.")


def main() -> None:
    args = build_parser().parse_args()
    if args.prepare_only:
        run_prepare_only()
        return
    from research.scripts.common.phase4_feasibility import run_phase4

    run_phase4()


if __name__ == "__main__":
    main()
