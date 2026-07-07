"""Runtime configuration and filesystem helpers for the Phase 1 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PHASE_NAME = "phase1-probe"
VIDEO_STEM = "sample_bowling"
EXPECTED_OUTPUT_FILES = {
    "features.npz",
    "representation.png",
    "correspondence.png",
    "completion.png",
    "metrics.json",
    "report.md",
    "manifest.json",
}
PHASE_SOURCE_RELATIVE_PATHS = (
    "research/__init__.py",
    "research/scripts/__init__.py",
    "research/scripts/run_phase1_probe.py",
    "research/scripts/common/__init__.py",
    "research/scripts/common/runtime.py",
    "research/scripts/common/video_models.py",
    "research/scripts/common/analysis.py",
    "research/scripts/common/visualization.py",
)

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSETS_DIR = RESEARCH_ROOT / "assets"
DEFAULT_OUTPUTS_DIR = RESEARCH_ROOT / "outputs"
SAMPLE_VIDEO_PATH = DEFAULT_ASSETS_DIR / "sample_bowling.mp4"
PHASE_OUTPUT_ROOT = DEFAULT_OUTPUTS_DIR / PHASE_NAME
FINAL_OUTPUT_DIR = PHASE_OUTPUT_ROOT / VIDEO_STEM
STAGED_OUTPUT_DIR = PHASE_OUTPUT_ROOT / f".{VIDEO_STEM}.tmp"

DOWNLOAD_COMMAND = (
    "mkdir -p research/assets\n"
    "wget -c -O research/assets/sample_bowling.mp4 "
    "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/"
    "bowling/-WH-lxmGJVY_000005_000015.mp4"
)


@dataclass(frozen=True)
class RepresentationConfig:
    num_frames: int = 64
    crop_size: int = 384
    start_token: int = 12
    end_token: int = 19
    selected_tokens: tuple[int, ...] = (12, 14, 16, 19)
    top_temporal_changes: int = 2
    top_local_changes: int = 2


@dataclass(frozen=True)
class CorrespondenceConfig:
    reference_token: int = 12
    start_token: int = 12
    end_token: int = 19
    selected_tokens: tuple[int, ...] = (12, 14, 16, 19)
    roi: tuple[int, int, int, int] = (48, 272, 64, 64)
    memory_frames: int = 3
    top_k_update: int = 12
    spatial_sigma: float = 6.0
    max_jump: float = 11.0
    spatial_weight: float = 0.08
    reference_weight: float = 0.75
    mask_quantile: float = 0.88
    peak_margin: float = 0.08
    min_mask_patches: int = 5
    max_mask_patches: int = 45


@dataclass(frozen=True)
class CompletionConfig:
    target_token: int = 16
    target_row: int = 16
    target_height: int = 6
    target_col: int = 2
    target_width: int = 8
    modes: tuple[str, ...] = ("full", "spatial_only", "temporal_bi", "past_only")


@dataclass(frozen=True)
class PhaseConfig:
    phase_name: str = PHASE_NAME
    video_path: Path = SAMPLE_VIDEO_PATH
    final_output_dir: Path = FINAL_OUTPUT_DIR
    staged_output_dir: Path = STAGED_OUTPUT_DIR
    representation: RepresentationConfig = RepresentationConfig()
    correspondence: CorrespondenceConfig = CorrespondenceConfig()
    completion: CompletionConfig = CompletionConfig()


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Run Phase 1 - V-JEPA 2.1 Capability Probe on the fixed "
            "research/assets/sample_bowling.mp4 sample video."
        )
    )


def relative_to_repo(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def repo_relative_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def require_sample_video(path: Path = SAMPLE_VIDEO_PATH) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            "Fixed Phase 1 input video was not found: "
            f"{relative_to_repo(path)}\n"
            "Download it with:\n"
            f"{DOWNLOAD_COMMAND}"
        )
    return path


def require_cuda() -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Phase 1 requires CUDA. Activate the CUDA-enabled vjepa environment "
            "and verify torch.cuda.is_available()."
        )
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    gpu_name = torch.cuda.get_device_name(0)
    return device, gpu_name


def prepare_staged_output(staged_dir: Path = STAGED_OUTPUT_DIR) -> Path:
    PHASE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if staged_dir.exists():
        print(f"[Setup] Removing stale staged output directory: {relative_to_repo(staged_dir)}")
        shutil.rmtree(staged_dir)
    print("[Setup] Preparing staged Phase 1 output directory")
    staged_dir.mkdir(parents=True, exist_ok=False)
    return staged_dir


def verify_compact_output(output_dir: Path) -> None:
    if not output_dir.is_dir():
        raise RuntimeError(f"Output directory was not created: {output_dir}")
    entries = list(output_dir.iterdir())
    actual = {path.name for path in entries if path.is_file()}
    unexpected_dirs = sorted(path.name for path in entries if path.is_dir())
    missing = EXPECTED_OUTPUT_FILES - actual
    unexpected = actual - EXPECTED_OUTPUT_FILES
    if missing or unexpected or unexpected_dirs:
        raise RuntimeError(
            "Phase 1 output directory must contain exactly seven final files. "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"unexpected_dirs={unexpected_dirs}"
        )


def replace_final_output(staged_dir: Path = STAGED_OUTPUT_DIR, final_dir: Path = FINAL_OUTPUT_DIR) -> None:
    verify_compact_output(staged_dir)
    if final_dir.exists():
        print(f"[Setup] Removing previous Phase 1 outputs: {relative_to_repo(final_dir)}")
        shutil.rmtree(final_dir)
    staged_dir.rename(final_dir)
    print("[Done] Replaced Phase 1 output directory:")
    print(f"       {relative_to_repo(final_dir)}/")


def cleanup_staged_output(staged_dir: Path = STAGED_OUTPUT_DIR) -> None:
    if staged_dir.exists():
        shutil.rmtree(staged_dir)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run_git(args: list[str], *, text: bool = True) -> subprocess.CompletedProcess:
    command = ["git", "-C", str(REPO_ROOT), *args]
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=text,
    )


def git_commit_hash() -> str | None:
    try:
        result = _run_git(["rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def collect_phase_source_fingerprint() -> dict[str, Any]:
    source_records = []
    for relative_path in sorted(PHASE_SOURCE_RELATIVE_PATHS):
        source_path = REPO_ROOT / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Required Phase 1 source file is missing: {relative_path}")
        file_hash = sha256_file(source_path)
        source_records.append({"path": relative_path, "sha256": file_hash})

    combined = bytearray()
    for record in source_records:
        combined.extend(f"{record['path']}\n{record['sha256']}\n".encode("utf-8"))
    return {
        "phase_source_files": source_records,
        "phase_source_fingerprint": sha256_bytes(bytes(combined)),
    }


def _unavailable_git_provenance() -> dict[str, Any]:
    return {
        "git_commit": None,
        "git_provenance_available": False,
        "git_worktree_dirty": None,
        "git_revision_label": "unknown",
        "git_status_porcelain": [],
        "git_tracked_diff_sha256": None,
    }


def collect_git_provenance() -> dict[str, Any]:
    try:
        commit_result = _run_git(["rev-parse", "HEAD"])
        status_result = _run_git(["status", "--porcelain", "--untracked-files=normal"])
        diff_result = _run_git(["diff", "HEAD", "--binary"], text=False)
    except (OSError, subprocess.CalledProcessError):
        return _unavailable_git_provenance()

    commit = commit_result.stdout.strip() or None
    status_lines = [line for line in status_result.stdout.splitlines() if line]
    dirty = bool(status_lines)
    if commit:
        revision_label = commit[:12] + ("-dirty" if dirty else "")
    else:
        revision_label = "unknown"
    return {
        "git_commit": commit,
        "git_provenance_available": True,
        "git_worktree_dirty": dirty,
        "git_revision_label": revision_label,
        "git_status_porcelain": status_lines,
        "git_tracked_diff_sha256": sha256_bytes(diff_result.stdout),
    }


def collect_phase_source_provenance() -> dict[str, Any]:
    return {
        **collect_git_provenance(),
        **collect_phase_source_fingerprint(),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return relative_to_repo(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    write_json(temp_path, payload)
    temp_path.replace(path)


def runtime_environment(device_label: str, gpu_name: str) -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "pytorch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device_label,
        "gpu_name": gpu_name,
    }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
