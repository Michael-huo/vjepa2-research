"""Phase 4: TUM RGB-D first-person state-refresh feasibility validation.

Dataset preparation and scientific metric helpers intentionally avoid importing
PyTorch. GPU/model modules are imported only by the full-run path.
"""

from __future__ import annotations

import bisect
import csv
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from research.scripts.common.data_paths import (
    REPO_ROOT,
    RESEARCH_ROOT,
    TUM_RGBD_MANIFEST_PATH,
    TUM_RGBD_ROOT,
)


PHASE4_NAME = "phase4-feasibility"
SEQUENCE_NAME = "rgbd_dataset_freiburg2_pioneer_slam"
DATASET_NAME = "TUM RGB-D"
SCOUT_RESOLUTION = 384
REFERENCE_RESOLUTION = 512
MAX_ASSOCIATION_DELTA_SECONDS = 0.02
TIME_HORIZONS_SECONDS = (0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00)
TIME_PAIR_MATCH_TOLERANCE_SECONDS = 0.04
MAX_PAIRS_PER_HORIZON = 512
OBSERVATION_BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.40)
CALIBRATION_FRACTION = 0.20
ADAPTIVE_TARGET_OBSERVATION_RATE = 0.10
STABLE_TIMEOUT_SECONDS = 2.00
NORMAL_TIMEOUT_SECONDS = 0.50
ADAPTIVE_LOW_QUANTILE_CANDIDATES = (0.50, 0.60, 0.70, 0.75, 0.80)
ADAPTIVE_HIGH_QUANTILE_CANDIDATES = (0.85, 0.90, 0.95, 0.975)
TIMING_MAX_FRAMES = 64
TIMING_WARMUPS = 3
TIMING_REPEATS = 5
FEATURE_CHUNK_SIZE = 8
FLOAT_TIE_TOLERANCE = 1e-12

PHASE4_OUTPUT_DIR = RESEARCH_ROOT / "outputs" / PHASE4_NAME
PHASE4_STAGED_OUTPUT_DIR = RESEARCH_ROOT / "outputs" / f".{PHASE4_NAME}.tmp"

EXPECTED_OUTPUT_FILES = {
    "ego_motion_consistency/metrics.json",
    "ego_motion_consistency/time_horizon_pairs.csv",
    "ego_motion_consistency/time_horizon_curves.png",
    "ego_motion_consistency/motion_conditioned_distance.png",
    "sparse_state_refresh/metrics.json",
    "sparse_state_refresh/budget_comparison.csv",
    "sparse_state_refresh/budget_quality_curves.png",
    "sparse_state_refresh/pose_coverage.png",
    "adaptive_observation/metrics.json",
    "adaptive_observation/policy_comparison.csv",
    "adaptive_observation/quality_compute_pareto.png",
    "adaptive_observation/observation_timeline.png",
    "summary.json",
    "report.md",
    "manifest.json",
}

PHASE4_SOURCE_PATHS = (
    "research/scripts/common/data_paths.py",
    "research/scripts/run_phase4_feasibility.py",
    "research/scripts/common/phase4_feasibility.py",
    "tests/test_phase4_feasibility.py",
    "research/README.md",
)

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class RgbEntry:
    timestamp: float
    relative_path: str


@dataclass(frozen=True)
class PoseEntry:
    timestamp: float
    translation: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class AssociatedFrame:
    index: int
    rgb_timestamp: float
    gt_timestamp: float
    association_delta_seconds: float
    relative_rgb_path: str
    rgb_path: Path
    translation: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class PreparedTumSequence:
    dataset_root: Path
    manifest_path: Path
    frames: tuple[AssociatedFrame, ...]
    manifest: dict[str, Any]
    skipped: bool


@dataclass(frozen=True)
class SourceViews:
    original_size_hw: tuple[int, int]
    source_crop_box_xyxy: tuple[int, int, int, int]
    rgb384: np.ndarray
    rgb512: np.ndarray
    gray384: np.ndarray


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _portable_manifest_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def normalize_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("Quaternion must contain four finite qx qy qz qw values.")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("Invalid zero-norm quaternion in groundtruth trajectory.")
    normalized = quaternion / norm
    return tuple(float(value) for value in normalized)


def _data_lines(path: Path) -> Iterable[tuple[int, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required TUM RGB-D metadata file is missing: {path}")
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if line and not line.startswith("#"):
            yield line_number, line


def _validate_rgb_relative_path(value: str, dataset_root: Path) -> str:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe TUM RGB relative path: {value!r}")
    if not pure.parts or pure.parts[0] != "rgb":
        raise ValueError(f"TUM RGB path must be inside rgb/: {value!r}")
    candidate = (dataset_root / Path(*pure.parts)).resolve()
    root = dataset_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"TUM RGB path escapes the dataset root: {value!r}") from error
    return pure.as_posix()


def parse_rgb_file(path: Path, *, dataset_root: Path) -> list[RgbEntry]:
    entries: list[RgbEntry] = []
    timestamps: set[float] = set()
    for line_number, line in _data_lines(path):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path.name}:{line_number}: expected timestamp and RGB path.")
        try:
            timestamp = float(fields[0])
        except ValueError as error:
            raise ValueError(f"{path.name}:{line_number}: invalid RGB timestamp.") from error
        if not math.isfinite(timestamp):
            raise ValueError(f"{path.name}:{line_number}: RGB timestamp must be finite.")
        if timestamp in timestamps:
            raise ValueError(f"{path.name}:{line_number}: duplicate RGB timestamp {timestamp}.")
        timestamps.add(timestamp)
        entries.append(RgbEntry(timestamp, _validate_rgb_relative_path(fields[1], dataset_root)))
    if not entries:
        raise ValueError(f"No RGB entries found in {path}.")
    return sorted(entries, key=lambda item: item.timestamp)


def parse_groundtruth_file(path: Path, *, reject_duplicate_timestamps: bool = True) -> list[PoseEntry]:
    entries: list[PoseEntry] = []
    timestamps: set[float] = set()
    for line_number, line in _data_lines(path):
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(
                f"{path.name}:{line_number}: expected timestamp tx ty tz qx qy qz qw."
            )
        try:
            values = [float(field) for field in fields]
        except ValueError as error:
            raise ValueError(f"{path.name}:{line_number}: invalid pose value.") from error
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name}:{line_number}: pose values must be finite.")
        timestamp = values[0]
        if timestamp in timestamps and reject_duplicate_timestamps:
            raise ValueError(f"{path.name}:{line_number}: duplicate GT timestamp {timestamp}.")
        timestamps.add(timestamp)
        entries.append(
            PoseEntry(
                timestamp=timestamp,
                translation=tuple(values[1:4]),
                quaternion_xyzw=normalize_quaternion(values[4:8]),
            )
        )
    if not entries:
        raise ValueError(f"No ground-truth poses found in {path}.")
    return sorted(entries, key=lambda item: item.timestamp)


def associate_rgb_to_poses(
    rgb_entries: Sequence[RgbEntry],
    poses: Sequence[PoseEntry],
    *,
    dataset_root: Path,
    max_delta_seconds: float = MAX_ASSOCIATION_DELTA_SECONDS,
) -> tuple[list[AssociatedFrame], int]:
    if max_delta_seconds < 0:
        raise ValueError("max_delta_seconds must be non-negative.")
    unique_poses: list[PoseEntry] = []
    seen_pose_times: set[float] = set()
    for pose in poses:
        if pose.timestamp not in seen_pose_times:
            unique_poses.append(pose)
            seen_pose_times.add(pose.timestamp)
    poses = unique_poses
    pose_times = [pose.timestamp for pose in poses]
    associated: list[AssociatedFrame] = []
    valid_rgb_file_count = 0
    for rgb in sorted(rgb_entries, key=lambda item: item.timestamp):
        rgb_path = dataset_root / Path(*PurePosixPath(rgb.relative_path).parts)
        if not rgb_path.is_file():
            continue
        valid_rgb_file_count += 1
        insertion = bisect.bisect_left(pose_times, rgb.timestamp)
        candidate_indices = []
        if insertion > 0:
            candidate_indices.append(insertion - 1)
        if insertion < len(poses):
            candidate_indices.append(insertion)
        if not candidate_indices:
            continue
        pose_index = min(
            candidate_indices,
            key=lambda index: (abs(poses[index].timestamp - rgb.timestamp), poses[index].timestamp),
        )
        pose = poses[pose_index]
        delta = abs(pose.timestamp - rgb.timestamp)
        if delta > max_delta_seconds + 1e-12:
            continue
        associated.append(
            AssociatedFrame(
                index=len(associated),
                rgb_timestamp=rgb.timestamp,
                gt_timestamp=pose.timestamp,
                association_delta_seconds=float(delta),
                relative_rgb_path=rgb.relative_path,
                rgb_path=rgb_path,
                translation=pose.translation,
                quaternion_xyzw=pose.quaternion_xyzw,
            )
        )
    validate_associated_frames(associated)
    return associated, valid_rgb_file_count


def validate_associated_frames(frames: Sequence[AssociatedFrame]) -> None:
    if not frames:
        raise ValueError("No TUM RGB frames could be associated with ground-truth poses.")
    for previous, current in zip(frames, frames[1:]):
        if current.rgb_timestamp <= previous.rgb_timestamp:
            raise ValueError("Associated RGB timestamps must be strictly increasing.")


def _manifest_frame_record(frame: AssociatedFrame, dataset_root_record: str) -> dict[str, Any]:
    relative = PurePosixPath(frame.relative_rgb_path)
    return {
        "index": frame.index,
        "rgb_path": (PurePosixPath(dataset_root_record) / relative).as_posix(),
        "rgb_relative_path": relative.as_posix(),
        "rgb_size_bytes": int(frame.rgb_path.stat().st_size),
        "rgb_timestamp": frame.rgb_timestamp,
        "gt_timestamp": frame.gt_timestamp,
        "association_delta_seconds": frame.association_delta_seconds,
        "tx": frame.translation[0],
        "ty": frame.translation[1],
        "tz": frame.translation[2],
        "qx": frame.quaternion_xyzw[0],
        "qy": frame.quaternion_xyzw[1],
        "qz": frame.quaternion_xyzw[2],
        "qw": frame.quaternion_xyzw[3],
    }


def build_tum_manifest(
    *,
    dataset_root: Path,
    manifest_path: Path,
    rgb_entries: Sequence[RgbEntry],
    poses: Sequence[PoseEntry],
    frames: Sequence[AssociatedFrame],
    valid_rgb_file_count: int,
) -> dict[str, Any]:
    deltas = np.asarray([frame.association_delta_seconds for frame in frames], dtype=np.float64)
    root_record = _portable_root(dataset_root)
    frame_records = [_manifest_frame_record(frame, root_record) for frame in frames]
    structure = {
        "dataset": DATASET_NAME,
        "sequence_name": SEQUENCE_NAME,
        "dataset_root": root_record,
        "rgb_txt_sha256": sha256_file(dataset_root / "rgb.txt"),
        "groundtruth_txt_sha256": sha256_file(dataset_root / "groundtruth.txt"),
        "max_association_delta_seconds": MAX_ASSOCIATION_DELTA_SECONDS,
        "frames": frame_records,
    }
    encoded = json.dumps(structure, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "schema_version": "1.0",
        "phase_name": PHASE4_NAME,
        "dataset": DATASET_NAME,
        "sequence_name": SEQUENCE_NAME,
        "dataset_root": root_record,
        "manifest_path": _portable_manifest_path(manifest_path),
        "prepared_at": now_utc(),
        "rgb_entry_count": len(rgb_entries),
        "valid_rgb_file_count": int(valid_rgb_file_count),
        "groundtruth_entry_count": len(poses),
        "duplicate_groundtruth_timestamp_count": len(poses) - len({pose.timestamp for pose in poses}),
        "associated_frame_count": len(frames),
        "dropped_frame_count": len(rgb_entries) - len(frames),
        "mean_association_delta_seconds": float(np.mean(deltas)),
        "max_association_delta_seconds": float(np.max(deltas)),
        "duration_seconds": float(frames[-1].rgb_timestamp - frames[0].rgb_timestamp),
        "association_policy": "nearest ground-truth timestamp; no interpolation; delta <= 0.02 seconds; equal-distance GT chooses earlier timestamp",
        "groundtruth_duplicate_policy": "Official sequence duplicates are retained in the raw count; association deterministically uses the first file-order pose for each duplicate timestamp.",
        "structure": structure,
        "structure_fingerprint": hashlib.sha256(encoded).hexdigest(),
        "frames": frame_records,
    }


def prepare_tum_sequence(
    *,
    dataset_root: Path = TUM_RGBD_ROOT,
    manifest_path: Path = TUM_RGBD_MANIFEST_PATH,
) -> PreparedTumSequence:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Fixed TUM RGB-D sequence is missing: {_portable_root(dataset_root)}")
    rgb_entries = parse_rgb_file(dataset_root / "rgb.txt", dataset_root=dataset_root)
    poses = parse_groundtruth_file(
        dataset_root / "groundtruth.txt",
        reject_duplicate_timestamps=dataset_root.resolve() != TUM_RGBD_ROOT.resolve(),
    )
    frames, valid_count = associate_rgb_to_poses(
        rgb_entries,
        poses,
        dataset_root=dataset_root,
    )
    current = build_tum_manifest(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        rgb_entries=rgb_entries,
        poses=poses,
        frames=frames,
        valid_rgb_file_count=valid_count,
    )
    skipped = False
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("structure_fingerprint") == current["structure_fingerprint"]:
            current = existing
            skipped = True
    if not skipped:
        write_json_atomic(manifest_path, current)
    return PreparedTumSequence(dataset_root, manifest_path, tuple(frames), current, skipped)


def print_prepare_summary(prepared: PreparedTumSequence) -> None:
    manifest = prepared.manifest
    print(f"      sequence: {manifest['sequence_name']}")
    print(
        "      RGB entries/files/associated/dropped: "
        f"{manifest['rgb_entry_count']}/{manifest['valid_rgb_file_count']}/"
        f"{manifest['associated_frame_count']}/{manifest['dropped_frame_count']}"
    )
    print(f"      ground-truth entries: {manifest['groundtruth_entry_count']}")
    if manifest.get("duplicate_groundtruth_timestamp_count"):
        print(
            "      duplicate GT timestamps retained with deterministic first-row association: "
            f"{manifest['duplicate_groundtruth_timestamp_count']}"
        )
    print(
        "      association delta mean/max: "
        f"{manifest['mean_association_delta_seconds']:.9f}s/"
        f"{manifest['max_association_delta_seconds']:.9f}s"
    )
    print(f"      duration: {manifest['duration_seconds']:.6f}s")


def source_crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid source image size: {(width, height)}")
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return left, top, left + side, top + side


def _rgb_pil(image: Image.Image | np.ndarray) -> Image.Image:
    return image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")


def _source_square(image: Image.Image | np.ndarray) -> tuple[Image.Image, tuple[int, int, int, int], tuple[int, int]]:
    pil = _rgb_pil(image)
    width, height = pil.size
    crop_box = source_crop_box(width, height)
    return pil.crop(crop_box), crop_box, (height, width)


def _resize_square(square: Image.Image, resolution: int) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(square.resize((resolution, resolution), resampling), dtype=np.uint8)


def build_source_views(image: Image.Image | np.ndarray) -> SourceViews:
    square, crop_box, original_size = _source_square(image)
    rgb384 = _resize_square(square, SCOUT_RESOLUTION)
    rgb512 = _resize_square(square, REFERENCE_RESOLUTION)
    gray384 = cv2.cvtColor(rgb384, cv2.COLOR_RGB2GRAY)
    return SourceViews(original_size, crop_box, rgb384, rgb512, gray384)


def build_scout_gray(image: Image.Image | np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    square, crop_box, _ = _source_square(image)
    rgb384 = _resize_square(square, SCOUT_RESOLUTION)
    return cv2.cvtColor(rgb384, cv2.COLOR_RGB2GRAY), crop_box


def build_resolution_rgb(image: Image.Image | np.ndarray, resolution: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if resolution not in (SCOUT_RESOLUTION, REFERENCE_RESOLUTION):
        raise ValueError(f"Unsupported fixed Phase 4 resolution: {resolution}")
    square, crop_box, _ = _source_square(image)
    return _resize_square(square, resolution), crop_box


def pixel_distance(current_gray: np.ndarray, anchor_gray: np.ndarray) -> float:
    if current_gray.shape != anchor_gray.shape:
        raise ValueError("Pixel-distance images must have the same shape.")
    return float(np.mean(cv2.absdiff(current_gray.astype(np.uint8), anchor_gray.astype(np.uint8))) / 255.0)


def flow_magnitude(current_gray: np.ndarray, anchor_gray: np.ndarray) -> float:
    if current_gray.shape != anchor_gray.shape:
        raise ValueError("Optical-flow images must have the same shape.")
    flow = cv2.calcOpticalFlowFarneback(
        anchor_gray.astype(np.uint8),
        current_gray.astype(np.uint8),
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    return float(np.linalg.norm(flow, axis=2).mean())


def global_latents(dense_tokens: np.ndarray) -> np.ndarray:
    if dense_tokens.ndim not in (3, 4):
        raise ValueError("Dense tokens must have shape (T,N,D) or (T,H,W,D).")
    axes = tuple(range(1, dense_tokens.ndim - 1))
    pooled = np.asarray(dense_tokens, dtype=np.float32).mean(axis=axes)
    return pooled / (np.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-8)


def latent_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        raise ValueError("Latent cosine distance requires non-zero vectors.")
    return float(1.0 - np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def translation_delta(left: Sequence[float], right: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)))


def rotation_delta_degrees(left_xyzw: Sequence[float], right_xyzw: Sequence[float]) -> float:
    left = np.asarray(normalize_quaternion(left_xyzw), dtype=np.float64)
    right = np.asarray(normalize_quaternion(right_xyzw), dtype=np.float64)
    relative_w = abs(float(np.dot(left, right)))
    return float(np.degrees(2.0 * np.arccos(np.clip(relative_w, 0.0, 1.0))))


def pose_arrays(frames: Sequence[AssociatedFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.asarray([frame.rgb_timestamp for frame in frames], dtype=np.float64)
    positions = np.asarray([frame.translation for frame in frames], dtype=np.float64)
    quaternions = np.asarray([frame.quaternion_xyzw for frame in frames], dtype=np.float64)
    return timestamps, positions, quaternions


def cumulative_motion_prefix(
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(positions)
    translation_prefix = np.zeros(count, dtype=np.float64)
    rotation_prefix = np.zeros(count, dtype=np.float64)
    for index in range(1, count):
        translation_prefix[index] = translation_prefix[index - 1] + translation_delta(positions[index - 1], positions[index])
        rotation_prefix[index] = rotation_prefix[index - 1] + rotation_delta_degrees(quaternions[index - 1], quaternions[index])
    return translation_prefix, rotation_prefix


def motion_between(
    anchor: int,
    target: int,
    positions: np.ndarray,
    quaternions: np.ndarray,
    translation_prefix: np.ndarray,
    rotation_prefix: np.ndarray,
) -> dict[str, float]:
    if not 0 <= anchor <= target < len(positions):
        raise ValueError("Motion indices must satisfy 0 <= anchor <= target < frame_count.")
    return {
        "net_translation_m": translation_delta(positions[anchor], positions[target]),
        "net_rotation_deg": rotation_delta_degrees(quaternions[anchor], quaternions[target]),
        "cumulative_translation_m": float(translation_prefix[target] - translation_prefix[anchor]),
        "cumulative_rotation_deg": float(rotation_prefix[target] - rotation_prefix[anchor]),
    }


def deterministic_indices(count: int, limit: int) -> list[int]:
    if count <= 0 or limit <= 0:
        return []
    if count <= limit:
        return list(range(count))
    values = np.linspace(0, count - 1, num=limit)
    indices = [int(round(value)) for value in values]
    deduped = list(dict.fromkeys(indices))
    if len(deduped) != limit:
        raise RuntimeError("Deterministic sampling unexpectedly produced duplicate indices.")
    return deduped


def percentile_stats(values: Iterable[float], *, include_median: bool = False) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        result: dict[str, Any] = {"count": 0, "mean": None, "p90": None, "p95": None, "max": None}
        if include_median:
            result["median"] = None
        return result
    result = {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }
    if include_median:
        result["median"] = float(np.median(array))
    return result


def signal_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "std": None, "p90": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def average_tie_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    cursor = 0
    while cursor < len(array):
        end = cursor + 1
        while end < len(array) and array[order[end]] == array[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal lengths.")
    if len(left) < 2:
        return {"value": None, "sample_count": len(left), "reason": "insufficient_samples"}
    left_rank = average_tie_ranks(left)
    right_rank = average_tie_ranks(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return {"value": None, "sample_count": len(left), "reason": "zero_rank_variance"}
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return {"value": value, "sample_count": len(left), "reason": None}


def half_up_target_count(budget: float, frame_count: int) -> int:
    if frame_count <= 0 or not 0.0 <= budget <= 1.0:
        raise ValueError("Budget must be in [0,1] and frame_count must be positive.")
    return min(frame_count, max(1, int(math.floor(budget * frame_count + 0.5))))


def uniform_time_schedule(timestamps: Sequence[float], target_count: int) -> np.ndarray:
    times = np.asarray(timestamps, dtype=np.float64)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("Uniform time selection requires strictly increasing timestamps.")
    if not 1 <= target_count <= len(times):
        raise ValueError("target_count must be within the segment frame count.")
    targets = np.linspace(times[0], times[-1], target_count)
    selected: list[int] = [0]
    selected_set = {0}
    for target in targets[1:]:
        candidates = sorted(range(len(times)), key=lambda index: (abs(times[index] - target), index))
        chosen = next(index for index in candidates if index not in selected_set)
        selected.append(chosen)
        selected_set.add(chosen)
    observed = np.zeros(len(times), dtype=bool)
    observed[selected] = True
    if int(observed.sum()) != target_count or not observed[0]:
        raise RuntimeError("Uniform time selection failed its exact-count invariant.")
    return observed


def offline_topk_schedule(
    adjacent_scores: Sequence[float],
    timestamps: Sequence[float],
    target_count: int,
) -> np.ndarray:
    times = np.asarray(timestamps, dtype=np.float64)
    scores = np.asarray(adjacent_scores, dtype=np.float64)
    if len(scores) != len(times) - 1 or not np.isfinite(scores).all():
        raise ValueError("Adjacent scores must contain one finite value for every frame after frame zero.")
    if not 1 <= target_count <= len(times):
        raise ValueError("target_count must be within the segment frame count.")
    ranked = sorted(range(1, len(times)), key=lambda index: (-scores[index - 1], times[index], index))
    selected = [0, *ranked[: target_count - 1]]
    observed = np.zeros(len(times), dtype=bool)
    observed[selected] = True
    if int(observed.sum()) != target_count:
        raise RuntimeError("Offline Top-K selection failed its exact-count invariant.")
    return observed


def causal_anchor_schedule(
    timestamps: Sequence[float],
    score_from_anchor: Callable[[int, int], float],
    low_threshold: float,
    high_threshold: float,
) -> tuple[np.ndarray, list[str], list[float]]:
    times = np.asarray(timestamps, dtype=np.float64)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("Causal scheduling requires strictly increasing timestamps.")
    if not low_threshold <= high_threshold:
        raise ValueError("Adaptive thresholds must be ordered.")
    observed = np.zeros(len(times), dtype=bool)
    observed[0] = True
    anchor = 0
    reasons = ["initial"]
    scores = [float("nan")]
    for index in range(1, len(times)):
        score = float(score_from_anchor(index, anchor))
        scores.append(score)
        elapsed = float(times[index] - times[anchor])
        if score >= high_threshold:
            reason = "dynamic"
        elif score >= low_threshold and elapsed >= NORMAL_TIMEOUT_SECONDS:
            reason = "normal_timeout"
        elif score < low_threshold and elapsed >= STABLE_TIMEOUT_SECONDS:
            reason = "stable_timeout"
        else:
            reason = "skip"
        if reason != "skip":
            observed[index] = True
            anchor = index
        reasons.append(reason)
    return observed, reasons, scores


def hold_last_state_metrics(
    observed: np.ndarray,
    reference_latents: np.ndarray,
    timestamps: Sequence[float] | None = None,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=bool)
    if len(observed) != len(reference_latents) or not len(observed) or not observed[0]:
        raise ValueError("Schedule must match latents and refresh frame zero.")
    errors: list[float] = []
    similarities: list[float] = []
    anchors: list[int] = []
    anchor = 0
    for index in range(len(observed)):
        if observed[index]:
            anchor = index
        error = latent_distance(reference_latents[anchor], reference_latents[index])
        errors.append(error)
        similarities.append(1.0 - error)
        anchors.append(anchor)
    refresh_indices = np.flatnonzero(observed)
    intervals = np.diff(refresh_indices)
    interval_seconds = (
        np.diff(np.asarray(timestamps, dtype=np.float64)[refresh_indices])
        if timestamps is not None else np.asarray([], dtype=np.float64)
    )
    return {
        "per_frame_state_error": errors,
        "per_frame_state_similarity": similarities,
        "anchor_indices": anchors,
        "mean_state_error": float(np.mean(errors)),
        "median_state_error": float(np.median(errors)),
        "p90_state_error": float(np.percentile(errors, 90)),
        "p95_state_error": float(np.percentile(errors, 95)),
        "max_state_error": float(np.max(errors)),
        "mean_state_similarity": float(np.mean(similarities)),
        "observation_count": int(observed.sum()),
        "observation_rate": float(observed.mean()),
        "observation_reduction_ratio": float(1.0 - observed.mean()),
        "refresh_interval": {
            "count": int(len(intervals)),
            "mean": float(np.mean(intervals)) if len(intervals) else None,
            "p90": float(np.percentile(intervals, 90)) if len(intervals) else None,
            "max": int(np.max(intervals)) if len(intervals) else None,
        },
        "refresh_interval_seconds": {
            "count": int(len(interval_seconds)),
            "mean": float(np.mean(interval_seconds)) if len(interval_seconds) else None,
            "median": float(np.median(interval_seconds)) if len(interval_seconds) else None,
            "p90": float(np.percentile(interval_seconds, 90)) if len(interval_seconds) else None,
            "p95": float(np.percentile(interval_seconds, 95)) if len(interval_seconds) else None,
            "max": float(np.max(interval_seconds)) if len(interval_seconds) else None,
        },
    }


def pose_staleness_metrics(
    observed: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> dict[str, Any]:
    translation_prefix, rotation_prefix = cumulative_motion_prefix(positions, quaternions)
    per_frame: list[dict[str, Any]] = []
    anchor = 0
    for index in range(len(observed)):
        if observed[index]:
            anchor = index
        per_frame.append({"frame_index": index, "anchor_index": anchor, **motion_between(anchor, index, positions, quaternions, translation_prefix, rotation_prefix)})
    refresh_indices = np.flatnonzero(observed)
    gaps = [
        {"anchor_i": int(left), "anchor_j": int(right), **motion_between(int(left), int(right), positions, quaternions, translation_prefix, rotation_prefix)}
        for left, right in zip(refresh_indices, refresh_indices[1:])
    ]
    metric_keys = ("net_translation_m", "net_rotation_deg", "cumulative_translation_m", "cumulative_rotation_deg")
    return {
        "per_frame": per_frame,
        "staleness_summary": {key: percentile_stats(record[key] for record in per_frame) for key in metric_keys},
        "coverage_gaps": gaps,
        "coverage_gap_summary": {key: percentile_stats(record[key] for record in gaps) for key in metric_keys},
        "primary_coverage_metrics": ["cumulative_translation_m", "cumulative_rotation_deg"],
    }


def large_motion_coverage(
    observed: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> dict[str, Any]:
    step_translation = np.asarray([translation_delta(positions[i - 1], positions[i]) for i in range(1, len(positions))])
    step_rotation = np.asarray([rotation_delta_degrees(quaternions[i - 1], quaternions[i]) for i in range(1, len(quaternions))])
    def evaluate(values: np.ndarray) -> dict[str, Any]:
        if not len(values):
            return {"threshold_p90": None, "large_step_count": 0, "covered_count": 0, "recall": None}
        threshold = float(np.percentile(values, 90))
        large_frames = [index + 1 for index, value in enumerate(values) if value >= threshold]
        covered = [index for index in large_frames if observed[max(0, index - 1):min(len(observed), index + 2)].any()]
        return {"threshold_p90": threshold, "large_step_count": len(large_frames), "covered_count": len(covered), "recall": float(len(covered) / len(large_frames)) if large_frames else None}
    return {"translation": evaluate(step_translation), "rotation": evaluate(step_rotation)}


def budget_curve_auc(budgets: Sequence[float], values: Sequence[float]) -> float:
    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if len(x) < 2 or len(x) != len(y) or np.any(np.diff(x) <= 0):
        raise ValueError("Budget AUC requires aligned values on a strictly increasing grid.")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def frame_support_durations(timestamps: Sequence[float]) -> dict[str, Any]:
    times = np.asarray(timestamps, dtype=np.float64)
    if not len(times) or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError("Frame support requires strictly increasing timestamps.")
    if len(times) == 1:
        return {"durations_seconds": [0.0], "status": "degenerate", "reason": "single_frame_segment"}
    deltas = np.diff(times)
    return {
        "durations_seconds": [*deltas.tolist(), float(np.median(deltas))],
        "status": "available",
        "reason": None,
    }


def high_error_episode_metrics(
    errors: Sequence[float],
    timestamps: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    support = frame_support_durations(timestamps)
    durations = np.asarray(support["durations_seconds"], dtype=np.float64)
    if len(values) != len(durations):
        raise ValueError("Errors and timestamps must have equal lengths.")
    above = values > threshold
    episodes: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(above):
        if not above[cursor]:
            cursor += 1
            continue
        end = cursor
        while end + 1 < len(above) and above[end + 1]:
            end += 1
        episodes.append(
            {
                "start_frame": int(cursor),
                "end_frame": int(end),
                "frame_count": int(end - cursor + 1),
                "start_timestamp": float(timestamps[cursor]),
                "end_support_timestamp": float(timestamps[end] + durations[end]),
                "duration_seconds": float(np.sum(durations[cursor : end + 1])),
            }
        )
        cursor = end + 1
    episode_durations = [episode["duration_seconds"] for episode in episodes]
    return {
        "threshold": float(threshold),
        "support_status": support["status"],
        "support_reason": support["reason"],
        "above_threshold_frame_count": int(above.sum()),
        "above_threshold_frame_fraction": float(above.mean()),
        "episode_count": len(episodes),
        "episodes": episodes,
        "max_duration_seconds": max(episode_durations, default=0.0),
        "p95_duration_seconds": float(np.percentile(episode_durations, 95)) if episode_durations else 0.0,
    }


def adjacent_reference_high_error_threshold(reference_latents: np.ndarray) -> float:
    if len(reference_latents) < 2:
        raise ValueError("High-error threshold requires at least two reference states.")
    errors = [latent_distance(reference_latents[index - 1], reference_latents[index]) for index in range(1, len(reference_latents))]
    return float(np.percentile(errors, 95))


def timing_statistics(samples_ms: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if not len(values):
        return {"sample_count": 0, "median_ms": None, "p90_ms": None}
    return {"sample_count": int(len(values)), "median_ms": float(np.median(values)), "p90_ms": float(np.percentile(values, 90))}


def estimate_policy_latency(
    *,
    observation_rate: float,
    pipeline_512_ms: float,
    scheduler_ms: float,
    pipeline_384_ms: float = 0.0,
    decision_ms: float = 0.0,
) -> dict[str, float]:
    latency = pipeline_384_ms + observation_rate * pipeline_512_ms + decision_ms + scheduler_ms
    return {"estimated_latency_ms_per_frame": float(latency), "estimated_fps": float(1000.0 / latency) if latency > 0 else float("inf"), "speedup_vs_full_512": float(pipeline_512_ms / latency) if latency > 0 else float("inf")}


def break_even_refresh_rate(pipeline_384_ms: float, pipeline_512_ms: float, decision_ms: float) -> float:
    if pipeline_512_ms <= 0:
        raise ValueError("512 pipeline latency must be positive.")
    return float((pipeline_512_ms - pipeline_384_ms - decision_ms) / pipeline_512_ms)


def find_time_horizon_pairs(
    timestamps: Sequence[float],
    horizon_seconds: float,
    *,
    tolerance_seconds: float = TIME_PAIR_MATCH_TOLERANCE_SECONDS,
    limit: int = MAX_PAIRS_PER_HORIZON,
) -> dict[str, Any]:
    times = np.asarray(timestamps, dtype=np.float64)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Time pairing requires at least two strictly increasing timestamps.")
    valid: list[tuple[int, int]] = []
    for left in range(len(times) - 1):
        target = float(times[left] + horizon_seconds)
        insertion = bisect.bisect_left(times, target, lo=left + 1)
        candidates = [index for index in (insertion - 1, insertion) if left < index < len(times)]
        if not candidates:
            continue
        right = min(candidates, key=lambda index: (abs(times[index] - target), index))
        if abs((times[right] - times[left]) - horizon_seconds) <= tolerance_seconds + FLOAT_TIE_TOLERANCE:
            valid.append((left, right))
    valid = list(dict.fromkeys(valid))
    used = [valid[index] for index in deterministic_indices(len(valid), limit)]
    return {
        "candidate_count": len(times) - 1,
        "valid_count": len(valid),
        "used_count": len(used),
        "tolerance_discarded_count": len(times) - 1 - len(valid),
        "pairs": used,
    }


def _time_pair_records(
    frames: Sequence[AssociatedFrame],
    gray_frames: np.ndarray,
    scout_latents: np.ndarray,
    reference_latents: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timestamps, positions, quaternions = pose_arrays(frames)
    translation_prefix, rotation_prefix = cumulative_motion_prefix(positions, quaternions)
    records: list[dict[str, Any]] = []
    pairing: dict[str, Any] = {}
    for horizon in TIME_HORIZONS_SECONDS:
        result = find_time_horizon_pairs(timestamps, horizon)
        pairing[f"{horizon:g}"] = {key: value for key, value in result.items() if key != "pairs"}
        for left, right in result["pairs"]:
            motion = motion_between(left, right, positions, quaternions, translation_prefix, rotation_prefix)
            distance384 = latent_distance(scout_latents[left], scout_latents[right])
            distance512 = latent_distance(reference_latents[left], reference_latents[right])
            actual_delta = float(timestamps[right] - timestamps[left])
            records.append(
                {
                    "target_horizon_seconds": horizon,
                    "frame_i": left,
                    "frame_j": right,
                    "timestamp_i": timestamps[left],
                    "timestamp_j": timestamps[right],
                    "actual_time_delta_seconds": actual_delta,
                    "time_match_error_seconds": abs(actual_delta - horizon),
                    "frame_index_delta": right - left,
                    "net_translation_delta_m": motion["net_translation_m"],
                    "net_rotation_delta_deg": motion["net_rotation_deg"],
                    "cumulative_translation_delta_m": motion["cumulative_translation_m"],
                    "cumulative_rotation_delta_deg": motion["cumulative_rotation_deg"],
                    "pixel_distance": pixel_distance(gray_frames[right], gray_frames[left]),
                    "flow_magnitude": flow_magnitude(gray_frames[right], gray_frames[left]),
                    "jepa384_distance": distance384,
                    "jepa384_similarity": 1.0 - distance384,
                    "jepa512_distance": distance512,
                    "jepa512_similarity": 1.0 - distance512,
                }
            )
    return records, pairing


def scout_reference_consistency(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    left = [record["jepa384_distance"] for record in records]
    right = [record["jepa512_distance"] for record in records]
    return {
        "distance_spearman": spearman_correlation(left, right),
        "mean_absolute_distance_difference": float(np.mean(np.abs(np.asarray(left) - np.asarray(right)))) if left else None,
    }


def _correlation_payload(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    motions = {
        "net_translation": np.asarray([record["net_translation_delta_m"] for record in records]),
        "net_rotation": np.asarray([record["net_rotation_delta_deg"] for record in records]),
        "cumulative_translation": np.asarray([record["cumulative_translation_delta_m"] for record in records]),
        "cumulative_rotation": np.asarray([record["cumulative_rotation_delta_deg"] for record in records]),
    }
    signals = {
        "pixel": np.asarray([record["pixel_distance"] for record in records]),
        "flow": np.asarray([record["flow_magnitude"] for record in records]),
        "jepa384": np.asarray([record["jepa384_distance"] for record in records]),
        "jepa512": np.asarray([record["jepa512_distance"] for record in records]),
    }
    rotation_median = float(np.median(motions["net_rotation"])) if records else None
    translation_median = float(np.median(motions["net_translation"])) if records else None
    if not records:
        return {
            "conditioning": {
                "net_rotation_median_deg": None,
                "net_translation_median_m": None,
                "low_rotation_sample_count": 0,
                "low_translation_sample_count": 0,
            },
            "signals": {
                name: {
                    **{motion: spearman_correlation([], []) for motion in motions},
                    "low_rotation_net_translation": spearman_correlation([], []),
                    "low_translation_net_rotation": spearman_correlation([], []),
                }
                for name in signals
            },
        }
    low_rotation = motions["net_rotation"] <= rotation_median
    low_translation = motions["net_translation"] <= translation_median
    output: dict[str, Any] = {
        "conditioning": {
            "net_rotation_median_deg": rotation_median,
            "net_translation_median_m": translation_median,
            "low_rotation_sample_count": int(low_rotation.sum()),
            "low_translation_sample_count": int(low_translation.sum()),
        },
        "signals": {},
    }
    for name, signal in signals.items():
        output["signals"][name] = {
            **{motion: spearman_correlation(signal, values) for motion, values in motions.items()},
            "low_rotation_net_translation": spearman_correlation(signal[low_rotation], motions["net_translation"][low_rotation]),
            "low_translation_net_rotation": spearman_correlation(signal[low_translation], motions["net_rotation"][low_translation]),
        }
    return output


def _motion_bins(records: Sequence[dict[str, Any]], motion_key: str) -> list[dict[str, Any]]:
    motion = np.asarray([record[motion_key] for record in records], dtype=np.float64)
    edges = np.quantile(motion, [0.0, 0.25, 0.5, 0.75, 1.0])
    output = []
    for index in range(4):
        selected = ((motion >= edges[index]) & (motion <= edges[index + 1])) if index == 3 else ((motion >= edges[index]) & (motion < edges[index + 1]))
        row: dict[str, Any] = {
            "quantile_bin": index + 1,
            "motion_low": float(edges[index]),
            "motion_high": float(edges[index + 1]),
            "sample_count": int(selected.sum()),
        }
        for state in ("jepa384_distance", "jepa512_distance"):
            values = np.asarray([record[state] for record in records], dtype=np.float64)[selected]
            row[state] = {"mean": float(np.mean(values)) if len(values) else None, "p90": float(np.percentile(values, 90)) if len(values) else None}
        output.append(row)
    return output


def _time_horizon_statistics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for horizon in TIME_HORIZONS_SECONDS:
        selected = [record for record in records if record["target_horizon_seconds"] == horizon]
        key = f"{horizon:g}"
        stats = {
            "pair_count": len(selected),
            "actual_time_delta_seconds": percentile_stats((record["actual_time_delta_seconds"] for record in selected), include_median=True),
            "frame_index_delta": percentile_stats((record["frame_index_delta"] for record in selected), include_median=True),
            "pixel_distance": signal_stats(record["pixel_distance"] for record in selected),
            "flow_magnitude": signal_stats(record["flow_magnitude"] for record in selected),
            "jepa384_distance": signal_stats(record["jepa384_distance"] for record in selected),
            "jepa384_similarity": signal_stats(record["jepa384_similarity"] for record in selected),
            "jepa512_distance": signal_stats(record["jepa512_distance"] for record in selected),
            "jepa512_similarity": signal_stats(record["jepa512_similarity"] for record in selected),
            "motion": {
                name: percentile_stats(record[field] for record in selected)
                for name, field in (
                    ("net_translation", "net_translation_delta_m"),
                    ("net_rotation", "net_rotation_delta_deg"),
                    ("cumulative_translation", "cumulative_translation_delta_m"),
                    ("cumulative_rotation", "cumulative_rotation_delta_deg"),
                )
            },
            "scout_reference_consistency": scout_reference_consistency(selected),
            "motion_correlations": _correlation_payload(selected),
        }
        output[key] = stats
    return output


def run_ego_motion_consistency(
    *,
    frames: Sequence[AssociatedFrame],
    gray_frames: np.ndarray,
    scout_latents: np.ndarray,
    reference_latents: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    records, pairing = _time_pair_records(frames, gray_frames, scout_latents, reference_latents)
    payload = {
        "experiment": "time_based_ego_motion_and_scout_reference_consistency",
        "time_horizons_seconds": list(TIME_HORIZONS_SECONDS),
        "pair_match_tolerance_seconds": TIME_PAIR_MATCH_TOLERANCE_SECONDS,
        "max_pairs_per_horizon": MAX_PAIRS_PER_HORIZON,
        "pairing_counts": pairing,
        "time_protocol": "nearest strictly future RGB timestamp; no fixed-FPS assumption",
        "horizon_statistics": _time_horizon_statistics(records),
        "overall_scout_reference_consistency": scout_reference_consistency(records),
        "correlations": _correlation_payload(records),
        "translation_conditioned_distance": _motion_bins(records, "net_translation_delta_m"),
        "rotation_conditioned_distance": _motion_bins(records, "net_rotation_delta_deg"),
    }
    write_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "time_horizon_pairs.csv", records)
    save_ego_motion_figures(output_dir, payload)
    return payload


def causal_score_function(method: str, gray_frames: np.ndarray, scout_latents: np.ndarray) -> Callable[[int, int], float]:
    if method == "jepa":
        return lambda index, anchor: latent_distance(scout_latents[index], scout_latents[anchor])
    if method == "pixel":
        return lambda index, anchor: pixel_distance(gray_frames[index], gray_frames[anchor])
    if method == "flow":
        return lambda index, anchor: flow_magnitude(gray_frames[index], gray_frames[anchor])
    raise ValueError(f"Unknown score method: {method}")


def offline_adjacent_scores(method: str, gray_frames: np.ndarray, scout_latents: np.ndarray) -> list[float]:
    scorer = causal_score_function(method, gray_frames, scout_latents)
    return [scorer(index, index - 1) for index in range(1, len(gray_frames))]


def _evaluate_refresh_schedule(
    *,
    method: str,
    variant: str,
    observed: np.ndarray,
    reasons: Sequence[str],
    scores: Sequence[float],
    timestamps: np.ndarray,
    reference_latents: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    high_error_threshold: float,
    target_budget: float | None = None,
    target_count: int | None = None,
) -> dict[str, Any]:
    state = hold_last_state_metrics(observed, reference_latents, timestamps)
    return {
        "status": "available",
        "reason": None,
        "method": method,
        "variant": variant,
        "target_observation_budget": target_budget,
        "target_observation_count": target_count,
        "actual_observation_count": int(observed.sum()),
        "observed": observed.tolist(),
        "refresh_reasons": list(reasons),
        "decision_scores": list(scores),
        **{key: value for key, value in state.items() if not key.startswith("per_frame") and key != "anchor_indices"},
        "per_frame_state_error": state["per_frame_state_error"],
        "per_frame_state_similarity": state["per_frame_state_similarity"],
        "anchor_indices": state["anchor_indices"],
        "high_error_episodes": high_error_episode_metrics(state["per_frame_state_error"], timestamps, high_error_threshold),
        "pose_staleness": pose_staleness_metrics(observed, positions, quaternions),
        "large_motion_coverage": large_motion_coverage(observed, positions, quaternions),
    }


def experiment2_selection_summary(operations: Sequence[dict[str, Any]], aucs: dict[str, dict[str, float]]) -> dict[str, Any]:
    budget_results = []
    mean_wins = 0
    p95_wins = 0
    sparse: dict[float, bool] = {}
    for budget in OBSERVATION_BUDGETS:
        rows = [row for row in operations if row["target_observation_budget"] == budget]
        minimum_mean = min(float(row["mean_state_error"]) for row in rows)
        minimum_p95 = min(float(row["p95_state_error"]) for row in rows)
        mean_winners = sorted(row["method"] for row in rows if float(row["mean_state_error"]) <= minimum_mean + FLOAT_TIE_TOLERANCE)
        p95_winners = sorted(row["method"] for row in rows if float(row["p95_state_error"]) <= minimum_p95 + FLOAT_TIE_TOLERANCE)
        mean_wins += int("jepa" in mean_winners)
        p95_wins += int("jepa" in p95_winners)
        sparse[budget] = "jepa" in mean_winners
        budget_results.append({"budget": budget, "mean_error_winner_set": mean_winners, "p95_error_winner_set": p95_winners, "jepa_mean_winner": "jepa" in mean_winners, "jepa_p95_winner": "jepa" in p95_winners})
    best_baseline_auc = min(aucs[method]["budget_error_auc"] for method in ("uniform", "pixel", "flow"))
    auc_pass = aucs["jepa"]["budget_error_auc"] <= best_baseline_auc + FLOAT_TIE_TOLERANCE
    win10 = sparse.get(0.10, False)
    win20 = sparse.get(0.20, False)
    conditions = {
        "mean_error_wins_at_least_3": {"passed": mean_wins >= 3, "actual": mean_wins, "required": 3, "reason": None if mean_wins >= 3 else "JEPA won fewer than three mean-error budget points."},
        "p95_error_wins_at_least_3": {"passed": p95_wins >= 3, "actual": p95_wins, "required": 3, "reason": None if p95_wins >= 3 else "JEPA won fewer than three P95-error budget points."},
        "sparse_budget_win": {"passed": win10 or win20, "actual": {"win_at_10_percent": win10, "win_at_20_percent": win20}, "reason": None if win10 or win20 else "JEPA did not win mean error at either 10% or 20%."},
        "error_auc_pass": {"passed": auc_pass, "actual": {"jepa": aucs["jepa"]["budget_error_auc"], "best_non_jepa": best_baseline_auc}, "reason": None if auc_pass else "JEPA budget error AUC exceeds the best non-JEPA baseline."},
    }
    return {
        "floating_tie_tolerance": FLOAT_TIE_TOLERANCE,
        "budget_winners": budget_results,
        "jepa_mean_error_wins": mean_wins,
        "jepa_p95_error_wins": p95_wins,
        "win_at_10_percent": win10,
        "win_at_20_percent": win20,
        "sparse_budget_win": win10 or win20,
        "auc_pass": auc_pass,
        "conditions": conditions,
        "experiment2_pass": all(item["passed"] for item in conditions.values()),
        "failed_conditions": [name for name, item in conditions.items() if not item["passed"]],
    }


def run_sparse_state_refresh(
    *,
    timestamps: np.ndarray,
    gray_frames: np.ndarray,
    scout_latents: np.ndarray,
    reference_latents: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    high_threshold = adjacent_reference_high_error_threshold(reference_latents)
    adjacent = {method: offline_adjacent_scores(method, gray_frames, scout_latents) for method in ("pixel", "flow", "jepa")}
    operations: list[dict[str, Any]] = []
    for budget in OBSERVATION_BUDGETS:
        target_count = half_up_target_count(budget, len(timestamps))
        for method in ("uniform", "pixel", "flow", "jepa"):
            observed = uniform_time_schedule(timestamps, target_count) if method == "uniform" else offline_topk_schedule(adjacent[method], timestamps, target_count)
            reasons = ["initial" if index == 0 else "offline_selected" if observed[index] else "skip" for index in range(len(observed))]
            scores = [float("nan"), *(adjacent.get(method, [float("nan")] * (len(observed) - 1)))]
            operations.append(_evaluate_refresh_schedule(method=method, variant=f"budget_{budget:g}", observed=observed, reasons=reasons, scores=scores, timestamps=timestamps, reference_latents=reference_latents, positions=positions, quaternions=quaternions, high_error_threshold=high_threshold, target_budget=budget, target_count=target_count))
    by_method = {method: sorted((row for row in operations if row["method"] == method), key=lambda row: row["target_observation_budget"]) for method in ("uniform", "pixel", "flow", "jepa")}
    aucs = {
        method: {
            "budget_error_auc": budget_curve_auc(OBSERVATION_BUDGETS, [row["mean_state_error"] for row in rows]),
            "error_direction": "lower_is_better",
            "budget_similarity_auc": budget_curve_auc(OBSERVATION_BUDGETS, [row["mean_state_similarity"] for row in rows]),
            "similarity_direction": "higher_is_better_auxiliary",
        }
        for method, rows in by_method.items()
    }
    selection = experiment2_selection_summary(operations, aucs)
    payload = {
        "experiment": "fixed_budget_offline_refresh_score_evaluation",
        "scientific_positioning": "offline ranking feasibility; fixed-budget score comparison; offline score-selection analysis",
        "score_definition": "adjacent local change: change(frame[t-1], frame[t])",
        "selection_uses": ["384 grayscale for Pixel/Flow", "z384 for JEPA"],
        "selection_does_not_use": ["pose", "z512", "task labels", "randomness"],
        "maximum_refresh_interval": None,
        "observation_budgets": list(OBSERVATION_BUDGETS),
        "target_count_rounding": "half-up: max(1, floor(budget*N + 0.5))",
        "high_error_threshold": {"value": high_threshold, "definition": "P95 adjacent z512 state error over complete sequence"},
        "auc_by_method": aucs,
        "selection_result": selection,
        "operations": operations,
    }
    write_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "budget_comparison.csv", [operation_csv_row(row, aucs[row["method"]]) for row in operations])
    save_sparse_refresh_figures(output_dir, payload)
    return payload


def split_time_segments(timestamps: Sequence[float], fraction: float = CALIBRATION_FRACTION) -> dict[str, Any]:
    times = np.asarray(timestamps, dtype=np.float64)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Time split requires at least two strictly increasing timestamps.")
    split = float(times[0] + fraction * (times[-1] - times[0]))
    calibration = np.flatnonzero(times <= split)
    evaluation = np.flatnonzero(times > split)
    return {
        "split_timestamp": split,
        "calibration_indices": calibration,
        "evaluation_indices": evaluation,
        "calibration_frame_count": int(len(calibration)),
        "evaluation_frame_count": int(len(evaluation)),
    }


def calibrate_causal_thresholds(
    timestamps: Sequence[float],
    adjacent_scores: Sequence[float],
    score_from_anchor: Callable[[int, int], float],
) -> dict[str, Any]:
    times = np.asarray(timestamps, dtype=np.float64)
    scores = np.asarray(adjacent_scores, dtype=np.float64)
    unique_count = int(len(np.unique(scores)))
    if len(times) < 2:
        return {"status": "unavailable", "reason": "calibration_requires_at_least_two_frames", "adjacent_score_count": len(scores), "unique_score_count": unique_count, "candidates": [], "unique_threshold_pair_count": 0, "selected": None}
    if len(scores) < 1 or len(scores) != len(times) - 1:
        return {"status": "unavailable", "reason": "calibration_requires_valid_adjacent_scores", "adjacent_score_count": len(scores), "unique_score_count": unique_count, "candidates": [], "unique_threshold_pair_count": 0, "selected": None}
    if unique_count < 2:
        return {"status": "unavailable", "reason": "calibration_scores_require_two_distinct_values", "adjacent_score_count": len(scores), "unique_score_count": unique_count, "candidates": [], "unique_threshold_pair_count": 0, "selected": None}
    candidates = []
    order = 0
    for low_q in ADAPTIVE_LOW_QUANTILE_CANDIDATES:
        for high_q in ADAPTIVE_HIGH_QUANTILE_CANDIDATES:
            if low_q >= high_q:
                continue
            low = float(np.quantile(scores, low_q))
            high = float(np.quantile(scores, high_q))
            observed, _, _ = causal_anchor_schedule(times, score_from_anchor, low, high)
            rate = float(observed.mean())
            candidates.append({"candidate_order": order, "low_quantile": low_q, "high_quantile": high_q, "low_threshold": low, "high_threshold": high, "calibration_observation_count": int(observed.sum()), "calibration_observation_rate": rate, "absolute_rate_error": abs(rate - ADAPTIVE_TARGET_OBSERVATION_RATE)})
            order += 1
    selected = min(candidates, key=lambda row: (row["absolute_rate_error"], 0 if row["calibration_observation_rate"] <= ADAPTIVE_TARGET_OBSERVATION_RATE else 1, row["calibration_observation_rate"], -row["high_quantile"], -row["low_quantile"], row["candidate_order"]))
    unique_pairs = {(row["low_threshold"], row["high_threshold"]) for row in candidates}
    return {"status": "available", "reason": None, "adjacent_score_count": len(scores), "unique_score_count": unique_count, "candidates": candidates, "unique_threshold_pair_count": len(unique_pairs), "selected": selected}


def operation_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("observation_rate", "mean_state_error", "p95_state_error")
    return all(float(left[key]) <= float(right[key]) for key in keys) and any(float(left[key]) < float(right[key]) for key in keys)


def experiment3_selection_summary(
    jepa: dict[str, Any] | None,
    matched_uniform: dict[str, Any] | None,
    pixel: dict[str, Any] | None,
) -> dict[str, Any]:
    rate_close = bool(jepa and abs(jepa["observation_rate"] - ADAPTIVE_TARGET_OBSERVATION_RATE) <= 0.02 + FLOAT_TIE_TOLERANCE)
    uniform_quality = bool(jepa and matched_uniform and jepa["mean_state_error"] < matched_uniform["mean_state_error"] and jepa["p95_state_error"] < matched_uniform["p95_state_error"])
    pixel_dominates = bool(jepa and pixel and operation_dominates(pixel, jepa))
    conditions = {
        "rate_close": {"passed": rate_close, "actual": None if not jepa else jepa["observation_rate"], "reason": None if rate_close else "JEPA evaluation rate is unavailable or outside 8%-12%."},
        "matched_uniform_quality": {"passed": uniform_quality, "actual": None if not (jepa and matched_uniform) else {"jepa_mean_error": jepa["mean_state_error"], "matched_uniform_mean_error": matched_uniform["mean_state_error"], "jepa_p95_error": jepa["p95_state_error"], "matched_uniform_p95_error": matched_uniform["p95_state_error"]}, "reason": None if uniform_quality else "JEPA does not strictly beat the equal-count Uniform baseline on both mean and P95 error."},
        "pixel_nondomination": {"passed": not pixel_dominates and bool(jepa and pixel), "actual": {"pixel_dominates_jepa": pixel_dominates}, "reason": None if (not pixel_dominates and jepa and pixel) else "Pixel calibration is unavailable or Pixel dominates JEPA."},
    }
    return {
        "conditions": conditions,
        "experiment3_pass": all(item["passed"] for item in conditions.values()),
        "failed_conditions": [name for name, item in conditions.items() if not item["passed"]],
    }


def _time_call(callable_: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = callable_()
    return result, (time.perf_counter() - start) * 1000.0


def benchmark_pipeline(
    encoder: Any,
    decoded_rgb: Sequence[np.ndarray],
    *,
    resolution: int,
    device: Any,
    warmups: int = TIMING_WARMUPS,
    repeats: int = TIMING_REPEATS,
) -> dict[str, Any]:
    import torch

    from research.scripts.common.video_models import autocast_context, synchronize

    def run_one(frame: np.ndarray) -> None:
        rgb, _ = build_resolution_rgb(frame, resolution)
        normalized = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        cpu_tensor = torch.from_numpy(normalized).permute(2, 0, 1).contiguous()
        gpu_tensor = cpu_tensor.unsqueeze(0).unsqueeze(2).to(device=device, non_blocking=True)
        with torch.inference_mode(), autocast_context(device):
            tokens = encoder(gpu_tensor)
        synchronize(device)
        del cpu_tensor, gpu_tensor, tokens

    for _ in range(warmups):
        for frame in decoded_rgb:
            run_one(frame)
    samples = []
    for _ in range(repeats):
        for frame in decoded_rgb:
            _, elapsed = _time_call(lambda frame=frame: run_one(frame))
            samples.append(elapsed)
    return {
        "resolution": resolution,
        "batch_size": 1,
        "warmup_passes": warmups,
        "timed_passes": repeats,
        "decoded_frame_count": len(decoded_rgb),
        "disk_io_included": False,
        "png_decode_included": False,
        "model_load_included": False,
        **timing_statistics(samples),
    }


def benchmark_pixel_decision(
    decoded_rgb: Sequence[np.ndarray],
    *,
    low_threshold: float,
    high_threshold: float,
    timestamps: Sequence[float] | None = None,
    warmups: int = TIMING_WARMUPS,
    repeats: int = TIMING_REPEATS,
) -> dict[str, Any]:
    times = np.asarray(timestamps if timestamps is not None else np.arange(len(decoded_rgb), dtype=float) * 0.1)
    def pass_once(record: bool) -> tuple[list[float], list[float]]:
        decision_samples: list[float] = []
        scheduler_samples: list[float] = []
        anchor_gray: np.ndarray | None = None
        anchor_index = 0
        for index, frame in enumerate(decoded_rgb):
            start = time.perf_counter()
            current_gray, _ = build_scout_gray(frame)
            score = 0.0 if anchor_gray is None else pixel_distance(current_gray, anchor_gray)
            state = "dynamic" if score >= high_threshold else "normal" if score >= low_threshold else "stable"
            decision_ms = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            elapsed = float(times[index] - times[anchor_index])
            refresh = index == 0 or state == "dynamic" or (state == "normal" and elapsed >= NORMAL_TIMEOUT_SECONDS) or (state == "stable" and elapsed >= STABLE_TIMEOUT_SECONDS)
            if refresh:
                anchor_gray = current_gray
                anchor_index = index
            _ = anchor_gray if not refresh else current_gray
            scheduler_ms = (time.perf_counter() - start) * 1000.0
            if record:
                decision_samples.append(decision_ms)
                scheduler_samples.append(scheduler_ms)
        return decision_samples, scheduler_samples
    for _ in range(warmups):
        pass_once(False)
    decision: list[float] = []
    scheduler: list[float] = []
    for _ in range(repeats):
        current_decision, current_scheduler = pass_once(True)
        decision.extend(current_decision)
        scheduler.extend(current_scheduler)
    return {"decision": timing_statistics(decision), "scheduler_fusion": timing_statistics(scheduler), "anchor_gray_cached": True}


def benchmark_latent_decision(
    latents: np.ndarray,
    *,
    low_threshold: float,
    high_threshold: float,
    timestamps: Sequence[float] | None = None,
    warmups: int = TIMING_WARMUPS,
    repeats: int = TIMING_REPEATS,
) -> dict[str, Any]:
    times = np.asarray(timestamps if timestamps is not None else np.arange(len(latents), dtype=float) * 0.1)
    def pass_once(record: bool) -> tuple[list[float], list[float]]:
        decision_samples: list[float] = []
        scheduler_samples: list[float] = []
        anchor = 0
        maintained = latents[0]
        for index in range(len(latents)):
            start = time.perf_counter()
            score = 0.0 if index == 0 else latent_distance(latents[index], latents[anchor])
            state = "dynamic" if score >= high_threshold else "normal" if score >= low_threshold else "stable"
            decision_ms = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            elapsed = float(times[index] - times[anchor])
            refresh = index == 0 or state == "dynamic" or (state == "normal" and elapsed >= NORMAL_TIMEOUT_SECONDS) or (state == "stable" and elapsed >= STABLE_TIMEOUT_SECONDS)
            if refresh:
                anchor = index
                maintained = latents[index]
            _ = maintained
            scheduler_ms = (time.perf_counter() - start) * 1000.0
            if record:
                decision_samples.append(decision_ms)
                scheduler_samples.append(scheduler_ms)
        return decision_samples, scheduler_samples
    for _ in range(warmups):
        pass_once(False)
    decision: list[float] = []
    scheduler: list[float] = []
    for _ in range(repeats):
        current_decision, current_scheduler = pass_once(True)
        decision.extend(current_decision)
        scheduler.extend(current_scheduler)
    return {"decision": timing_statistics(decision), "scheduler_fusion": timing_statistics(scheduler), "anchor_latent_cached": True}


def benchmark_schedule_replay(observed: Sequence[bool], *, warmups: int = TIMING_WARMUPS, repeats: int = TIMING_REPEATS) -> dict[str, Any]:
    def pass_once(record: bool) -> list[float]:
        samples = []
        maintained = 0
        for index, refresh in enumerate(observed):
            start = time.perf_counter()
            if refresh:
                maintained = index
            _ = maintained
            if record:
                samples.append((time.perf_counter() - start) * 1000.0)
        return samples
    for _ in range(warmups):
        pass_once(False)
    samples: list[float] = []
    for _ in range(repeats):
        samples.extend(pass_once(True))
    return timing_statistics(samples)


def run_latency_microbenchmark(
    *,
    encoder: Any,
    device: Any,
    frames: Sequence[AssociatedFrame],
    scout_latents: np.ndarray,
    pixel_thresholds: dict[str, float],
    jepa_thresholds: dict[str, float],
) -> dict[str, Any]:
    indices = deterministic_indices(len(frames), TIMING_MAX_FRAMES)
    decoded = []
    for index in indices:
        with Image.open(frames[index].rgb_path) as image:
            decoded.append(np.array(image.convert("RGB"), dtype=np.uint8, copy=True))
    selected_latents = scout_latents[indices]
    selected_timestamps = np.asarray([frames[index].rgb_timestamp for index in indices], dtype=np.float64)
    pipeline_384 = benchmark_pipeline(encoder, decoded, resolution=SCOUT_RESOLUTION, device=device)
    pipeline_512 = benchmark_pipeline(encoder, decoded, resolution=REFERENCE_RESOLUTION, device=device)
    pixel = benchmark_pixel_decision(decoded, low_threshold=pixel_thresholds["low"], high_threshold=pixel_thresholds["high"], timestamps=selected_timestamps)
    jepa = benchmark_latent_decision(selected_latents, low_threshold=jepa_thresholds["low"], high_threshold=jepa_thresholds["high"], timestamps=selected_timestamps)
    uniform_observed = uniform_time_schedule(selected_timestamps, half_up_target_count(ADAPTIVE_TARGET_OBSERVATION_RATE, len(selected_timestamps)))
    uniform_scheduler = benchmark_schedule_replay(uniform_observed)
    return {
        "protocol": {
            "batch_size": 1,
            "sampled_frame_indices": indices,
            "warmup_passes": TIMING_WARMUPS,
            "timed_passes": TIMING_REPEATS,
            "decoded_in_memory": True,
            "disk_io_png_decode_model_load_excluded": True,
            "per_frame_sample_aggregation": True,
        },
        "pipeline_384": pipeline_384,
        "pipeline_512": pipeline_512,
        "pixel": pixel,
        "jepa": jepa,
        "uniform_scheduler": uniform_scheduler,
    }


def run_adaptive_observation(
    *,
    timestamps: np.ndarray,
    gray_frames: np.ndarray,
    scout_latents: np.ndarray,
    reference_latents: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    timing: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    split = split_time_segments(timestamps)
    calibration_indices = split["calibration_indices"]
    evaluation_indices = split["evaluation_indices"]
    calibration_times = timestamps[calibration_indices]
    evaluation_times = timestamps[evaluation_indices]
    high_threshold = adjacent_reference_high_error_threshold(reference_latents)

    calibrations: dict[str, Any] = {}
    for method in ("pixel", "jepa"):
        calibration_gray = gray_frames[calibration_indices]
        calibration_scout = scout_latents[calibration_indices]
        adjacent = offline_adjacent_scores(method, calibration_gray, calibration_scout) if len(calibration_indices) >= 2 else []
        scorer = causal_score_function(method, calibration_gray, calibration_scout)
        calibrations[method] = calibrate_causal_thresholds(calibration_times, adjacent, scorer)

    operations: list[dict[str, Any]] = []
    evaluation_available = len(evaluation_indices) >= 2
    if evaluation_available:
        eval_reference = reference_latents[evaluation_indices]
        eval_positions = positions[evaluation_indices]
        eval_quaternions = quaternions[evaluation_indices]
        eval_gray = gray_frames[evaluation_indices]
        eval_scout = scout_latents[evaluation_indices]
        target_count = half_up_target_count(ADAPTIVE_TARGET_OBSERVATION_RATE, len(evaluation_indices))
        target_observed = uniform_time_schedule(evaluation_times, target_count)
        operations.append(_evaluate_refresh_schedule(method="uniform_target_10", variant="uniform_target_10", observed=target_observed, reasons=["initial" if index == 0 else "uniform_selected" if target_observed[index] else "skip" for index in range(len(target_observed))], scores=[float("nan")] * len(target_observed), timestamps=evaluation_times, reference_latents=eval_reference, positions=eval_positions, quaternions=eval_quaternions, high_error_threshold=high_threshold, target_budget=ADAPTIVE_TARGET_OBSERVATION_RATE, target_count=target_count))
        for method in ("pixel", "jepa"):
            calibration = calibrations[method]
            if calibration["status"] != "available":
                operations.append({"status": "unavailable", "reason": calibration["reason"], "method": f"{method}_adaptive", "variant": f"{method}_adaptive"})
                continue
            selected = calibration["selected"]
            observed, reasons, scores = causal_anchor_schedule(evaluation_times, causal_score_function(method, eval_gray, eval_scout), selected["low_threshold"], selected["high_threshold"])
            operations.append(_evaluate_refresh_schedule(method=f"{method}_adaptive", variant=f"{method}_adaptive", observed=observed, reasons=reasons, scores=scores, timestamps=evaluation_times, reference_latents=eval_reference, positions=eval_positions, quaternions=eval_quaternions, high_error_threshold=high_threshold))
        jepa = next(row for row in operations if row["method"] == "jepa_adaptive")
        if jepa["status"] == "available":
            matched_count = int(jepa["observation_count"])
            matched_observed = uniform_time_schedule(evaluation_times, matched_count)
            matched = _evaluate_refresh_schedule(method="uniform_matched_jepa", variant="uniform_matched_jepa", observed=matched_observed, reasons=["initial" if index == 0 else "uniform_selected" if matched_observed[index] else "skip" for index in range(len(matched_observed))], scores=[float("nan")] * len(matched_observed), timestamps=evaluation_times, reference_latents=eval_reference, positions=eval_positions, quaternions=eval_quaternions, high_error_threshold=high_threshold, target_count=matched_count)
        else:
            matched = {"status": "unavailable", "reason": "JEPA adaptive count unavailable", "method": "uniform_matched_jepa", "variant": "uniform_matched_jepa"}
        operations.insert(1, matched)
    else:
        for method in ("uniform_target_10", "uniform_matched_jepa", "pixel_adaptive", "jepa_adaptive"):
            operations.append({"status": "unavailable", "reason": "evaluation_requires_at_least_two_frames", "method": method, "variant": method})

    available = {row["method"]: row for row in operations if row["status"] == "available"}
    t384 = float(timing["pipeline_384"]["median_ms"])
    t512 = float(timing["pipeline_512"]["median_ms"])
    for operation in operations:
        if operation["status"] != "available":
            continue
        method = operation["method"]
        if method.startswith("uniform"):
            estimate = estimate_policy_latency(observation_rate=operation["observation_rate"], pipeline_512_ms=t512, scheduler_ms=float(timing["uniform_scheduler"]["median_ms"]))
        elif method == "pixel_adaptive":
            estimate = estimate_policy_latency(observation_rate=operation["observation_rate"], pipeline_512_ms=t512, scheduler_ms=float(timing["pixel"]["scheduler_fusion"]["median_ms"]), decision_ms=float(timing["pixel"]["decision"]["median_ms"]))
        else:
            estimate = estimate_policy_latency(observation_rate=operation["observation_rate"], pipeline_512_ms=t512, scheduler_ms=float(timing["jepa"]["scheduler_fusion"]["median_ms"]), pipeline_384_ms=t384, decision_ms=float(timing["jepa"]["decision"]["median_ms"]))
        operation.update(estimate)
    decision_cost = float(timing["jepa"]["decision"]["median_ms"]) + float(timing["jepa"]["scheduler_fusion"]["median_ms"])
    break_even = break_even_refresh_rate(t384, t512, decision_cost)
    jepa = available.get("jepa_adaptive")
    matched = available.get("uniform_matched_jepa")
    pixel = available.get("pixel_adaptive")
    selection = experiment3_selection_summary(jepa, matched, pixel)
    conditions = selection["conditions"]
    experiment3_pass = selection["experiment3_pass"] and all(calibrations[method]["status"] == "available" for method in calibrations) and evaluation_available
    payload = {
        "experiment": "calibrated_causal_adaptive_observation",
        "score_definition": "causal anchor change: change(last_refresh_anchor,current_frame)",
        "time_split": {key: _jsonable(value) for key, value in split.items()},
        "calibration_fraction": CALIBRATION_FRACTION,
        "target_observation_rate": ADAPTIVE_TARGET_OBSERVATION_RATE,
        "timeouts_seconds": {"stable": STABLE_TIMEOUT_SECONDS, "normal": NORMAL_TIMEOUT_SECONDS},
        "calibrations": calibrations,
        "operations": operations,
        "timing": timing,
        "break_even_refresh_rate": break_even,
        "break_even_has_positive_margin": break_even > 0,
        "selection_conditions": conditions,
        "experiment3_pass": experiment3_pass,
        "failed_conditions": [name for name, item in conditions.items() if not item["passed"]] + [f"{method}_calibration" for method, value in calibrations.items() if value["status"] != "available"],
        "refresh_cost_note": "JEPA refresh frames include both the 384 scout and 512 reference costs; no cross-resolution feature reuse.",
    }
    write_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "policy_comparison.csv", [adaptive_csv_row(operation) for operation in operations])
    save_adaptive_figures(output_dir, payload)
    return payload


def build_feasibility_verdict(sparse: dict[str, Any], adaptive: dict[str, Any]) -> dict[str, Any]:
    experiment2_pass = bool(sparse["selection_result"]["experiment2_pass"])
    experiment3_pass = bool(adaptive["experiment3_pass"])
    selection_pass = experiment2_pass and experiment3_pass
    operations = {operation["method"]: operation for operation in adaptive["operations"] if operation["status"] == "available"}
    jepa_latency = float(operations["jepa_adaptive"]["estimated_latency_ms_per_frame"]) if "jepa_adaptive" in operations else float("inf")
    full_latency = float(adaptive["timing"]["pipeline_512"]["median_ms"])
    speed_pass = jepa_latency < full_latency
    if selection_pass and speed_pass:
        label = "Go"
    elif selection_pass:
        label = "Conditional Go"
    else:
        label = "No-Go"
    failed = []
    if not experiment2_pass:
        failed.append("experiment2_pass")
    if not experiment3_pass:
        failed.append("experiment3_pass")
    if not speed_pass:
        failed.append("speed_pass")
    return {
        "label": label,
        "selection_pass": selection_pass,
        "selection_conditions": {
            "experiment2_pass": {"passed": experiment2_pass, "actual": sparse["selection_result"], "reason": None if experiment2_pass else "Fixed-budget offline refresh-score criteria failed."},
            "experiment3_pass": {"passed": experiment3_pass, "actual": adaptive["selection_conditions"], "reason": None if experiment3_pass else "Calibrated causal evaluation criteria failed."},
        },
        "speed_pass": {
            "passed": speed_pass,
            "jepa_adaptive_latency_ms": jepa_latency,
            "full_512_latency_ms": full_latency,
            "reason": None if speed_pass else "JEPA adaptive estimated latency is not below the full ViT-B@512 pipeline.",
        },
        "failed_conditions": failed,
        "rules": {
            "go": "selection_pass AND speed_pass",
            "conditional_go": "selection_pass AND NOT speed_pass",
            "no_go": "NOT selection_pass",
        },
    }


def operation_csv_row(operation: dict[str, Any], auc: dict[str, float] | None = None) -> dict[str, Any]:
    if operation.get("status") != "available":
        return {"method": operation["method"], "variant": operation["variant"], "status": "unavailable", "reason": operation.get("reason")}
    staleness = operation["pose_staleness"]["staleness_summary"]
    return {
        "method": operation["method"],
        "variant": operation["variant"],
        "status": "available",
        "reason": None,
        "target_observation_budget": operation.get("target_observation_budget"),
        "target_observation_count": operation.get("target_observation_count"),
        "observation_count": operation["observation_count"],
        "observation_rate": operation["observation_rate"],
        "observation_reduction_ratio": operation["observation_reduction_ratio"],
        "mean_state_error": operation["mean_state_error"],
        "p90_state_error": operation["p90_state_error"],
        "p95_state_error": operation["p95_state_error"],
        "mean_state_similarity": operation["mean_state_similarity"],
        "p95_net_translation_m": staleness["net_translation_m"]["p95"],
        "p95_net_rotation_deg": staleness["net_rotation_deg"]["p95"],
        "p95_cumulative_translation_m": staleness["cumulative_translation_m"]["p95"],
        "p95_cumulative_rotation_deg": staleness["cumulative_rotation_deg"]["p95"],
        "translation_large_motion_recall": operation["large_motion_coverage"]["translation"]["recall"],
        "rotation_large_motion_recall": operation["large_motion_coverage"]["rotation"]["recall"],
        "budget_error_auc": None if auc is None else auc["budget_error_auc"],
        "error_auc_direction": None if auc is None else "lower_is_better",
        "budget_similarity_auc": None if auc is None else auc["budget_similarity_auc"],
        "similarity_auc_direction": None if auc is None else "higher_is_better_auxiliary",
        "max_high_error_episode_seconds": operation["high_error_episodes"]["max_duration_seconds"],
        "p95_high_error_episode_seconds": operation["high_error_episodes"]["p95_duration_seconds"],
    }


def adaptive_csv_row(operation: dict[str, Any]) -> dict[str, Any]:
    row = operation_csv_row(operation)
    if operation.get("status") == "available":
        row.update(
        {
            "estimated_latency_ms_per_frame": operation["estimated_latency_ms_per_frame"],
            "estimated_fps": operation["estimated_fps"],
            "speedup_vs_full_512": operation["speedup_vs_full_512"],
        })
    return row


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: _jsonable(row.get(key)) for key in fieldnames} for row in rows])


def _get_plt():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_ego_motion_figures(output_dir: Path, metrics: dict[str, Any]) -> None:
    plt = _get_plt()
    horizons = list(TIME_HORIZONS_SECONDS)
    rows = metrics["horizon_statistics"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    axes[0].plot(horizons, [rows[f"{value:g}"]["jepa384_distance"]["mean"] for value in horizons], marker="o", label="ViT-B@384")
    axes[0].plot(horizons, [rows[f"{value:g}"]["jepa512_distance"]["mean"] for value in horizons], marker="o", label="ViT-B@512")
    axes[1].plot(horizons, [rows[f"{value:g}"]["pixel_distance"]["mean"] for value in horizons], marker="o", color="#ff7f0e")
    axes[2].plot(horizons, [rows[f"{value:g}"]["flow_magnitude"]["mean"] for value in horizons], marker="o", color="#2ca02c")
    for axis, title, ylabel in zip(axes, ("Scout-reference state change", "Pixel change", "Farneback flow"), ("Mean cosine distance", "Mean normalized difference", "Mean flow magnitude")):
        axis.set_xscale("log"); axis.set(xlabel="Target horizon (seconds)", ylabel=ylabel, title=title); axis.set_xticks(horizons, [f"{value:g}" for value in horizons]); axis.grid(alpha=0.25)
    axes[0].legend()
    fig.savefig(output_dir / "time_horizon_curves.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, key, title in ((axes[0], "translation_conditioned_distance", "Translation-conditioned"), (axes[1], "rotation_conditioned_distance", "Rotation-conditioned")):
        bins = metrics[key]
        for state, label in (("jepa384_distance", "ViT-B@384"), ("jepa512_distance", "ViT-B@512")):
            axis.plot([item["quantile_bin"] for item in bins], [item[state]["mean"] or 0.0 for item in bins], marker="o", label=label)
        axis.set(xlabel="Motion quantile bin", ylabel="Mean cosine distance", title=title); axis.legend(); axis.grid(alpha=0.2)
    fig.savefig(output_dir / "motion_conditioned_distance.png", dpi=170); plt.close(fig)


def save_sparse_refresh_figures(output_dir: Path, metrics: dict[str, Any]) -> None:
    plt = _get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for method in ("uniform", "pixel", "flow", "jepa"):
        rows = sorted([row for row in metrics["operations"] if row["method"] == method], key=lambda row: row["target_observation_budget"])
        budgets = [row["target_observation_budget"] for row in rows]
        axes[0].plot(budgets, [row["mean_state_error"] for row in rows], marker="o", label=method)
        axes[1].plot(budgets, [row["p95_state_error"] for row in rows], marker="o", label=method)
    for axis, title, ylabel in ((axes[0], "Mean state error", "Mean state error"), (axes[1], "P95 state error", "P95 state error")):
        axis.set(xlabel="High-quality observation budget", ylabel=ylabel, title=title); axis.set_xticks(OBSERVATION_BUDGETS, [f"{value:.0%}" for value in OBSERVATION_BUDGETS]); axis.axvline(0.10, color="grey", ls="--", alpha=0.5); axis.axvline(0.20, color="grey", ls="--", alpha=0.5); axis.grid(alpha=0.25); axis.legend()
    fig.savefig(output_dir / "budget_quality_curves.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for method in ("uniform", "pixel", "flow", "jepa"):
        rows = sorted([row for row in metrics["operations"] if row["method"] == method], key=lambda row: row["target_observation_budget"])
        axes[0].plot([row["target_observation_budget"] for row in rows], [row["pose_staleness"]["staleness_summary"]["cumulative_translation_m"]["p95"] for row in rows], marker="o", label=method)
        axes[1].plot([row["target_observation_budget"] for row in rows], [row["pose_staleness"]["staleness_summary"]["cumulative_rotation_deg"]["p95"] for row in rows], marker="o", label=method)
    axes[0].set(xlabel="Observation budget", ylabel="P95 cumulative translation (m)", title="Primary translation coverage")
    axes[1].set(xlabel="Observation budget", ylabel="P95 cumulative rotation (deg)", title="Primary rotation coverage")
    for axis in axes: axis.grid(alpha=0.25); axis.legend()
    fig.savefig(output_dir / "pose_coverage.png", dpi=170); plt.close(fig)


def save_adaptive_figures(output_dir: Path, metrics: dict[str, Any]) -> None:
    plt = _get_plt()
    fig, axis = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    for operation in metrics["operations"]:
        if operation["status"] == "available":
            size = 110 if operation["method"] == "uniform_matched_jepa" else 75
            axis.scatter(operation["estimated_latency_ms_per_frame"], operation["mean_state_error"], s=size, label=operation["method"])
    axis.set(xlabel="Estimated latency (ms/frame)", ylabel="Mean state error", title="Evaluation quality-compute trade-off")
    axis.grid(alpha=0.25); axis.legend()
    fig.savefig(output_dir / "quality_compute_pareto.png", dpi=170); plt.close(fig)
    available = [operation for operation in metrics["operations"] if operation["status"] == "available"]
    fig, axes = plt.subplots(max(1, len(available)), 1, figsize=(12, 6.5), constrained_layout=True)
    for axis, operation in zip(np.atleast_1d(axes), available):
        observed = np.asarray(operation["observed"], dtype=bool)
        indices = np.arange(len(observed))
        axis.scatter(indices[~observed], np.zeros((~observed).sum()), s=4, color="#aaaaaa")
        axis.scatter(indices[observed], np.zeros(observed.sum()), s=22, marker="|", color="#d62728")
        axis.set_yticks([]); axis.set_ylabel(operation["method"], rotation=0, ha="right"); axis.grid(axis="x", alpha=0.2)
    np.atleast_1d(axes)[-1].set_xlabel("Evaluation-segment frame index")
    fig.suptitle(f"Evaluation-only timelines; split timestamp={metrics['time_split']['split_timestamp']:.6f}")
    fig.savefig(output_dir / "observation_timeline.png", dpi=160); plt.close(fig)


def require_cuda() -> tuple[Any, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 4 full run requires CUDA; --prepare-only only validates TUM RGB-D.")
    torch.cuda.set_device(0)
    return torch.device("cuda"), torch.cuda.get_device_name(0)


def _normalized_tensor(rgb: np.ndarray) -> Any:
    import torch

    normalized = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(normalized).permute(2, 0, 1).contiguous()


def extract_global_states(
    *,
    encoder: Any,
    device: Any,
    frames: Sequence[AssociatedFrame],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as torch_functional

    from research.scripts.common.video_models import autocast_context, peak_memory_mb, reset_peak_memory, synchronize

    scout_batches: list[np.ndarray] = []
    reference_batches: list[np.ndarray] = []
    gray_frames: list[np.ndarray] = []
    forward_seconds = {SCOUT_RESOLUTION: 0.0, REFERENCE_RESOLUTION: 0.0}
    token_counts: dict[int, int] = {}
    output_dtype = "unknown"
    reset_peak_memory(device)
    for start in range(0, len(frames), FEATURE_CHUNK_SIZE):
        chunk = frames[start:start + FEATURE_CHUNK_SIZE]
        views = []
        for frame in chunk:
            with Image.open(frame.rgb_path) as image:
                views.append(build_source_views(image))
        gray_frames.extend(view.gray384 for view in views)
        for resolution, collector in ((SCOUT_RESOLUTION, scout_batches), (REFERENCE_RESOLUTION, reference_batches)):
            tensors = [_normalized_tensor(view.rgb384 if resolution == SCOUT_RESOLUTION else view.rgb512) for view in views]
            cpu_batch = torch.stack(tensors, dim=0)
            gpu_batch = cpu_batch.unsqueeze(2).to(device=device, non_blocking=True)
            synchronize(device)
            forward_start = time.perf_counter()
            with torch.inference_mode(), autocast_context(device):
                tokens = encoder(gpu_batch)
            synchronize(device)
            forward_seconds[resolution] += time.perf_counter() - forward_start
            if tokens.ndim != 3:
                raise RuntimeError(f"Unexpected ViT-B token shape: {tuple(tokens.shape)}")
            token_counts[resolution] = int(tokens.shape[1])
            output_dtype = str(tokens.dtype)
            pooled = torch_functional.normalize(tokens.float().mean(dim=1), dim=-1)
            collector.append(pooled.cpu().numpy().astype(np.float32))
            del cpu_batch, gpu_batch, tokens, pooled
    return {
        "z384": np.concatenate(scout_batches, axis=0),
        "z512": np.concatenate(reference_batches, axis=0),
        "gray384": np.stack(gray_frames, axis=0),
        "model_metadata": {
            "model": "V-JEPA 2.1 ViT-B",
            "dtype": output_dtype,
            "scout_resolution": SCOUT_RESOLUTION,
            "reference_resolution": REFERENCE_RESOLUTION,
            "scout_tokens_per_frame": token_counts[SCOUT_RESOLUTION],
            "reference_tokens_per_frame": token_counts[REFERENCE_RESOLUTION],
            "feature_dimension": int(scout_batches[0].shape[-1]),
            "global_state": "final dense patch tokens -> spatial mean pooling -> L2 normalization",
            "dense_token_cache_saved": False,
            "peak_gpu_memory_mb": peak_memory_mb(device),
            "forward_seconds": {str(key): value for key, value in forward_seconds.items()},
        },
    }


def output_file_list(output_dir: Path, *, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    return sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.relative_to(output_dir).as_posix() not in excluded
    )


def validate_casefold_unique_paths(paths: Iterable[str]) -> None:
    folded: dict[str, str] = {}
    for path in paths:
        key = path.casefold()
        previous = folded.get(key)
        if previous is not None and previous != path:
            raise RuntimeError(f"Phase 4 artifact paths collide case-insensitively: {previous!r}, {path!r}")
        folded[key] = path


def validate_expected_output(output_dir: Path) -> None:
    actual = set(output_file_list(output_dir))
    missing = EXPECTED_OUTPUT_FILES - actual
    unexpected = actual - EXPECTED_OUTPUT_FILES
    if missing or unexpected:
        raise RuntimeError(f"Unexpected Phase 4 artifact set: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    validate_casefold_unique_paths(actual)


def prepare_staged_output(staged_dir: Path = PHASE4_STAGED_OUTPUT_DIR) -> Path:
    staged_dir.parent.mkdir(parents=True, exist_ok=True)
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir()
    for directory in ("ego_motion_consistency", "sparse_state_refresh", "adaptive_observation"):
        (staged_dir / directory).mkdir()
    return staged_dir


def publish_staged_output(staged_dir: Path, final_dir: Path = PHASE4_OUTPUT_DIR) -> None:
    validate_expected_output(staged_dir)
    backup = final_dir.with_name(f".{final_dir.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if final_dir.exists():
        final_dir.rename(backup)
    try:
        staged_dir.rename(final_dir)
    except Exception:
        if backup.exists() and not final_dir.exists():
            backup.rename(final_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def validate_manifest_paths(output_dir: Path, entries: Sequence[dict[str, Any]]) -> None:
    for entry in entries:
        relative = str(entry["relative_path"])
        path = output_dir / relative
        if not path.is_file():
            raise RuntimeError(f"Manifest references missing artifact: {relative}")
        if int(entry["size_bytes"]) != path.stat().st_size or entry["sha256"] != sha256_file(path):
            raise RuntimeError(f"Manifest integrity mismatch for artifact: {relative}")


def artifact_records(output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for relative in output_file_list(output_dir, exclude={"manifest.json"}):
        path = output_dir / relative
        records.append({"relative_path": relative, "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return records


def collect_source_provenance() -> dict[str, Any]:
    records = []
    for relative in PHASE4_SOURCE_PATHS:
        path = REPO_ROOT / relative
        if path.is_file():
            records.append({"path": relative, "sha256": sha256_file(path)})
    combined = "".join(f"{record['path']}\n{record['sha256']}\n" for record in records).encode("utf-8")
    return {"phase_source_files": records, "phase_source_fingerprint": hashlib.sha256(combined).hexdigest()}


def collect_git_provenance() -> dict[str, Any]:
    import subprocess

    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_worktree_dirty": None, "git_status_porcelain": []}
    return {"git_commit": commit, "git_worktree_dirty": bool(status), "git_status_porcelain": status}


def build_summary(
    *,
    prepared: PreparedTumSequence,
    ego: dict[str, Any],
    sparse: dict[str, Any],
    adaptive: dict[str, Any],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    adaptive_rows = {
        operation["method"]: {
            "status": operation["status"],
            "observation_count": operation["observation_count"],
            "observation_rate": operation["observation_rate"],
            "mean_state_error": operation["mean_state_error"],
            "p95_state_error": operation["p95_state_error"],
            "estimated_latency_ms_per_frame": operation["estimated_latency_ms_per_frame"],
            "speedup_vs_full_512": operation["speedup_vs_full_512"],
        }
        for operation in adaptive["operations"] if operation["status"] == "available"
    }
    return {
        "phase": PHASE4_NAME,
        "dataset": {
            key: prepared.manifest[key]
            for key in (
                "sequence_name", "rgb_entry_count", "valid_rgb_file_count", "groundtruth_entry_count",
                "associated_frame_count", "dropped_frame_count", "duration_seconds",
                "mean_association_delta_seconds", "max_association_delta_seconds",
            )
        },
        "experiment_1": {
            "horizon_statistics": ego["horizon_statistics"],
            "scout_reference_consistency": ego["overall_scout_reference_consistency"],
            "correlations": ego["correlations"],
        },
        "experiment_2": {
            "auc_by_method": sparse["auc_by_method"],
            "selection_result": sparse["selection_result"],
            "high_error_threshold": sparse["high_error_threshold"],
            "jepa_pose_coverage": [
                {
                    "variant": operation["variant"],
                    "staleness_summary": operation["pose_staleness"]["staleness_summary"],
                    "large_motion_coverage": operation["large_motion_coverage"],
                }
                for operation in sparse["operations"] if operation["method"] == "jepa"
            ],
        },
        "experiment_3": {
            "time_split": adaptive["time_split"],
            "calibrations": adaptive["calibrations"],
            "policies": adaptive_rows,
            "selection_conditions": adaptive["selection_conditions"],
            "experiment3_pass": adaptive["experiment3_pass"],
            "break_even_refresh_rate": adaptive["break_even_refresh_rate"],
        },
        "verdict": verdict,
    }


def write_report(path: Path, *, summary: dict[str, Any], model_metadata: dict[str, Any]) -> None:
    verdict = summary["verdict"]
    conditions = verdict["selection_conditions"]
    condition_lines = [
        f"- `{name}`：{'通过' if item['passed'] else '失败'}；原因：{item['reason'] or '无'}。"
        for name, item in conditions.items()
    ]
    experiment2 = summary["experiment_2"]["selection_result"]
    experiment3 = summary["experiment_3"]
    experiment2_lines = [
        f"- `{name}`：{'通过' if item['passed'] else '失败'}；实际值：`{item['actual']}`；原因：{item['reason'] or '无'}。"
        for name, item in experiment2["conditions"].items()
    ]
    experiment3_lines = [
        f"- `{name}`：{'通过' if item['passed'] else '失败'}；实际值：`{item['actual']}`；原因：{item['reason'] or '无'}。"
        for name, item in experiment3["selection_conditions"].items()
    ]
    calibration_lines = [
        f"- `{method}` calibration：`{item['status']}`；unique scores={item['unique_score_count']}，unique threshold pairs={item['unique_threshold_pair_count']}；原因：{item['reason'] or '无'}；全部候选：`{item['candidates']}`。"
        for method, item in experiment3["calibrations"].items()
    ]
    lines = [
        "# Phase 4：TUM RGB-D 第一人称视觉状态刷新可行性报告",
        "",
        "## 实验范围",
        "",
        "本阶段只验证全局视觉状态刷新可行性，不实现区域刷新、SLAM、控制或部署系统。ViT-B@384 scout 与 ViT-B@512 reference 使用同一个原始中心正方形视野。",
        "",
        "## 数据与状态",
        "",
        f"固定序列 `{summary['dataset']['sequence_name']}`，关联 {summary['dataset']['associated_frame_count']} 帧，持续 {summary['dataset']['duration_seconds']:.3f} 秒。标称采集频率只作数据背景；实验不假设严格固定帧率，全部时间尺度由真实 RGB timestamp 决定。",
        f"状态定义：{model_metadata['global_state']}。",
        "",
        "## Experiment 1：真实时间尺度与 scout-reference 一致性",
        "",
        f"目标时间尺度固定为 {list(TIME_HORIZONS_SECONDS)} 秒，匹配容差 {TIME_PAIR_MATCH_TOLERANCE_SECONDS:.2f} 秒。报告同时比较 ViT-B@384 scout 与 ViT-B@512 reference 的距离、相似度、排序相关和绝对距离差，并分析四类位姿运动相关性。零方差或样本不足时保留 unavailable。",
        "",
        "## Experiment 2：固定预算下的离线刷新评分可行性验证",
        "",
        f"预算固定为 {list(OBSERVATION_BUDGETS)}，计数使用 half-up：`max(1, floor(budget*N + 0.5))`。Pixel、Flow 与 JEPA 只对完整序列的相邻帧变化分数做 offline ranking feasibility / fixed-budget score comparison；不使用 pose、z512、任务标签或周期性强制刷新，也不将结果表述为理论最优选择。主要 AUC 为 mean state error 的归一化梯形面积（越低越好）；similarity AUC 仅为辅助。Experiment 2：{'通过' if experiment2['experiment2_pass'] else '失败'}。",
        f"各方法 AUC：`{summary['experiment_2']['auc_by_method']}`。逐预算 winner sets（含 10%/20% 分开结果）：`{experiment2['budget_winners']}`。",
        *experiment2_lines,
        "高误差 episode 使用真实帧时间支持区间：非末帧代表 `[t_i,t_{i+1})`，段末帧使用该段相邻 timestamp 差的中位数；单帧段标记 degenerate。",
        "累计平移和累计旋转是未刷新期间自运动覆盖的主要指标；net 位姿差只作为终点差异辅助指标。",
        "",
        "## Experiment 3：校准后因果自适应观测",
        "",
        f"按真实时间前 {CALIBRATION_FRACTION:.0%} 校准、后 {1-CALIBRATION_FRACTION:.0%} evaluation；在线分数只比较当前帧与最近 refresh anchor。Stable/Normal timeout 分别为 {STABLE_TIMEOUT_SECONDS:.1f}/{NORMAL_TIMEOUT_SECONDS:.1f} 秒。主要公平质量比较使用与 JEPA 实际观测数量严格一致的 Uniform-matched-JEPA；Uniform-target-10% 仅作固定目标参考。Experiment 3：{'通过' if experiment3['experiment3_pass'] else '失败'}。",
        *calibration_lines,
        *experiment3_lines,
        "Latency microbenchmark 使用 decoded in-memory RGB、batch=1、3次完整 warmup、5次完整 timed passes，并从全部逐帧样本报告 median/P90。磁盘读取、PNG 解码和模型加载均排除。JEPA 刷新帧同时计入384 scout与512 reference，当前实现没有跨分辨率特征复用。",
        "",
        "## Feasibility verdict",
        "",
        f"最终判定：**{verdict['label']}**。",
        *condition_lines,
        f"- `speed_pass`：{'通过' if verdict['speed_pass']['passed'] else '失败'}；原因：{verdict['speed_pass']['reason'] or '无'}。",
        "",
        "所有失败、负收益和 unavailable 结果均按确定性规则保留，不以主观描述覆盖。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_run_manifest(
    *,
    output_dir: Path,
    prepared: PreparedTumSequence,
    model_metadata: dict[str, Any],
    environment: dict[str, Any],
    stage_runtime: dict[str, float],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = artifact_records(output_dir)
    return {
        "schema_version": "1.0",
        "phase_name": PHASE4_NAME,
        "timestamp": now_utc(),
        **collect_git_provenance(),
        **collect_source_provenance(),
        "dataset": {
            "name": DATASET_NAME,
            "sequence_name": SEQUENCE_NAME,
            "dataset_root": prepared.manifest["dataset_root"],
            "prepared_manifest_path": prepared.manifest["manifest_path"],
            "prepared_manifest_sha256": sha256_file(prepared.manifest_path),
            "prepared_structure_fingerprint": prepared.manifest["structure_fingerprint"],
            "frame_count": len(prepared.frames),
        },
        "model": model_metadata,
        "preprocessing": {
            "source_crop_policy": "center_square_from_original",
            "crop_bounds": "side=min(width,height); left/top=floor(remaining/2); xyxy right/bottom exclusive",
            "scout_resolution": SCOUT_RESOLUTION,
            "reference_resolution": REFERENCE_RESOLUTION,
            "pixel_flow_resolution": SCOUT_RESOLUTION,
        },
        "configuration": {
            "time_horizons_seconds": list(TIME_HORIZONS_SECONDS),
            "time_pair_match_tolerance_seconds": TIME_PAIR_MATCH_TOLERANCE_SECONDS,
            "max_pairs_per_horizon": MAX_PAIRS_PER_HORIZON,
            "observation_budgets": list(OBSERVATION_BUDGETS),
            "target_count_rounding_policy": "half-up: max(1, floor(budget*N + 0.5))",
            "calibration_fraction": CALIBRATION_FRACTION,
            "adaptive_target_observation_rate": ADAPTIVE_TARGET_OBSERVATION_RATE,
            "stable_timeout_seconds": STABLE_TIMEOUT_SECONDS,
            "normal_timeout_seconds": NORMAL_TIMEOUT_SECONDS,
            "adaptive_low_quantile_candidates": list(ADAPTIVE_LOW_QUANTILE_CANDIDATES),
            "adaptive_high_quantile_candidates": list(ADAPTIVE_HIGH_QUANTILE_CANDIDATES),
            "timing_warmups": TIMING_WARMUPS,
            "timing_repeats": TIMING_REPEATS,
            "timing_max_frames": TIMING_MAX_FRAMES,
        },
        "scientific_protocol": {
            "experiment_1_states": {"scout": "ViT-B@384 mean-pooled L2-normalized global latent", "reference": "ViT-B@512 mean-pooled L2-normalized global latent", "consistency_metrics": ["distance Spearman", "mean absolute distance difference", "per-horizon distance difference"]},
            "experiment_2_score": "offline adjacent-frame score ranking at exact fixed budgets",
            "experiment_2_auc": {"error": "normalized trapezoidal AUC of mean state error; lower is better", "similarity": "normalized trapezoidal AUC of mean state similarity; higher is better and auxiliary"},
            "high_error_episode_support": "[timestamp_t,timestamp_t+1); segment-final frame uses segment median adjacent delta; one-frame segment is degenerate",
            "experiment_3_score": "causal change(last_refresh_anchor,current_frame)",
            "matched_uniform_rule": "Uniform count exactly equals JEPA adaptive evaluation observation count",
        },
        "result_protocol_values": None if summary is None else {
            "uniform_target_count": summary.get("experiment_3", {}).get("policies", {}).get("uniform_target_10", {}).get("observation_count"),
            "uniform_matched_jepa_count": summary.get("experiment_3", {}).get("policies", {}).get("uniform_matched_jepa", {}).get("observation_count"),
            "calibration_unique_score_counts": {method: item["unique_score_count"] for method, item in summary.get("experiment_3", {}).get("calibrations", {}).items()},
            "unique_threshold_pair_counts": {method: item["unique_threshold_pair_count"] for method, item in summary.get("experiment_3", {}).get("calibrations", {}).items()},
        },
        "environment": environment,
        "runtime_seconds": stage_runtime,
        "manifest_self_excluded": True,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "output_file_list": [artifact["relative_path"] for artifact in artifacts],
    }


def run_phase4() -> dict[str, Any]:
    import torch

    from research.scripts.common.video_models import load_vitb_encoder, release_cuda

    phase_start = time.perf_counter()
    runtimes: dict[str, float] = {}
    print("[Phase 4 / Feasibility] TUM RGB-D first-person state refresh")
    prepare_start = time.perf_counter()
    prepared = prepare_tum_sequence()
    runtimes["prepare"] = time.perf_counter() - prepare_start
    print_prepare_summary(prepared)
    device, gpu_name = require_cuda()
    environment = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "pytorch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": "cuda:0",
        "gpu_name": gpu_name,
    }
    print(f"[1/6] Loading fixed V-JEPA 2.1 ViT-B on {gpu_name}")
    encoder = load_vitb_encoder(device)
    stage: Path | None = None
    try:
        extraction_start = time.perf_counter()
        print("[2/6] Extracting shared 384 scout and 512 reference global states")
        features = extract_global_states(encoder=encoder, device=device, frames=prepared.frames)
        runtimes["feature_extraction"] = time.perf_counter() - extraction_start
        timestamps, positions, quaternions = pose_arrays(prepared.frames)
        split = split_time_segments(timestamps)
        calibration_indices = split["calibration_indices"]
        calibration_times = timestamps[calibration_indices]
        timing_thresholds: dict[str, dict[str, float]] = {}
        for method in ("pixel", "jepa"):
            calibration_gray = features["gray384"][calibration_indices]
            calibration_scout = features["z384"][calibration_indices]
            calibration = calibrate_causal_thresholds(
                calibration_times,
                offline_adjacent_scores(method, calibration_gray, calibration_scout),
                causal_score_function(method, calibration_gray, calibration_scout),
            )
            if calibration["status"] != "available":
                raise RuntimeError(f"{method} calibration unavailable before timing: {calibration['reason']}")
            timing_thresholds[method] = {"low": calibration["selected"]["low_threshold"], "high": calibration["selected"]["high_threshold"]}
        timing_start = time.perf_counter()
        print("[3/6] Measuring canonical batch=1 component latencies")
        timing = run_latency_microbenchmark(
            encoder=encoder,
            device=device,
            frames=prepared.frames,
            scout_latents=features["z384"],
            pixel_thresholds=timing_thresholds["pixel"],
            jepa_thresholds=timing_thresholds["jepa"],
        )
        runtimes["latency_microbenchmark"] = time.perf_counter() - timing_start
        del encoder
        release_cuda(device=device)

        stage = prepare_staged_output()
        experiment_start = time.perf_counter()
        print("[4/6] Running ego-motion consistency and sparse refresh analyses")
        ego = run_ego_motion_consistency(
            frames=prepared.frames,
            gray_frames=features["gray384"],
            scout_latents=features["z384"],
            reference_latents=features["z512"],
            output_dir=stage / "ego_motion_consistency",
        )
        runtimes["experiment_1"] = time.perf_counter() - experiment_start
        experiment_start = time.perf_counter()
        sparse = run_sparse_state_refresh(
            timestamps=timestamps,
            gray_frames=features["gray384"],
            scout_latents=features["z384"],
            reference_latents=features["z512"],
            positions=positions,
            quaternions=quaternions,
            output_dir=stage / "sparse_state_refresh",
        )
        runtimes["experiment_2"] = time.perf_counter() - experiment_start
        experiment_start = time.perf_counter()
        adaptive = run_adaptive_observation(
            timestamps=timestamps,
            gray_frames=features["gray384"],
            scout_latents=features["z384"],
            reference_latents=features["z512"],
            positions=positions,
            quaternions=quaternions,
            timing=timing,
            output_dir=stage / "adaptive_observation",
        )
        runtimes["experiment_3"] = time.perf_counter() - experiment_start
        verdict = build_feasibility_verdict(sparse, adaptive)
        summary = build_summary(prepared=prepared, ego=ego, sparse=sparse, adaptive=adaptive, verdict=verdict)
        write_json(stage / "summary.json", summary)
        write_report(stage / "report.md", summary=summary, model_metadata=features["model_metadata"])
        runtimes["total_before_manifest"] = time.perf_counter() - phase_start
        manifest = build_run_manifest(
            output_dir=stage,
            prepared=prepared,
            model_metadata=features["model_metadata"],
            environment=environment,
            stage_runtime=runtimes,
            summary=summary,
        )
        write_json(stage / "manifest.json", manifest)
        validate_manifest_paths(stage, manifest["artifacts"])
        validate_expected_output(stage)
        print("[5/6] Publishing exact staged artifact set")
        publish_staged_output(stage)
        stage = None
        print(f"[6/6] Done: {PHASE4_OUTPUT_DIR.relative_to(REPO_ROOT).as_posix()}/")
        return summary
    except Exception:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        raise
    finally:
        if "encoder" in locals():
            try:
                del encoder
            except UnboundLocalError:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
