"""DAVIS 2017 filesystem helpers for the Phase 2 dense baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE2_NAME = "phase2-dense"
DATASET_NAME = "DAVIS 2017 TrainVal 480p"
DEFAULT_SEQUENCES = ("dog", "car-shadow", "parkour")
FRAME_SAMPLE_COUNT = 8

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE2_ASSETS_ROOT = RESEARCH_ROOT / "assets" / PHASE2_NAME
DAVIS_ROOT = PHASE2_ASSETS_ROOT / "datasets" / "davis2017" / "DAVIS"
PREPARED_MANIFEST_PATH = PHASE2_ASSETS_ROOT / "prepared" / "davis2017" / "manifest.json"
PHASE2_OUTPUT_DIR = RESEARCH_ROOT / "outputs" / PHASE2_NAME
PHASE2_STAGED_OUTPUT_DIR = RESEARCH_ROOT / "outputs" / f".{PHASE2_NAME}.tmp"

DAVIS_DOWNLOAD_COMMAND = (
    "mkdir -p research/assets/phase2-dense/datasets/davis2017 && cd "
    "research/assets/phase2-dense/datasets/davis2017 && wget -c "
    "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip && "
    "{ [ -d DAVIS ] || unzip -q DAVIS-2017-trainval-480p.zip; } && "
    "test -d DAVIS/JPEGImages/480p && test -d DAVIS/Annotations/480p && "
    'echo "DAVIS 2017 ready: $(pwd)/DAVIS"'
)


@dataclass(frozen=True)
class DavisSequence:
    name: str
    image_dir: Path
    mask_dir: Path
    frame_paths: tuple[Path, ...]
    mask_paths: tuple[Path, ...]
    frame_count: int
    mask_count: int
    first_frame: str
    last_frame: str
    first_mask: str
    last_mask: str
    frame_bytes: int
    mask_bytes: int
    latest_mtime_ns: int

    def frame_path_for_index(self, frame_index: int) -> Path:
        path = self.image_dir / f"{int(frame_index):05d}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"Missing DAVIS frame: {repo_relative(path)}")
        return path

    def mask_path_for_index(self, frame_index: int) -> Path:
        path = self.mask_dir / f"{int(frame_index):05d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing DAVIS mask: {repo_relative(path)}")
        return path


@dataclass(frozen=True)
class PreparedDavis:
    davis_root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    sequences: dict[str, DavisSequence]
    skipped: bool


def repo_relative(path: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_relative(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _missing_davis_error(path: Path) -> FileNotFoundError:
    return FileNotFoundError(
        "DAVIS 2017 TrainVal 480p is missing or incomplete at "
        f"{repo_relative(path)}.\n"
        "Prepare it with this exact one-line command:\n"
        f"{DAVIS_DOWNLOAD_COMMAND}"
    )


def _require_dir(path: Path) -> None:
    if not path.is_dir():
        raise _missing_davis_error(path)


def _file_stats(paths: tuple[Path, ...]) -> tuple[int, int]:
    total_bytes = 0
    latest_mtime_ns = 0
    for path in paths:
        stat = path.stat()
        total_bytes += int(stat.st_size)
        latest_mtime_ns = max(latest_mtime_ns, int(stat.st_mtime_ns))
    return total_bytes, latest_mtime_ns


def _scan_sequence(davis_root: Path, sequence: str) -> DavisSequence:
    image_dir = davis_root / "JPEGImages" / "480p" / sequence
    mask_dir = davis_root / "Annotations" / "480p" / sequence
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            "Required DAVIS sequence is missing from JPEGImages/480p or Annotations/480p: "
            f"{sequence}\n"
            "Prepare DAVIS with:\n"
            f"{DAVIS_DOWNLOAD_COMMAND}"
        )

    frame_paths = tuple(sorted(image_dir.glob("*.jpg")))
    mask_paths = tuple(sorted(mask_dir.glob("*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"No DAVIS frames found for sequence: {sequence}")
    if not mask_paths:
        raise FileNotFoundError(f"No DAVIS masks found for sequence: {sequence}")

    first_frame = frame_paths[0].name
    last_frame = frame_paths[-1].name
    first_mask = f"{Path(first_frame).stem}.png"
    last_mask = f"{Path(last_frame).stem}.png"
    if not (mask_dir / first_mask).is_file() or not (mask_dir / last_mask).is_file():
        raise FileNotFoundError(
            f"DAVIS sequence {sequence!r} is missing first/last masks: "
            f"{first_mask}, {last_mask}"
        )
    if len(frame_paths) != len(mask_paths):
        raise RuntimeError(
            f"DAVIS sequence {sequence!r} has mismatched frame/mask counts: "
            f"{len(frame_paths)} frames, {len(mask_paths)} masks."
        )

    frame_bytes, frame_mtime = _file_stats(frame_paths)
    mask_bytes, mask_mtime = _file_stats(mask_paths)
    return DavisSequence(
        name=sequence,
        image_dir=image_dir,
        mask_dir=mask_dir,
        frame_paths=frame_paths,
        mask_paths=mask_paths,
        frame_count=len(frame_paths),
        mask_count=len(mask_paths),
        first_frame=first_frame,
        last_frame=last_frame,
        first_mask=first_mask,
        last_mask=last_mask,
        frame_bytes=frame_bytes,
        mask_bytes=mask_bytes,
        latest_mtime_ns=max(frame_mtime, mask_mtime),
    )


def inspect_davis_dataset(
    davis_root: Path = DAVIS_ROOT,
    sequences: tuple[str, ...] = DEFAULT_SEQUENCES,
) -> dict[str, DavisSequence]:
    _require_dir(davis_root)
    _require_dir(davis_root / "JPEGImages" / "480p")
    _require_dir(davis_root / "Annotations" / "480p")
    _require_dir(davis_root / "ImageSets")
    return {sequence: _scan_sequence(davis_root, sequence) for sequence in sequences}


def sequence_manifest_record(sequence: DavisSequence) -> dict[str, Any]:
    return {
        "name": sequence.name,
        "image_dir": repo_relative(sequence.image_dir),
        "mask_dir": repo_relative(sequence.mask_dir),
        "frame_count": sequence.frame_count,
        "mask_count": sequence.mask_count,
        "first_frame": sequence.first_frame,
        "last_frame": sequence.last_frame,
        "first_mask": sequence.first_mask,
        "last_mask": sequence.last_mask,
        "frame_bytes": sequence.frame_bytes,
        "mask_bytes": sequence.mask_bytes,
        "latest_mtime_ns": sequence.latest_mtime_ns,
    }


def _structure_payload(
    *,
    davis_root: Path,
    sequences: dict[str, DavisSequence],
) -> dict[str, Any]:
    return {
        "dataset": DATASET_NAME,
        "davis_root": repo_relative(davis_root),
        "sequences": [sequence_manifest_record(sequences[name]) for name in sorted(sequences)],
    }


def structure_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_prepared_manifest(
    *,
    davis_root: Path,
    sequences: dict[str, DavisSequence],
) -> dict[str, Any]:
    structure = _structure_payload(davis_root=davis_root, sequences=sequences)
    return {
        "phase": PHASE2_NAME,
        "prepared_at": now_utc(),
        "dataset": DATASET_NAME,
        "davis_root": repo_relative(davis_root),
        "jpeg_root": repo_relative(davis_root / "JPEGImages" / "480p"),
        "annotation_root": repo_relative(davis_root / "Annotations" / "480p"),
        "imagesets_root": repo_relative(davis_root / "ImageSets"),
        "selected_sequences": list(sequences),
        "structure": structure,
        "structure_fingerprint": structure_fingerprint(structure),
        "preparation_note": "Paths only; DAVIS frames and masks are not copied.",
    }


def prepare_davis_dataset(
    *,
    davis_root: Path = DAVIS_ROOT,
    manifest_path: Path = PREPARED_MANIFEST_PATH,
    sequences: tuple[str, ...] = DEFAULT_SEQUENCES,
    force: bool = False,
) -> PreparedDavis:
    sequence_records = inspect_davis_dataset(davis_root=davis_root, sequences=sequences)
    current_manifest = build_prepared_manifest(davis_root=davis_root, sequences=sequence_records)
    skipped = False

    if manifest_path.is_file() and not force:
        existing = _read_json(manifest_path)
        same_fingerprint = existing.get("structure_fingerprint") == current_manifest["structure_fingerprint"]
        same_sequences = existing.get("selected_sequences") == list(sequences)
        same_root = existing.get("davis_root") == repo_relative(davis_root)
        if same_fingerprint and same_sequences and same_root:
            skipped = True
            current_manifest = existing

    if not skipped:
        _write_json(manifest_path, current_manifest)

    return PreparedDavis(
        davis_root=davis_root,
        manifest_path=manifest_path,
        manifest=current_manifest,
        sequences=sequence_records,
        skipped=skipped,
    )


def uniform_sample_indices(frame_count: int, count: int = FRAME_SAMPLE_COUNT) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if count <= 0:
        raise ValueError("sample count must be positive.")
    if frame_count <= count:
        return list(range(frame_count))
    if count == 1:
        return [0]
    indices = [
        int(round(position * (frame_count - 1) / (count - 1)))
        for position in range(count)
    ]
    deduped: list[int] = []
    for index in indices:
        index = min(max(index, 0), frame_count - 1)
        if not deduped or deduped[-1] != index:
            deduped.append(index)
    return deduped
