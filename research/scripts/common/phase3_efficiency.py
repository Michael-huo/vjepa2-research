"""Phase 3 efficiency and compute-demand evaluation for V-JEPA 2.1."""

from __future__ import annotations

import csv
import gc
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from research.scripts.common.dense_pca import (
    FEATURE_MODE,
    FrameFeatureBatch,
    adjacent_patch_cosine,
    adjacent_pca_drift,
    extract_frame_features,
    fit_pca_rgb,
    infer_feature_grid,
    pca_metric_payload,
    project_pca_rgb,
    resize_center_crop_pil,
)
from research.scripts.common.phase2_data import (
    DEFAULT_SEQUENCES,
    FRAME_SAMPLE_COUNT,
    PREPARED_MANIFEST_PATH,
    PreparedDavis,
    prepare_davis_dataset,
    repo_relative,
    uniform_sample_indices,
)
from research.scripts.common.runtime import (
    REPO_ROOT,
    collect_git_provenance,
    now_utc,
    runtime_environment,
    sha256_bytes,
    sha256_file,
    write_json,
)
from research.scripts.common.video_models import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    autocast_context,
    release_cuda,
    synchronize,
)
from research.scripts.common.vos import aggregate_vos_metrics, run_vos_for_sequence
from research.scripts.common.visualization import (
    save_phase2_video_pca_summary,
    save_phase2_vos_summary,
)


PHASE3_NAME = "phase3-efficiency"
RESEARCH_ROOT = Path(__file__).resolve().parents[2]
PHASE3_OUTPUT_DIR = RESEARCH_ROOT / "outputs" / PHASE3_NAME
PHASE3_STAGED_OUTPUT_DIR = RESEARCH_ROOT / "outputs" / f".{PHASE3_NAME}.tmp"

QUICK_MODELS = ("vit_b",)
FULL_MODELS = ("vit_b", "vit_l", "vit_g", "vit_G")
RECOMMENDATION_MODEL_ORDER = ("vit_b", "vit_l", "vit_g", "vit_G")
QUICK_RESOLUTIONS = (384, 512)
FULL_RESOLUTIONS = (384, 512, 768, 1024)
QUICK_BATCH_SIZES = (1, 4)
FULL_BATCH_SIZES = (1, 2, 4, 8, 16)
QUICK_QUALITY_CONFIGS = (("vit_b", 384), ("vit_b", 512))
FULL_QUALITY_CONFIGS = (
    ("vit_b", 384),
    ("vit_b", 512),
    ("vit_b", 768),
    ("vit_b", 1024),
    ("vit_l", 512),
    ("vit_l", 768),
    ("vit_g", 512),
    ("vit_G", 512),
)
TIMING_STATS = ("median", "mean", "min", "max", "std")
FIGURE_NAMES = (
    "latency_vs_resolution.png",
    "memory_vs_resolution.png",
    "throughput_vs_batch.png",
    "quality_vs_latency.png",
    "quality_vs_memory.png",
    "pareto_front.png",
)
PHASE3_SOURCE_RELATIVE_PATHS = (
    "research/scripts/run_phase3_efficiency.py",
    "research/scripts/common/phase3_efficiency.py",
    "research/scripts/common/dense_pca.py",
    "research/scripts/common/vos.py",
    "research/scripts/common/visualization.py",
    "research/scripts/common/runtime.py",
    "research/phase3_efficiency_README.md",
    "tests/test_phase3_efficiency.py",
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    artifact_slug: str
    official_alias: str


@dataclass(frozen=True)
class Phase3Config:
    mode: str
    models: tuple[str, ...]
    resolutions: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    quality_configs: tuple[tuple[str, int], ...]
    quality_enabled: bool
    warmups: int
    repeats: int


@dataclass(frozen=True)
class BenchmarkFrame:
    sequence: str
    frame_index: int
    path: Path
    rgb: np.ndarray


@dataclass(frozen=True)
class RecommendationRecord:
    model: str
    display_name: str
    resolution: int
    config_id: str
    sampled_crop_j_mean: float
    pipeline_ms_per_frame_median: float
    pipeline_fps: float


class ModelUnavailableError(RuntimeError):
    pass


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "vit_b": ModelSpec("vit_b", "ViT-B", "vit_b", "vjepa2_1_vit_base_384"),
    "vit_l": ModelSpec("vit_l", "ViT-L", "vit_l", "vjepa2_1_vit_large_384"),
    "vit_g": ModelSpec("vit_g", "ViT-g", "vit_g_lc", "vjepa2_1_vit_giant_384"),
    "vit_G": ModelSpec("vit_G", "ViT-G", "vit_G_uc", "vjepa2_1_vit_gigantic_384"),
}
MODEL_ORDER = {
    model_id: index for index, model_id in enumerate(RECOMMENDATION_MODEL_ORDER)
}


def validate_model_registry(model_registry: dict[str, ModelSpec] = MODEL_REGISTRY) -> None:
    slugs = [entry.artifact_slug for entry in model_registry.values()]
    assert len({slug.casefold() for slug in slugs}) == len(slugs), (
        "Phase 3 artifact slugs must be unique on case-insensitive filesystems."
    )
    for model_id, entry in model_registry.items():
        assert model_id == entry.model_id, f"Registry key mismatch for {model_id}."


validate_model_registry()


def make_config_slug(model_id: str, resolution: int) -> str:
    return f"{MODEL_REGISTRY[model_id].artifact_slug}_{int(resolution)}"


def _timing_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_ms_per_{unit}_{stat}"
        for unit in ("batch", "frame")
        for stat in TIMING_STATS
    ]


IDENTITY_COLUMNS = [
    "mode",
    "model",
    "display_name",
    "artifact_slug",
    "official_model_alias",
    "active_dtype",
    "resolution",
    "batch_size",
    "status",
    "grid_shape",
    "tokens_per_frame",
]
MEMORY_COLUMNS = [
    "cuda_preload_allocated_mb",
    "cuda_preload_reserved_mb",
    "model_resident_allocated_mb",
    "model_resident_reserved_mb",
    "peak_allocated_memory_mb",
    "peak_reserved_memory_mb",
    "incremental_inference_allocated_mb",
    "incremental_inference_reserved_mb",
    "total_gpu_memory_mb",
    "memory_utilization_proxy",
]
FEASIBILITY_COLUMNS = [
    *IDENTITY_COLUMNS,
    "model_load_seconds",
    "preprocess_ms_per_frame_median",
    "encoder_core_ms_per_frame_median",
    "encoder_core_fps",
    "pipeline_ms_per_frame_median",
    "pipeline_fps",
    *MEMORY_COLUMNS,
    "total_seconds",
    "error",
]
BATCH_COLUMNS = [
    *IDENTITY_COLUMNS,
    *_timing_columns("preprocess"),
    *_timing_columns("encoder_core"),
    "encoder_core_throughput_fps",
    *_timing_columns("pipeline"),
    "pipeline_throughput_fps",
    *MEMORY_COLUMNS,
    "total_seconds",
    "error",
]
QUALITY_COLUMNS = [
    "mode",
    "model",
    "display_name",
    "artifact_slug",
    "official_model_alias",
    "active_dtype",
    "resolution",
    "status",
    "grid_shape",
    "tokens_per_frame",
    "mean_adjacent_cosine",
    "mean_pca_color_drift",
    "sampled_crop_j_mean",
    "sampled_crop_j_min",
    "low_iou_frames",
    "total_quality_seconds",
    "preprocess_ms_per_frame_median",
    "encoder_core_ms_per_frame_median",
    "encoder_core_fps",
    "pipeline_ms_per_frame_median",
    "pipeline_fps",
    "model_resident_allocated_mb",
    "peak_allocated_memory_mb",
    "peak_reserved_memory_mb",
    "error",
]


def build_config(*, full: bool, no_quality: bool) -> Phase3Config:
    if full:
        return Phase3Config(
            mode="full",
            models=FULL_MODELS,
            resolutions=FULL_RESOLUTIONS,
            batch_sizes=FULL_BATCH_SIZES,
            quality_configs=FULL_QUALITY_CONFIGS,
            quality_enabled=not no_quality,
            warmups=5,
            repeats=10,
        )
    return Phase3Config(
        mode="quick",
        models=QUICK_MODELS,
        resolutions=QUICK_RESOLUTIONS,
        batch_sizes=QUICK_BATCH_SIZES,
        quality_configs=QUICK_QUALITY_CONFIGS,
        quality_enabled=not no_quality,
        warmups=3,
        repeats=5,
    )


def benchmark_protocol(config: Phase3Config) -> dict[str, Any]:
    return {
        "quick": {"warmup_runs": 3, "timed_repeats": 5},
        "full": {"warmup_runs": 5, "timed_repeats": 10},
        "active_mode": config.mode,
        "warmup_runs": int(config.warmups),
        "timed_repeats": int(config.repeats),
        "timing_source": "decoded_rgb_arrays_in_ram",
        "disk_io_included": False,
        "jpeg_decode_included": False,
        "model_load_included_in_latency": False,
        "pipeline_measured_directly": True,
        "authoritative_latency_source": "canonical_batch_sweep_batch1_pipeline",
        "active_dtype_detection": (
            "Encoder output dtype from the first successful canonical benchmark; "
            "floating model parameter/buffer dtype is the load-time fallback."
        ),
        "population_std_ddof": 0,
    }


def run_from_args(args: Any) -> None:
    config = build_config(full=bool(args.full), no_quality=bool(args.no_quality))
    phase_start = time.perf_counter()
    print("[Phase 3 / Efficiency] V-JEPA 2.1 dense feature compute-demand evaluation")
    print("[0/N] Preparing DAVIS dataset")
    prepare_start = time.perf_counter()
    prepared = prepare_davis_dataset(
        manifest_path=PREPARED_MANIFEST_PATH,
        sequences=DEFAULT_SEQUENCES,
        force=bool(args.force_prepare),
    )
    prepare_seconds = time.perf_counter() - prepare_start
    status = "reused" if prepared.skipped else "wrote"
    print(f"      prepared manifest {status}: {repo_relative(prepared.manifest_path)}")
    for sequence in DEFAULT_SEQUENCES:
        record = prepared.sequences[sequence]
        print(f"      {sequence}: {record.frame_count} frames, {record.mask_count} masks")
    print(f"      elapsed: {_seconds(prepare_seconds)}")

    if args.prepare_only:
        print("[Done] Prepare-only check complete.")
        return

    device, gpu_name = require_phase3_cuda()
    environment = runtime_environment("cuda:0", gpu_name)
    environment["total_gpu_memory_mb"] = _total_gpu_memory_mb(device)
    print(f"[Setup] Device: cuda:0 | GPU: {gpu_name} | CUDA: {torch.version.cuda}")

    print("[Setup] Decoding deterministic DAVIS benchmark frame pool")
    frame_pool = build_benchmark_frame_pool(
        prepared,
        pool_size=max(config.batch_sizes),
    )
    frame_pool_payload = benchmark_frame_pool_payload(
        frame_pool,
        largest_requested_batch=max(config.batch_sizes),
    )
    staged_dir = prepare_staged_output()
    try:
        run_phase3(
            prepared=prepared,
            config=config,
            frame_pool=frame_pool,
            frame_pool_payload=frame_pool_payload,
            output_dir=staged_dir,
            device=device,
            environment=environment,
            prepare_seconds=prepare_seconds,
            phase_start=phase_start,
        )
        verify_phase3_output(staged_dir)
        replace_final_output(staged_dir)
        print("[Done] Phase 3 outputs:")
        print(f"       {repo_relative(PHASE3_OUTPUT_DIR)}/")
    except Exception:
        cleanup_staged_output(staged_dir)
        raise
    finally:
        cleanup_cuda_state(device)


def run_phase3(
    *,
    prepared: PreparedDavis,
    config: Phase3Config,
    frame_pool: list[BenchmarkFrame],
    frame_pool_payload: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    environment: dict[str, Any],
    prepare_seconds: float,
    phase_start: float,
) -> dict[str, Any]:
    print()
    print("[1/N] Canonical model-centric benchmarks and quality evaluation")
    benchmark_start = time.perf_counter()
    canonical_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    model_load_results: list[dict[str, Any]] = []
    selected_outputs: dict[str, Any] = {}
    sweep_configs = set(batch_sweep_configs(config))

    for model_id in config.models:
        spec = MODEL_REGISTRY[model_id]
        print(f"      {spec.display_name} ({model_id})")
        cleanup_cuda_state(device)
        preload_allocated = _allocated_memory_mb(device)
        preload_reserved = _reserved_memory_mb(device)
        load_start = time.perf_counter()
        encoder = None
        try:
            encoder = load_phase3_encoder(spec, device=device)
            model_load_seconds = time.perf_counter() - load_start
            encoder.eval()
            cleanup_cuda_cache(device)
            resident_allocated = _allocated_memory_mb(device)
            resident_reserved = _reserved_memory_mb(device)
            detected_dtype = detect_model_dtype(encoder)
            load_result = model_load_record(
                spec=spec,
                status="success",
                error="",
                model_load_seconds=model_load_seconds,
                active_dtype=detected_dtype,
                preload_allocated=preload_allocated,
                preload_reserved=preload_reserved,
                resident_allocated=resident_allocated,
                resident_reserved=resident_reserved,
            )
        except Exception as error:
            model_load_seconds = time.perf_counter() - load_start
            status = status_from_exception(error)
            message = compact_error(error)
            handle_cuda_failure(error, device)
            load_result = model_load_record(
                spec=spec,
                status=status,
                error=message,
                model_load_seconds=model_load_seconds,
                active_dtype="",
                preload_allocated=preload_allocated,
                preload_reserved=preload_reserved,
                resident_allocated=0.0,
                resident_reserved=0.0,
            )
            append_model_load_failures(
                canonical_rows=canonical_rows,
                quality_rows=quality_rows,
                config=config,
                spec=spec,
                load_result=load_result,
                device=device,
            )
            model_load_results.append(load_result)
            continue

        batch1_by_resolution: dict[int, dict[str, Any]] = {}
        for resolution in config.resolutions:
            row = benchmark_base_record(
                config=config,
                spec=spec,
                resolution=resolution,
                batch_size=1,
                load_result=load_result,
                device=device,
            )
            try:
                result = canonical_benchmark(
                    encoder=encoder,
                    spec=spec,
                    resolution=resolution,
                    batch_size=1,
                    frame_pool=frame_pool,
                    device=device,
                    warmups=config.warmups,
                    repeats=config.repeats,
                )
                row.update(result)
                if result.get("active_dtype"):
                    load_result["active_dtype"] = result["active_dtype"]
            except Exception as error:
                row["status"] = status_from_exception(error)
                row["error"] = compact_error(error)
                handle_cuda_failure(error, device)
            canonical_rows.append(row)
            batch1_by_resolution[resolution] = row

        for resolution in config.resolutions:
            if (model_id, resolution) not in sweep_configs:
                continue
            batch1 = batch1_by_resolution[resolution]
            stop_after_oom = False
            for batch_size in (value for value in config.batch_sizes if value > 1):
                row = benchmark_base_record(
                    config=config,
                    spec=spec,
                    resolution=resolution,
                    batch_size=batch_size,
                    load_result=load_result,
                    device=device,
                )
                if batch1["status"] != "success":
                    row["status"] = "skipped"
                    row["error"] = (
                        f"Skipped because canonical batch=1 status was {batch1['status']}."
                    )
                elif stop_after_oom:
                    row["status"] = "skipped"
                    row["error"] = "Skipped after OOM for this model/resolution."
                else:
                    try:
                        row.update(
                            canonical_benchmark(
                                encoder=encoder,
                                spec=spec,
                                resolution=resolution,
                                batch_size=batch_size,
                                frame_pool=frame_pool,
                                device=device,
                                warmups=config.warmups,
                                repeats=config.repeats,
                            )
                        )
                    except Exception as error:
                        row["status"] = status_from_exception(error)
                        row["error"] = compact_error(error)
                        handle_cuda_failure(error, device)
                        stop_after_oom = row["status"] == "oom"
                canonical_rows.append(row)

        if config.quality_enabled:
            for quality_model, resolution in config.quality_configs:
                if quality_model != model_id:
                    continue
                canonical = batch1_by_resolution[resolution]
                row = quality_base_record(config=config, spec=spec, resolution=resolution)
                copy_canonical_efficiency(row, canonical)
                if canonical["status"] != "success":
                    row["status"] = canonical["status"]
                    row["error"] = (
                        f"Skipped because canonical batch=1 status was {canonical['status']}."
                    )
                    quality_rows.append(row)
                    continue
                config_slug = make_config_slug(model_id, resolution)
                config_output_dir = output_dir / "selected_outputs" / config_slug
                try:
                    result = run_quality_config(
                        prepared=prepared,
                        encoder=encoder,
                        spec=spec,
                        device=device,
                        resolution=resolution,
                        output_dir=config_output_dir,
                    )
                    row.update(result["row_metrics"])
                    copy_canonical_efficiency(row, canonical)
                    selected_outputs[config_slug] = {
                        "model": model_id,
                        "display_name": spec.display_name,
                        "artifact_slug": spec.artifact_slug,
                        "resolution": int(resolution),
                        "path": f"selected_outputs/{config_slug}",
                    }
                except Exception as error:
                    row["status"] = status_from_exception(error)
                    row["error"] = compact_error(error)
                    if config_output_dir.exists():
                        shutil.rmtree(config_output_dir)
                    handle_cuda_failure(error, device)
                quality_rows.append(row)

        model_load_results.append(load_result)
        del encoder
        cleanup_cuda_state(device)

    benchmark_seconds = time.perf_counter() - benchmark_start
    feasibility_rows = project_feasibility_rows(canonical_rows)
    batch_rows = list(canonical_rows)
    if not config.quality_enabled:
        quality_rows = []
        selected_outputs = {}
    recommendations = select_quality_efficiency_recommendations(quality_rows)

    print()
    print("[2/N] Writing CSVs and figures")
    write_csv(output_dir / "feasibility_matrix.csv", FEASIBILITY_COLUMNS, feasibility_rows)
    write_csv(output_dir / "batch_sweep.csv", BATCH_COLUMNS, batch_rows)
    write_csv(output_dir / "quality_efficiency.csv", QUALITY_COLUMNS, quality_rows)
    save_phase3_figures(
        figures_dir=output_dir / "figures",
        feasibility_rows=feasibility_rows,
        batch_rows=batch_rows,
        quality_rows=quality_rows,
        quality_enabled=config.quality_enabled,
        recommendations=recommendations,
    )

    runtime_summary = {
        "prepare_seconds": float(prepare_seconds),
        "canonical_benchmark_and_quality_seconds": float(benchmark_seconds),
        "end_to_end_total_seconds_before_publication": float(time.perf_counter() - phase_start),
    }
    integrity_summary = {
        "status": "validated_before_publication",
        "casefold_file_and_directory_validation": True,
        "successful_quality_artifact_contract": [
            "metrics.json",
            "video_pca_summary.png",
            "vos_summary.png",
        ],
        "successful_quality_config_slugs": [
            make_config_slug(str(row["model"]), int(row["resolution"]))
            for row in quality_rows
            if row.get("status") == "success"
        ],
        "artifact_index_path": "artifact_index.json",
        "artifact_index_excludes": ["artifact_index.json", "manifest.json"],
        "manifest_is_self_excluded": True,
    }
    metrics = {
        "phase": PHASE3_NAME,
        "mode": config.mode,
        "dataset": "DAVIS 2017 TrainVal 480p",
        "feature_mode": FEATURE_MODE,
        "quality_enabled": bool(config.quality_enabled),
        "benchmark_protocol": benchmark_protocol(config),
        "benchmark_frame_pool": frame_pool_payload,
        "model_load_results": model_load_results,
        "canonical_batch_benchmarks": canonical_rows,
        "feasibility_summary": summary_payload(feasibility_rows),
        "batch_sweep_summary": summary_payload(batch_rows),
        "quality_efficiency_summary": {
            **summary_payload(quality_rows),
            "enabled": bool(config.quality_enabled),
            "selected_outputs": selected_outputs,
            "recommendations": recommendations,
        },
        "runtime_summary": runtime_summary,
        "artifact_integrity": integrity_summary,
        "environment": environment,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_report(output_dir / "report.md", metrics=metrics)

    print()
    print("[3/N] Validating and indexing staged artifacts")
    validate_successful_quality_artifacts(output_dir, quality_rows)
    validate_casefold_unique_paths(output_dir)
    artifact_entries = build_artifact_index_entries(output_dir)
    write_json(
        output_dir / "artifact_index.json",
        {"artifact_count": len(artifact_entries), "artifacts": artifact_entries},
    )
    validate_casefold_unique_paths(output_dir)
    output_files = output_file_list(output_dir, exclude={"manifest.json"})
    runtime_summary["end_to_end_total_seconds"] = float(time.perf_counter() - phase_start)
    write_manifest(
        output_dir / "manifest.json",
        config=config,
        prepared=prepared,
        environment=environment,
        frame_pool_payload=frame_pool_payload,
        model_load_results=model_load_results,
        runtime_summary=runtime_summary,
        output_files=output_files,
    )
    validate_manifest_output_paths(output_dir, output_files)
    validate_casefold_unique_paths(output_dir)
    return metrics


def build_benchmark_frame_pool(
    prepared: PreparedDavis,
    *,
    pool_size: int,
) -> list[BenchmarkFrame]:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive.")
    slot_sequences = [DEFAULT_SEQUENCES[index % len(DEFAULT_SEQUENCES)] for index in range(pool_size)]
    counts = {sequence: slot_sequences.count(sequence) for sequence in DEFAULT_SEQUENCES}
    indices_by_sequence = {
        sequence: iter(uniform_sample_indices(prepared.sequences[sequence].frame_count, count))
        for sequence, count in counts.items()
        if count > 0
    }
    frames: list[BenchmarkFrame] = []
    seen: set[Path] = set()
    for sequence in slot_sequences:
        frame_index = int(next(indices_by_sequence[sequence]))
        path = prepared.sequences[sequence].frame_path_for_index(frame_index)
        if path in seen:
            continue
        seen.add(path)
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        frames.append(BenchmarkFrame(sequence, frame_index, path, rgb))
    if not frames:
        raise RuntimeError("The deterministic DAVIS benchmark frame pool is empty.")
    return frames


def benchmark_frame_pool_payload(
    frames: list[BenchmarkFrame],
    *,
    largest_requested_batch: int,
) -> dict[str, Any]:
    return {
        "benchmark_sequences": list(DEFAULT_SEQUENCES),
        "frames": [
            {
                "sequence": frame.sequence,
                "frame_index": int(frame.frame_index),
                "relative_frame_path": repo_relative(frame.path),
                "original_height": int(frame.rgb.shape[0]),
                "original_width": int(frame.rgb.shape[1]),
            }
            for frame in frames
        ],
        "number_of_unique_decoded_frames": len({frame.path for frame in frames}),
        "largest_requested_batch": int(largest_requested_batch),
        "deterministic_sampling_rule": (
            "Round-robin sequence assignment across dog, car-shadow, and parkour; "
            "uniform deterministic indices within each sequence."
        ),
        "cycling_used": bool(largest_requested_batch > len(frames)),
        "disk_io_in_timing": False,
        "jpeg_decode_in_timing": False,
    }


def select_benchmark_frames(
    frame_pool: list[BenchmarkFrame],
    batch_size: int,
) -> list[BenchmarkFrame]:
    if not frame_pool:
        raise ValueError("frame_pool must not be empty.")
    return [frame_pool[index % len(frame_pool)] for index in range(batch_size)]


def preprocess_decoded_frame(frame: BenchmarkFrame, *, resolution: int) -> torch.Tensor:
    image = Image.fromarray(frame.rgb, mode="RGB")
    cropped, _ = resize_center_crop_pil(image, crop_size=resolution, interpolation="bilinear")
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    normalized = (array - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    return torch.from_numpy(normalized).permute(2, 0, 1).contiguous()


def preprocess_frame_batch(frames: list[BenchmarkFrame], *, resolution: int) -> torch.Tensor:
    return torch.stack(
        [preprocess_decoded_frame(frame, resolution=resolution) for frame in frames],
        dim=0,
    ).contiguous()


def canonical_benchmark(
    *,
    encoder: torch.nn.Module,
    spec: ModelSpec,
    resolution: int,
    batch_size: int,
    frame_pool: list[BenchmarkFrame],
    device: torch.device,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    del spec
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    trial_start = time.perf_counter()
    frames = select_benchmark_frames(frame_pool, batch_size)
    cleanup_cuda_cache(device)
    resident_allocated = _allocated_memory_mb(device)
    resident_reserved = _reserved_memory_mb(device)
    reset_cuda_peak(device)
    token_count = 0
    feature_dim = 0
    grid_shape: tuple[int, int] | None = None
    patch_size = 0
    output_dtype = ""

    with torch.inference_mode():
        for _ in range(max(0, warmups)):
            cpu_batch = preprocess_frame_batch(frames, resolution=resolution)
            gpu_batch = cpu_batch.to(device=device, non_blocking=True).unsqueeze(2)
            with autocast_context(device):
                tokens = encoder(gpu_batch)
            synchronize(device)
            del cpu_batch, gpu_batch, tokens

        preprocess_samples: list[float] = []
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            cpu_batch = preprocess_frame_batch(frames, resolution=resolution)
            preprocess_samples.append((time.perf_counter() - start) * 1000.0)
            del cpu_batch

        core_cpu_batch = preprocess_frame_batch(frames, resolution=resolution)
        processed_resolution = tuple(int(value) for value in core_cpu_batch.shape[-2:])
        encoder_samples: list[float] = []
        for _ in range(max(1, repeats)):
            synchronize(device)
            start = time.perf_counter()
            gpu_batch = core_cpu_batch.to(device=device, non_blocking=True).unsqueeze(2)
            with autocast_context(device):
                tokens = encoder(gpu_batch)
            synchronize(device)
            encoder_samples.append((time.perf_counter() - start) * 1000.0)
            token_count, feature_dim, grid_shape, patch_size, output_dtype = inspect_tokens(
                tokens=tokens,
                processed_resolution=processed_resolution,
                encoder=encoder,
            )
            del gpu_batch, tokens
        del core_cpu_batch

        pipeline_samples: list[float] = []
        for _ in range(max(1, repeats)):
            synchronize(device)
            start = time.perf_counter()
            cpu_batch = preprocess_frame_batch(frames, resolution=resolution)
            gpu_batch = cpu_batch.to(device=device, non_blocking=True).unsqueeze(2)
            with autocast_context(device):
                tokens = encoder(gpu_batch)
            synchronize(device)
            pipeline_samples.append((time.perf_counter() - start) * 1000.0)
            token_count, feature_dim, grid_shape, patch_size, output_dtype = inspect_tokens(
                tokens=tokens,
                processed_resolution=tuple(int(value) for value in cpu_batch.shape[-2:]),
                encoder=encoder,
            )
            del cpu_batch, gpu_batch, tokens

    synchronize(device)
    peak_allocated = _peak_allocated_memory_mb(device)
    peak_reserved = _peak_reserved_memory_mb(device)
    total_gpu_memory = _total_gpu_memory_mb(device)
    result: dict[str, Any] = {
        "status": "success",
        "active_dtype": output_dtype or detect_model_dtype(encoder),
        "grid_shape": format_grid(grid_shape),
        "tokens_per_frame": int(token_count),
        "feature_dim": int(feature_dim),
        "patch_size": int(patch_size),
        "model_resident_allocated_mb": resident_allocated,
        "model_resident_reserved_mb": resident_reserved,
        "peak_allocated_memory_mb": peak_allocated,
        "peak_reserved_memory_mb": peak_reserved,
        "incremental_inference_allocated_mb": max(0.0, peak_allocated - resident_allocated),
        "incremental_inference_reserved_mb": max(0.0, peak_reserved - resident_reserved),
        "total_gpu_memory_mb": total_gpu_memory,
        "memory_utilization_proxy": safe_divide(peak_allocated, total_gpu_memory),
        "total_seconds": float(time.perf_counter() - trial_start),
        "error": "",
    }
    result.update(timing_statistics("preprocess", preprocess_samples, batch_size))
    result.update(timing_statistics("encoder_core", encoder_samples, batch_size))
    result.update(timing_statistics("pipeline", pipeline_samples, batch_size))
    result["encoder_core_throughput_fps"] = throughput_from_median(
        batch_size, result["encoder_core_ms_per_batch_median"]
    )
    result["pipeline_throughput_fps"] = throughput_from_median(
        batch_size, result["pipeline_ms_per_batch_median"]
    )
    result["encoder_core_fps"] = result["encoder_core_throughput_fps"]
    result["pipeline_fps"] = result["pipeline_throughput_fps"]
    return result


def inspect_tokens(
    *,
    tokens: torch.Tensor,
    processed_resolution: tuple[int, int],
    encoder: torch.nn.Module,
) -> tuple[int, int, tuple[int, int], int, str]:
    if tokens.ndim != 3:
        raise RuntimeError(f"Unexpected encoder output shape: {tuple(tokens.shape)}")
    token_count = int(tokens.shape[1])
    feature_dim = int(tokens.shape[2])
    grid_h, grid_w, patch_size = infer_feature_grid(
        token_count=token_count,
        processed_resolution=processed_resolution,
        encoder=encoder,
    )
    return token_count, feature_dim, (grid_h, grid_w), patch_size, str(tokens.dtype)


def timing_statistics(prefix: str, samples_ms: list[float], batch_size: int) -> dict[str, float]:
    batch_values = np.asarray(samples_ms, dtype=np.float64)
    frame_values = batch_values / float(batch_size)
    result: dict[str, float] = {}
    for unit, values in (("batch", batch_values), ("frame", frame_values)):
        result[f"{prefix}_ms_per_{unit}_median"] = float(np.median(values))
        result[f"{prefix}_ms_per_{unit}_mean"] = float(np.mean(values))
        result[f"{prefix}_ms_per_{unit}_min"] = float(np.min(values))
        result[f"{prefix}_ms_per_{unit}_max"] = float(np.max(values))
        result[f"{prefix}_ms_per_{unit}_std"] = float(np.std(values, ddof=0))
    return result


def throughput_from_median(batch_size: int, median_ms_per_batch: float) -> float:
    return safe_divide(float(batch_size) * 1000.0, float(median_ms_per_batch))


def run_quality_config(
    *,
    prepared: PreparedDavis,
    encoder: torch.nn.Module,
    spec: ModelSpec,
    device: torch.device,
    resolution: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    encoder.eval()
    stage_start = time.perf_counter()
    visual_records: list[dict[str, Any]] = []
    feature_batches: dict[str, FrameFeatureBatch] = {}
    sequence_metrics: dict[str, Any] = {}
    quality_peak_allocated = 0.0
    quality_peak_reserved = 0.0
    grid_shape: tuple[int, int] | None = None
    tokens_per_frame = 0

    with torch.inference_mode():
        for sequence in DEFAULT_SEQUENCES:
            record = prepared.sequences[sequence]
            frame_indices = uniform_sample_indices(record.frame_count, FRAME_SAMPLE_COUNT)
            frame_paths = [record.frame_path_for_index(index) for index in frame_indices]
            batch = extract_frame_features(
                frame_paths=frame_paths,
                frame_indices=frame_indices,
                encoder=encoder,
                device=device,
                crop_size=resolution,
                batch_size=1,
            )
            pca_start = time.perf_counter()
            pca = fit_pca_rgb(batch.features.reshape(-1, batch.feature_dim))
            pca_maps = project_pca_rgb(batch.features, pca)
            pca_seconds = time.perf_counter() - pca_start
            adjacent_cosine = adjacent_patch_cosine(batch.features)
            pca_drift = adjacent_pca_drift(pca_maps)
            feature_batches[sequence] = batch
            grid_shape = batch.grid_shape
            tokens_per_frame = int(batch.grid_shape[0] * batch.grid_shape[1])
            quality_peak_allocated = max(
                quality_peak_allocated,
                float(batch.runtime.get("peak_gpu_memory_mb", 0.0)),
            )
            quality_peak_reserved = max(quality_peak_reserved, _peak_reserved_memory_mb(device))
            visual_records.append(
                {
                    "sequence": sequence,
                    "frame_indices": frame_indices,
                    "frames": batch.frames,
                    "pca_maps": pca_maps,
                }
            )
            sequence_metrics[sequence] = {
                "sampled_frame_indices": [int(index) for index in frame_indices],
                "processed_resolution_hw": list(batch.processed_resolution),
                "feature_grid_shape": list(batch.grid_shape),
                "feature_dimension": int(batch.feature_dim),
                "patch_size": int(batch.patch_size),
                "pca": pca_metric_payload(pca),
                "adjacent_frame_feature_cosine": adjacent_cosine,
                "mean_adjacent_patch_cosine": _nanmean(adjacent_cosine),
                "temporal_pca_color_drift_proxy": pca_drift,
                "mean_temporal_pca_color_drift_proxy": _nanmean(pca_drift),
                "runtime": {
                    "feature_extraction": batch.runtime,
                    "pca_seconds": float(pca_seconds),
                },
            }

        save_phase2_video_pca_summary(output_dir / "video_pca_summary.png", records=visual_records)
        sequence_results = []
        vos_visual_records = []
        for sequence in DEFAULT_SEQUENCES:
            result = run_vos_for_sequence(
                sequence=prepared.sequences[sequence],
                feature_batch=feature_batches[sequence],
            )
            sequence_results.append(result)
            vos_visual_records.append(
                {
                    "sequence": sequence,
                    "frame_indices": feature_batches[sequence].frame_indices,
                    "frames": result["frames"],
                    "gt_masks": result["gt_masks"],
                    "predicted_masks": result["predicted_masks"],
                    "metrics": result["metrics"],
                }
            )

    vos_metrics = aggregate_vos_metrics(sequence_results)
    save_phase2_vos_summary(output_dir / "vos_summary.png", records=vos_visual_records)
    aggregate_j_values = [
        item["aggregate"]["sampled_crop_j_mean"]
        for item in vos_metrics["sequence_metrics"].values()
        if _is_finite_number(item["aggregate"].get("sampled_crop_j_mean"))
    ]
    min_j_values = [
        item["aggregate"].get("sampled_crop_j_min", item["aggregate"].get("sampled_crop_iou_min"))
        for item in vos_metrics["sequence_metrics"].values()
        if _is_finite_number(
            item["aggregate"].get("sampled_crop_j_min", item["aggregate"].get("sampled_crop_iou_min"))
        )
    ]
    low_iou_frames = {
        sequence: item["aggregate"]["low_iou_frames"]
        for sequence, item in vos_metrics["sequence_metrics"].items()
    }
    video_cosines = [
        item["mean_adjacent_patch_cosine"]
        for item in sequence_metrics.values()
        if _is_finite_number(item.get("mean_adjacent_patch_cosine"))
    ]
    video_drifts = [
        item["mean_temporal_pca_color_drift_proxy"]
        for item in sequence_metrics.values()
        if _is_finite_number(item.get("mean_temporal_pca_color_drift_proxy"))
    ]
    total_seconds = time.perf_counter() - stage_start
    metrics = {
        "model": spec.model_id,
        "display_name": spec.display_name,
        "artifact_slug": spec.artifact_slug,
        "official_model_alias": spec.official_alias,
        "resolution": int(resolution),
        "config_slug": make_config_slug(spec.model_id, resolution),
        "feature_mode": FEATURE_MODE,
        "selected_sequences": list(DEFAULT_SEQUENCES),
        "video_dense_pca": {"sequence_metrics": sequence_metrics},
        "vos_label_propagation": vos_metrics,
        "runtime": {
            "elapsed_seconds": float(total_seconds),
            "quality_peak_allocated_memory_mb": float(quality_peak_allocated),
            "quality_peak_reserved_memory_mb": float(quality_peak_reserved),
        },
        "output_files": ["metrics.json", "video_pca_summary.png", "vos_summary.png"],
    }
    row_metrics = {
        "status": "success",
        "grid_shape": format_grid(grid_shape),
        "tokens_per_frame": int(tokens_per_frame),
        "mean_adjacent_cosine": _fmt(_nanmean(video_cosines)),
        "mean_pca_color_drift": _fmt(_nanmean(video_drifts)),
        "sampled_crop_j_mean": _fmt(_nanmean(aggregate_j_values)),
        "sampled_crop_j_min": _fmt(float(np.min(min_j_values)) if min_j_values else math.nan),
        "low_iou_frames": format_low_iou(low_iou_frames),
        "total_quality_seconds": _fmt(total_seconds),
        "error": "",
    }
    write_json(output_dir / "metrics.json", metrics)
    return {"metrics": metrics, "row_metrics": row_metrics}


def benchmark_base_record(
    *,
    config: Phase3Config,
    spec: ModelSpec,
    resolution: int,
    batch_size: int,
    load_result: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    expected_grid, expected_tokens = expected_grid_for_resolution(resolution)
    row: dict[str, Any] = {
        "mode": config.mode,
        "model": spec.model_id,
        "display_name": spec.display_name,
        "artifact_slug": spec.artifact_slug,
        "official_model_alias": spec.official_alias,
        "active_dtype": load_result.get("active_dtype", ""),
        "resolution": int(resolution),
        "batch_size": int(batch_size),
        "status": "success",
        "expected_grid_shape": expected_grid,
        "expected_tokens_per_frame": expected_tokens,
        "grid_shape": "",
        "tokens_per_frame": "",
        "model_load_seconds": load_result.get("model_load_seconds", ""),
        "cuda_preload_allocated_mb": load_result.get("cuda_preload_allocated_mb", ""),
        "cuda_preload_reserved_mb": load_result.get("cuda_preload_reserved_mb", ""),
        "model_resident_allocated_mb": load_result.get("model_resident_allocated_mb", ""),
        "model_resident_reserved_mb": load_result.get("model_resident_reserved_mb", ""),
        "peak_allocated_memory_mb": "",
        "peak_reserved_memory_mb": "",
        "incremental_inference_allocated_mb": "",
        "incremental_inference_reserved_mb": "",
        "total_gpu_memory_mb": _total_gpu_memory_mb(device),
        "memory_utilization_proxy": "",
        "encoder_core_fps": "",
        "pipeline_fps": "",
        "encoder_core_throughput_fps": "",
        "pipeline_throughput_fps": "",
        "total_seconds": "",
        "error": "",
    }
    for prefix in ("preprocess", "encoder_core", "pipeline"):
        for column in _timing_columns(prefix):
            row[column] = ""
    return row


def quality_base_record(
    *,
    config: Phase3Config,
    spec: ModelSpec,
    resolution: int,
) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "model": spec.model_id,
        "display_name": spec.display_name,
        "artifact_slug": spec.artifact_slug,
        "official_model_alias": spec.official_alias,
        "active_dtype": "",
        "resolution": int(resolution),
        "status": "success",
        "grid_shape": "",
        "tokens_per_frame": "",
        "mean_adjacent_cosine": "",
        "mean_pca_color_drift": "",
        "sampled_crop_j_mean": "",
        "sampled_crop_j_min": "",
        "low_iou_frames": "",
        "total_quality_seconds": "",
        "preprocess_ms_per_frame_median": "",
        "encoder_core_ms_per_frame_median": "",
        "encoder_core_fps": "",
        "pipeline_ms_per_frame_median": "",
        "pipeline_fps": "",
        "model_resident_allocated_mb": "",
        "peak_allocated_memory_mb": "",
        "peak_reserved_memory_mb": "",
        "error": "",
    }


def copy_canonical_efficiency(target: dict[str, Any], canonical: dict[str, Any]) -> None:
    for key in (
        "active_dtype",
        "grid_shape",
        "tokens_per_frame",
        "preprocess_ms_per_frame_median",
        "encoder_core_ms_per_frame_median",
        "encoder_core_fps",
        "pipeline_ms_per_frame_median",
        "pipeline_fps",
        "model_resident_allocated_mb",
        "peak_allocated_memory_mb",
        "peak_reserved_memory_mb",
    ):
        target[key] = canonical.get(key, "")


def model_load_record(
    *,
    spec: ModelSpec,
    status: str,
    error: str,
    model_load_seconds: float,
    active_dtype: str,
    preload_allocated: float,
    preload_reserved: float,
    resident_allocated: float,
    resident_reserved: float,
) -> dict[str, Any]:
    return {
        "model": spec.model_id,
        "display_name": spec.display_name,
        "artifact_slug": spec.artifact_slug,
        "official_model_alias": spec.official_alias,
        "status": status,
        "active_dtype": active_dtype,
        "model_load_seconds": float(model_load_seconds),
        "cuda_preload_allocated_mb": float(preload_allocated),
        "cuda_preload_reserved_mb": float(preload_reserved),
        "model_resident_allocated_mb": float(resident_allocated),
        "model_resident_reserved_mb": float(resident_reserved),
        "error": error,
    }


def append_model_load_failures(
    *,
    canonical_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    config: Phase3Config,
    spec: ModelSpec,
    load_result: dict[str, Any],
    device: torch.device,
) -> None:
    for resolution, batch_size in planned_benchmark_configs(config, spec.model_id):
        row = benchmark_base_record(
            config=config,
            spec=spec,
            resolution=resolution,
            batch_size=batch_size,
            load_result=load_result,
            device=device,
        )
        row["status"] = load_result["status"]
        row["error"] = load_result["error"]
        canonical_rows.append(row)
    if config.quality_enabled:
        for model_id, resolution in config.quality_configs:
            if model_id != spec.model_id:
                continue
            row = quality_base_record(config=config, spec=spec, resolution=resolution)
            row["status"] = load_result["status"]
            row["error"] = load_result["error"]
            quality_rows.append(row)


def planned_benchmark_configs(config: Phase3Config, model_id: str) -> list[tuple[int, int]]:
    planned = [(resolution, 1) for resolution in config.resolutions]
    sweep = set(batch_sweep_configs(config))
    for resolution in config.resolutions:
        if (model_id, resolution) in sweep:
            planned.extend((resolution, batch_size) for batch_size in config.batch_sizes if batch_size > 1)
    return planned


def project_feasibility_rows(canonical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in canonical_rows if int(row.get("batch_size", 0)) == 1]


def batch_sweep_configs(config: Phase3Config) -> list[tuple[str, int]]:
    if config.mode == "quick":
        return [("vit_b", 384), ("vit_b", 512)]
    return [
        *(('vit_b', resolution) for resolution in (384, 512, 768, 1024)),
        *(('vit_l', resolution) for resolution in (384, 512, 768)),
        *(('vit_g', resolution) for resolution in (384, 512)),
        *(('vit_G', resolution) for resolution in (384, 512)),
    ]


def summary_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return {"row_count": len(rows), "status_counts": counts, "rows": rows}


def load_phase3_encoder(
    spec: ModelSpec,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    if not hub_alias_available(spec.official_alias):
        raise ModelUnavailableError(f"Torch hub alias is unavailable in this repo: {spec.official_alias}")
    encoder, predictor = torch.hub.load(
        str(REPO_ROOT),
        spec.official_alias,
        source="local",
        pretrained=True,
    )
    del predictor
    gc.collect()
    encoder = encoder.to(device=device, dtype=dtype)
    return encoder.eval()


def detect_model_dtype(model: torch.nn.Module) -> str:
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point():
            return str(tensor.dtype)
    return "unknown"


def hub_alias_available(official_alias: str) -> bool:
    try:
        import hubconf
    except Exception:
        return False
    return hasattr(hubconf, official_alias)


def require_phase3_cuda() -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Phase 3 requires CUDA for non-prepare modes. "
            "Run --prepare-only for dataset validation without CUDA."
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    return device, torch.cuda.get_device_name(0)


def prepare_staged_output(staged_dir: Path = PHASE3_STAGED_OUTPUT_DIR) -> Path:
    staged_dir.parent.mkdir(parents=True, exist_ok=True)
    if staged_dir.exists():
        print(f"[Setup] Removing stale staged output directory: {repo_relative(staged_dir)}")
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=False)
    return staged_dir


def cleanup_staged_output(staged_dir: Path = PHASE3_STAGED_OUTPUT_DIR) -> None:
    if staged_dir.exists():
        shutil.rmtree(staged_dir)


def replace_final_output(staged_dir: Path, final_dir: Path = PHASE3_OUTPUT_DIR) -> None:
    backup_dir = final_dir.with_name(f".{final_dir.name}.previous")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if final_dir.exists():
        print(f"[Setup] Replacing previous Phase 3 outputs: {repo_relative(final_dir)}")
        final_dir.rename(backup_dir)
    try:
        staged_dir.rename(final_dir)
    except Exception:
        if backup_dir.exists() and not final_dir.exists():
            backup_dir.rename(final_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def verify_phase3_output(output_dir: Path) -> None:
    required = {
        "feasibility_matrix.csv",
        "batch_sweep.csv",
        "quality_efficiency.csv",
        "metrics.json",
        "report.md",
        "artifact_index.json",
        "manifest.json",
        *(f"figures/{name}" for name in FIGURE_NAMES),
    }
    actual = set(output_file_list(output_dir))
    missing = required - actual
    if missing:
        raise RuntimeError(f"Phase 3 output directory is missing expected files: {sorted(missing)}")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest_output_paths(output_dir, manifest.get("output_file_list", []))
    validate_casefold_unique_paths(output_dir)


def output_file_list(output_dir: Path, *, exclude: set[str] | None = None) -> list[str]:
    if not output_dir.is_dir():
        return []
    excluded = exclude or set()
    return sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.relative_to(output_dir).as_posix() not in excluded
    )


def validate_casefold_unique_paths(output_dir: Path) -> None:
    if not output_dir.is_dir():
        raise RuntimeError(f"Output directory does not exist: {output_dir}")
    validate_casefold_unique_relative_paths(
        path.relative_to(output_dir).as_posix() for path in sorted(output_dir.rglob("*"))
    )


def validate_casefold_unique_relative_paths(relative_paths: Iterable[str]) -> None:
    folded: dict[str, str] = {}
    for relative in relative_paths:
        key = relative.casefold()
        previous = folded.get(key)
        if previous is not None and previous != relative:
            raise RuntimeError(
                "Phase 3 output paths collide on a case-insensitive filesystem: "
                f"{previous!r} and {relative!r}."
            )
        folded[key] = relative


def validate_successful_quality_artifacts(
    output_dir: Path,
    quality_rows: list[dict[str, Any]],
) -> None:
    required_names = ("metrics.json", "video_pca_summary.png", "vos_summary.png")
    for row in quality_rows:
        if row.get("status") != "success":
            continue
        slug = make_config_slug(str(row["model"]), int(row["resolution"]))
        config_dir = output_dir / "selected_outputs" / slug
        missing = [name for name in required_names if not (config_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Successful quality configuration {slug} is missing artifacts: {missing}"
            )


def build_artifact_index_entries(output_dir: Path) -> list[dict[str, Any]]:
    excluded = {"artifact_index.json", "manifest.json"}
    entries = []
    for relative in output_file_list(output_dir, exclude=excluded):
        path = output_dir / relative
        entries.append(
            {
                "relative_path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return entries


def validate_manifest_output_paths(output_dir: Path, output_files: Iterable[str]) -> None:
    missing = [relative for relative in output_files if not (output_dir / relative).is_file()]
    if missing:
        raise RuntimeError(f"Manifest lists nonexistent Phase 3 files: {missing}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fieldnames})


def write_manifest(
    path: Path,
    *,
    config: Phase3Config,
    prepared: PreparedDavis,
    environment: dict[str, Any],
    frame_pool_payload: dict[str, Any],
    model_load_results: list[dict[str, Any]],
    runtime_summary: dict[str, Any],
    output_files: list[str],
) -> None:
    registry = [model_record(model_id) for model_id in config.models]
    payload = {
        "phase_name": PHASE3_NAME,
        "timestamp": now_utc(),
        **collect_git_provenance(),
        **collect_phase3_source_provenance(),
        "mode": config.mode,
        "model_registry": registry,
        "model_list": list(config.models),
        "active_dtype_by_model": {
            row["model"]: row.get("active_dtype") for row in model_load_results
        },
        "resolution_list": list(config.resolutions),
        "batch_size_list": list(config.batch_sizes),
        "quality_enabled": bool(config.quality_enabled),
        "quality_config_list": [
            {
                "model": model,
                "display_name": MODEL_REGISTRY[model].display_name,
                "artifact_slug": MODEL_REGISTRY[model].artifact_slug,
                "config_slug": make_config_slug(model, resolution),
                "resolution": int(resolution),
            }
            for model, resolution in config.quality_configs
        ],
        "davis_root": repo_relative(prepared.davis_root),
        "prepared_dataset_manifest": repo_relative(prepared.manifest_path),
        "prepared_structure_fingerprint": prepared.manifest.get("structure_fingerprint"),
        "selected_sequences": list(DEFAULT_SEQUENCES),
        "benchmark_frame_pool": frame_pool_payload,
        "canonical_benchmark_protocol": benchmark_protocol(config),
        "authoritative_latency_source": "canonical_batch_sweep_batch1_pipeline",
        "memory_metric_definitions": {
            "model_resident": "CUDA memory after model load and before benchmark input allocation.",
            "incremental_inference_allocated": (
                "max(0, peak_allocated_memory_mb - model_resident_allocated_mb)"
            ),
            "incremental_inference_reserved": (
                "max(0, peak_reserved_memory_mb - model_resident_reserved_mb)"
            ),
            "memory_utilization_proxy": "peak_allocated_memory_mb / total_gpu_memory_mb",
        },
        "model_load_results": model_load_results,
        "runtime_summary": runtime_summary,
        "hardware_info": environment,
        "cuda_version": environment.get("torch_cuda_version"),
        "pytorch_version": environment.get("pytorch_version"),
        "total_gpu_memory_mb": environment.get("total_gpu_memory_mb"),
        "artifact_index_path": "artifact_index.json",
        "output_file_list": output_files,
    }
    write_json(path, payload)


def collect_phase3_source_provenance() -> dict[str, Any]:
    records = []
    for relative_path in PHASE3_SOURCE_RELATIVE_PATHS:
        source_path = REPO_ROOT / relative_path
        if source_path.is_file():
            records.append({"path": relative_path, "sha256": sha256_file(source_path)})
    combined = bytearray()
    for record in records:
        combined.extend(f"{record['path']}\n{record['sha256']}\n".encode("utf-8"))
    return {
        "phase_source_files": records,
        "phase_source_fingerprint": sha256_bytes(bytes(combined)),
    }


def model_record(model_id: str) -> dict[str, str]:
    spec = MODEL_REGISTRY[model_id]
    return {
        "model_id": spec.model_id,
        "display_name": spec.display_name,
        "artifact_slug": spec.artifact_slug,
        "official_model_alias": spec.official_alias,
    }


def select_quality_efficiency_recommendations(
    quality_rows: list[dict[str, Any]],
    minimum_quality_gain: float = 0.01,
) -> dict[str, Any]:
    if not _is_finite_number(minimum_quality_gain) or float(minimum_quality_gain) < 0:
        raise ValueError("minimum_quality_gain must be a finite non-negative number.")
    threshold = float(minimum_quality_gain)
    method = {
        "pareto_filter": True,
        "quality_metric": "sampled_crop_j_mean",
        "authoritative_latency": "pipeline_ms_per_frame_median",
        "authoritative_latency_scope": "canonical_batch_size_1_online_pipeline",
        "balanced_method": (
            "marginal_quality_gain_per_added_pipeline_ms_from_low_latency_baseline"
        ),
        "minimum_absolute_quality_gain": threshold,
        "model_order": list(RECOMMENDATION_MODEL_ORDER),
    }
    records_by_id: dict[str, RecommendationRecord] = {}
    for row in quality_rows:
        record = _normalize_recommendation_row(row)
        if record is None:
            continue
        previous = records_by_id.get(record.config_id)
        if previous is None:
            records_by_id[record.config_id] = record
            continue
        if _recommendation_metrics(previous) != _recommendation_metrics(record):
            raise ValueError(
                "Conflicting duplicate quality-efficiency rows for "
                f"configuration {record.config_id}."
            )

    valid = sorted(records_by_id.values(), key=_valid_recommendation_sort_key)
    base_result: dict[str, Any] = {
        "method": method,
        "valid_configuration_ids": [record.config_id for record in valid],
        "pareto_configuration_ids": [],
        "low_latency": None,
        "balanced": None,
        "high_quality": None,
        "large_model": None,
        "unavailable_reason": None,
    }
    if not valid:
        base_result["unavailable_reason"] = (
            "当前运行未产生有效的质量—效率联合结果，无法生成配置推荐。"
        )
        return base_result

    pareto = sorted(
        [
            candidate
            for candidate in valid
            if not any(
                _dominates(other, candidate)
                for other in valid
                if other.config_id != candidate.config_id
            )
        ],
        key=_pareto_sort_key,
    )
    base_result["pareto_configuration_ids"] = [record.config_id for record in pareto]

    low = min(
        pareto,
        key=lambda record: (
            record.pipeline_ms_per_frame_median,
            -record.sampled_crop_j_mean,
            MODEL_ORDER[record.model],
            record.resolution,
            record.config_id,
        ),
    )
    high = min(
        pareto,
        key=lambda record: (
            -record.sampled_crop_j_mean,
            record.pipeline_ms_per_frame_median,
            MODEL_ORDER[record.model],
            record.resolution,
            record.config_id,
        ),
    )
    low_payload = _recommendation_payload(
        low,
        selection_reason=(
            "该配置是有效质量—延迟 Pareto 集中 canonical batch=1 在线流水线延迟最低的配置。"
        ),
    )
    high_payload = _recommendation_payload(
        high,
        selection_reason=(
            "该配置在有效质量—延迟 Pareto 集中 sampled-crop J 最高；并以更低在线流水线延迟作为第一顺位平局判定。"
        ),
    )

    eligible_balanced: list[tuple[RecommendationRecord, float, float, float]] = []
    for candidate in pareto:
        if candidate.config_id == low.config_id:
            continue
        quality_gain = candidate.sampled_crop_j_mean - low.sampled_crop_j_mean
        additional_latency = (
            candidate.pipeline_ms_per_frame_median - low.pipeline_ms_per_frame_median
        )
        if quality_gain >= threshold and additional_latency > 0:
            eligible_balanced.append(
                (
                    candidate,
                    quality_gain,
                    additional_latency,
                    quality_gain / additional_latency,
                )
            )

    if eligible_balanced:
        balanced_record, quality_gain, additional_latency, marginal_gain = min(
            eligible_balanced,
            key=lambda item: (
                -item[3],
                -item[1],
                -item[0].sampled_crop_j_mean,
                item[0].pipeline_ms_per_frame_median,
                MODEL_ORDER[item[0].model],
                item[0].resolution,
                item[0].config_id,
            ),
        )
        balanced_payload = _recommendation_payload(
            balanced_record,
            selection_reason=(
                f"相对于低延迟基线 {low.config_id}，该配置的 sampled-crop J 提升 "
                f"{quality_gain:.6f}，增加在线流水线延迟 {additional_latency:.6f} ms，"
                f"边际质量收益为 {marginal_gain:.6f} J/ms，在达到至少 "
                f"{threshold:.6f} 绝对 J 提升的 Pareto 配置中最高。"
            ),
        )
        balanced_payload.update(
            {
                "fallback_used": False,
                "baseline_config_id": low.config_id,
                "quality_gain_over_low_latency": quality_gain,
                "additional_latency_ms_over_low_latency": additional_latency,
                "marginal_quality_gain_per_ms": marginal_gain,
            }
        )
    else:
        fallback_reason = (
            f"没有其他 Pareto 配置达到至少 {threshold:.6f} 的绝对 sampled-crop J 提升，"
            "因此平衡候选回退为低延迟候选。"
        )
        balanced_payload = _recommendation_payload(low, selection_reason=fallback_reason)
        balanced_payload.update(
            {
                "fallback_used": True,
                "baseline_config_id": low.config_id,
                "quality_gain_over_low_latency": None,
                "additional_latency_ms_over_low_latency": None,
                "marginal_quality_gain_per_ms": None,
            }
        )

    large_candidates = [record for record in valid if record.model != "vit_b"]
    large_payload = None
    if large_candidates:
        large = min(
            large_candidates,
            key=lambda record: (
                -record.sampled_crop_j_mean,
                record.pipeline_ms_per_frame_median,
                MODEL_ORDER[record.model],
                record.resolution,
                record.config_id,
            ),
        )
        large_payload = _recommendation_payload(
            large,
            selection_reason="该配置是当前已测试非 ViT-B 大模型配置中 sampled-crop J 最高的候选。",
        )

    base_result.update(
        {
            "low_latency": low_payload,
            "balanced": balanced_payload,
            "high_quality": high_payload,
            "large_model": large_payload,
        }
    )
    return base_result


def _normalize_recommendation_row(
    row: dict[str, Any],
) -> RecommendationRecord | None:
    if row.get("status") != "success":
        return None
    model = str(row.get("model", ""))
    if model not in MODEL_REGISTRY:
        return None
    resolution = _positive_integer(row.get("resolution"))
    if resolution is None:
        return None
    batch_size = row.get("batch_size")
    if batch_size not in (None, ""):
        parsed_batch_size = _positive_integer(batch_size)
        if parsed_batch_size != 1:
            return None
    quality = _to_float(row.get("sampled_crop_j_mean"))
    latency = _to_float(row.get("pipeline_ms_per_frame_median"))
    if quality is None or latency is None or latency <= 0:
        return None
    fps = _to_float(row.get("pipeline_fps"))
    if fps is None or fps <= 0:
        fps = 1000.0 / latency
    return RecommendationRecord(
        model=model,
        display_name=MODEL_REGISTRY[model].display_name,
        resolution=resolution,
        config_id=f"{model}@{resolution}",
        sampled_crop_j_mean=quality,
        pipeline_ms_per_frame_median=latency,
        pipeline_fps=fps,
    )


def _positive_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    number = _to_float(value)
    if number is None or number <= 0 or not number.is_integer():
        return None
    return int(number)


def _recommendation_metrics(record: RecommendationRecord) -> tuple[float, float, float]:
    return (
        record.sampled_crop_j_mean,
        record.pipeline_ms_per_frame_median,
        record.pipeline_fps,
    )


def _valid_recommendation_sort_key(record: RecommendationRecord) -> tuple[Any, ...]:
    return (MODEL_ORDER[record.model], record.resolution, record.config_id)


def _pareto_sort_key(record: RecommendationRecord) -> tuple[Any, ...]:
    return (
        record.pipeline_ms_per_frame_median,
        -record.sampled_crop_j_mean,
        MODEL_ORDER[record.model],
        record.resolution,
        record.config_id,
    )


def _dominates(left: RecommendationRecord, right: RecommendationRecord) -> bool:
    no_worse_quality = left.sampled_crop_j_mean >= right.sampled_crop_j_mean
    no_worse_latency = (
        left.pipeline_ms_per_frame_median <= right.pipeline_ms_per_frame_median
    )
    strictly_better = (
        left.sampled_crop_j_mean > right.sampled_crop_j_mean
        or left.pipeline_ms_per_frame_median < right.pipeline_ms_per_frame_median
    )
    return no_worse_quality and no_worse_latency and strictly_better


def _recommendation_payload(
    record: RecommendationRecord,
    *,
    selection_reason: str,
) -> dict[str, Any]:
    return {
        "config_id": record.config_id,
        "model": record.model,
        "display_name": record.display_name,
        "resolution": record.resolution,
        "sampled_crop_j_mean": record.sampled_crop_j_mean,
        "pipeline_ms_per_frame_median": record.pipeline_ms_per_frame_median,
        "pipeline_fps": record.pipeline_fps,
        "selection_reason": selection_reason,
    }


def save_phase3_figures(
    *,
    figures_dir: Path,
    feasibility_rows: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    quality_enabled: bool,
    recommendations: dict[str, Any],
) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_line_by_model(
        plt,
        figures_dir / "latency_vs_resolution.png",
        rows=feasibility_rows,
        x_key="resolution",
        y_key="pipeline_ms_per_frame_median",
        title="Batch=1 online pipeline latency vs resolution",
        ylabel="median online pipeline latency (ms/frame)",
    )
    _plot_memory_by_model(plt, figures_dir / "memory_vs_resolution.png", feasibility_rows)
    _plot_line_by_config(
        plt,
        figures_dir / "throughput_vs_batch.png",
        rows=batch_rows,
        x_key="batch_size",
        y_key="pipeline_throughput_fps",
        title="Online pipeline throughput vs batch size",
        ylabel="pipeline throughput (frames/s)",
    )
    if quality_enabled and quality_rows:
        _plot_scatter(
            plt,
            figures_dir / "quality_vs_latency.png",
            rows=quality_rows,
            x_key="pipeline_ms_per_frame_median",
            y_key="sampled_crop_j_mean",
            title="Sampled VOS quality vs batch=1 pipeline latency",
            xlabel="median online pipeline latency (ms/frame)",
            ylabel="sampled crop J mean",
        )
        _plot_scatter(
            plt,
            figures_dir / "quality_vs_memory.png",
            rows=quality_rows,
            x_key="peak_allocated_memory_mb",
            y_key="sampled_crop_j_mean",
            title="Sampled VOS quality vs peak allocated memory",
            xlabel="peak allocated memory (MB)",
            ylabel="sampled crop J mean",
        )
        _plot_pareto(
            plt,
            figures_dir / "pareto_front.png",
            quality_rows,
            recommendations,
        )
    else:
        for name, title in (
            ("quality_vs_latency.png", "Quality vs pipeline latency"),
            ("quality_vs_memory.png", "Quality vs peak memory"),
            ("pareto_front.png", "Quality-efficiency Pareto front"),
        ):
            _plot_placeholder(
                plt,
                figures_dir / name,
                title,
                "Quality evaluation was disabled or produced no valid rows.",
            )


def write_report(path: Path, *, metrics: dict[str, Any]) -> None:
    feasibility = metrics["feasibility_summary"]["rows"]
    batch = metrics["batch_sweep_summary"]["rows"]
    quality = metrics["quality_efficiency_summary"]["rows"]
    env = metrics["environment"]
    model_loads = metrics["model_load_results"]
    recommendations = format_recommendations_for_report(
        metrics["quality_efficiency_summary"]["recommendations"]
    )
    active_dtypes = sorted(
        {str(row.get("active_dtype")) for row in model_loads if row.get("active_dtype")}
    )
    dtype_text = ", ".join(active_dtypes) if active_dtypes else "未能从成功运行中检测"
    lines = [
        "# Phase 3：V-JEPA 2.1 稠密特征效率与算力需求评估报告",
        "",
        "## 1. 实验目的",
        "",
        "Phase 3 评估 V-JEPA 2.1 稠密特征的算力需求与效率，不是新的表征能力验证。Phase 2 已验证基础稠密特征结构和 sampled DAVIS VOS 传播能力；本阶段研究其在实际延迟、显存和吞吐约束下是否实用。",
        "",
        "## 2. 测试变量与固定条件",
        "",
        f"- 模式：`{metrics['mode']}`。",
        f"- 模型：{', '.join(row['display_name'] for row in model_loads)}。",
        f"- 数据：DAVIS 2017 TrainVal 480p，序列 {', '.join(DEFAULT_SEQUENCES)}。",
        f"- 实际检测到的推理 dtype：{dtype_text}。",
        "- DAVIS 480p 上采样到 1024 不会增加新的物理图像信息；潜在收益来自更密的 patch grid 以及更细的 mask 离散化或对齐。",
        "",
        "## 3. 硬件与软件环境",
        "",
        f"- 设备：{env.get('device')} / {env.get('gpu_name')}。",
        f"- PyTorch：{env.get('pytorch_version')}；CUDA：{env.get('torch_cuda_version')}。",
        f"- GPU 总显存：{_display_number(env.get('total_gpu_memory_mb'), suffix=' MB')}。",
        "- 模型加载是冷启动成本，包含构建、checkpoint 加载和迁移到 CUDA，不计入稳态延迟或 FPS。",
        "",
        "## 4. 单帧可行性矩阵",
        "",
        "权威在线指标为 canonical batch=1 的 `pipeline_ms_per_frame_median`。它直接测量内存中已解码 RGB 的 resize/crop、tensor 转换、归一化与堆叠、CPU→GPU 传输、encoder forward 和 CUDA 同步。它不包含磁盘读取、JPEG 解码、PCA、VOS 传播、可视化或模型加载。Encoder-core 延迟仅作为 GPU 计算参考。",
        "",
        _status_table(
            feasibility,
            value_key="pipeline_ms_per_frame_median",
            value_label="pipeline ms/frame",
        ),
        "",
        "## 5. Batch / 显存利用率测试",
        "",
        _status_table(
            batch,
            value_key="pipeline_throughput_fps",
            value_label="pipeline fps",
        ),
        "",
        "峰值 allocated 显存反映部署总需求；model-resident 是模型常驻部分，incremental inference 是输入和推理临时增量。大 batch pipeline 吞吐描述云端或批处理能力，不等价于低延迟在线推理。",
        "",
        "## 6. 效果—效率联合分析",
        "",
    ]
    if metrics["quality_enabled"]:
        lines.extend(
            [
                _quality_table(quality),
                "",
                model_scale_conclusion(quality),
                "",
                resolution_conclusion(quality),
            ]
        )
    else:
        lines.append("本次使用 `--no-quality`，未运行效果—效率评估；质量 CSV 只有表头，相关图表为可读占位。")
    lines.extend(
        [
            "",
            "当前 VOS 指标是 sampled aligned-crop J/IoU，不是官方 DAVIS J&F；固定 top-k/temperature 传播超参数也未必对每个模型都最优。",
            "",
            "## 7. 实时性分级",
            "",
            "- `>= 30 FPS`：实时视频。",
            "- `10–30 FPS`：准实时。",
            "- `1–10 FPS`：低频在线处理。",
            "- `< 1 FPS`：离线/云端分析。",
            "",
            _realtime_summary(feasibility),
            "",
            "## 8. 边端部署可行性判断",
            "",
            "RTX 4090 结果是边端可行性的上界参考，不是 Jetson 或其他边缘设备实测。如果 4090 的 batch=1 pipeline 延迟已经较慢，边端通常需要压缩、蒸馏或专门推理优化。只有大 batch 吞吐较好而 batch=1 延迟较差时，更适合云端批处理。",
            "",
            "## 9. 推荐配置",
            "",
            recommendations,
            "",
            "## 10. 当前限制",
            "",
            "- 本阶段不是官方 DAVIS J&F，不与论文榜单直接比较。",
            "- 未实现 VSLAM、depth/segmentation probe、video-tokenizer 对比或边缘设备部署。",
            "- 结论仅针对三个 DAVIS 序列、当前采样和固定 VOS 传播协议。",
            "- 配置推荐不表示其在所有下游任务中普遍更优；大模型可能需要单独调整传播超参数。",
            "- artifact slug 中的 `lc`/`uc` 只用于跨平台文件路径，不是模型科学名称。",
            "",
            "完整实际文件列表和 SHA-256 索引见 `manifest.json` 与 `artifact_index.json`。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def model_scale_conclusion(quality_rows: list[dict[str, Any]]) -> str:
    order = {model_id: index for index, model_id in enumerate(FULL_MODELS)}
    rows = sorted(
        [
            row for row in quality_rows
            if row.get("status") == "success"
            and int(row.get("resolution", 0)) == 512
            and _is_finite_number(row.get("sampled_crop_j_mean"))
        ],
        key=lambda row: order.get(str(row["model"]), 99),
    )
    if len(rows) < 2:
        return "模型尺度结论：512 分辨率下有效质量数据不足，无法可靠判断尺度趋势。"
    values = [float(row["sampled_crop_j_mean"]) for row in rows]
    detail = "、".join(
        f"{row['display_name']}={float(row['sampled_crop_j_mean']):.4f}" for row in rows
    )
    monotonic = all(right >= left for left, right in zip(values, values[1:]))
    if monotonic:
        return (
            f"模型尺度结论：当前 512 probe 的成功结果为 {detail}，呈非下降趋势；"
            "这仍不能推广为更大模型拥有更好的整体表征质量。"
        )
    return (
        f"模型尺度结论：当前 512 probe 的成功结果为 {detail}，质量未随模型尺度单调增加。"
        "在三个 DAVIS 序列和固定 top-k/temperature 协议下，更大模型不一定优于 ViT-B；"
        "这不表示更大模型的整体表征质量更差，传播超参数可能并非对每个模型最优。"
    )


def resolution_conclusion(quality_rows: list[dict[str, Any]]) -> str:
    rows = sorted(
        [
            row for row in quality_rows
            if row.get("status") == "success"
            and row.get("model") == "vit_b"
            and _is_finite_number(row.get("sampled_crop_j_mean"))
        ],
        key=lambda row: int(row["resolution"]),
    )
    if len(rows) < 2:
        return "分辨率结论：ViT-B 有效质量数据不足，无法可靠判断分辨率趋势。"
    values = [float(row["sampled_crop_j_mean"]) for row in rows]
    detail = "、".join(
        f"{int(row['resolution'])}={float(row['sampled_crop_j_mean']):.4f}" for row in rows
    )
    monotonic = all(right >= left for left, right in zip(values, values[1:]))
    if monotonic:
        return (
            f"分辨率结论：ViT-B 的成功结果为 {detail}，当前 VOS probe 中提高分辨率带来更清晰且一致的质量改善；"
            "若同一次运行的 512 模型尺度结果仍非单调，则该趋势比单纯增加模型尺度更清晰。该结论是任务和 probe 特定的。"
        )
    return (
        f"分辨率结论：ViT-B 的成功结果为 {detail}，未形成稳定单调趋势；"
        "当前数据不足以声称提高分辨率优于增加模型尺度。"
    )


def format_recommendations_for_report(recommendations: dict[str, Any]) -> str:
    low = recommendations.get("low_latency")
    balanced = recommendations.get("balanced")
    high = recommendations.get("high_quality")
    large = recommendations.get("large_model")
    if low is None or balanced is None or high is None:
        return str(
            recommendations.get("unavailable_reason")
            or "当前运行未产生有效的质量—效率联合结果，无法生成配置推荐。"
        )

    threshold = float(
        recommendations["method"].get("minimum_absolute_quality_gain", 0.01)
    )
    lines = [
        (
            f"- 低延迟候选：{low['display_name']} (`{low['config_id']}`)，"
            f"sampled-crop J={float(low['sampled_crop_j_mean']):.4f}，"
            f"batch=1 pipeline={float(low['pipeline_ms_per_frame_median']):.2f} ms。"
            f"{low['selection_reason']}"
        )
    ]
    if balanced.get("fallback_used"):
        lines.append(
            f"- 质量—效率平衡候选：{balanced['display_name']} (`{balanced['config_id']}`)。"
            f"{balanced['selection_reason']}"
        )
    else:
        lines.append(
            f"- 质量—效率平衡候选：{balanced['display_name']} (`{balanced['config_id']}`)。"
            f"以 `{balanced['baseline_config_id']}` 为最低延迟基线，sampled-crop J 提升 "
            f"{float(balanced['quality_gain_over_low_latency']):.4f}，在线流水线延迟增加 "
            f"{float(balanced['additional_latency_ms_over_low_latency']):.2f} ms，"
            f"边际收益为 {float(balanced['marginal_quality_gain_per_ms']):.6f} J/ms。"
            f"该候选在绝对 J 提升至少 {threshold:.2f} 的 Pareto 配置中，"
            "实现了相对于最低延迟候选每增加 1 ms 在线流水线延迟所获得的最高 sampled-crop J 提升。"
        )
    lines.append(
        f"- 高质量候选：{high['display_name']} (`{high['config_id']}`)，"
        f"sampled-crop J={float(high['sampled_crop_j_mean']):.4f}，"
        f"batch=1 pipeline FPS={float(high['pipeline_fps']):.2f}。"
        f"{high['selection_reason']}"
    )
    if large is not None:
        lines.append(
            f"- 大模型候选：{large['display_name']} (`{large['config_id']}`)，"
            f"sampled-crop J={float(large['sampled_crop_j_mean']):.4f}。"
            "它仅表示当前已测试大模型配置中质量最高的候选，不代表普遍优于 ViT-B 或其他下游任务。"
        )
    return "\n".join(lines)


def _plot_placeholder(plt, output_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _valid_xy(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
) -> list[tuple[dict[str, Any], float, float]]:
    output = []
    for row in rows:
        if row.get("status") != "success":
            continue
        x = _to_float(row.get(x_key))
        y = _to_float(row.get(y_key))
        if x is not None and y is not None:
            output.append((row, x, y))
    return output


def _plot_line_by_model(
    plt,
    output_path: Path,
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    ylabel: str,
) -> None:
    valid = _valid_xy(rows, x_key, y_key)
    if not valid:
        _plot_placeholder(plt, output_path, title, "No successful measurements available.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for model_id in FULL_MODELS:
        points = sorted((x, y) for row, x, y in valid if row["model"] == model_id)
        if points:
            ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                marker="o",
                label=MODEL_REGISTRY[model_id].display_name,
            )
    ax.set_title(title)
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_line_by_config(
    plt,
    output_path: Path,
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    ylabel: str,
) -> None:
    valid = _valid_xy(rows, x_key, y_key)
    if not valid:
        _plot_placeholder(plt, output_path, title, "No successful measurements available.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    labels = sorted(
        {(row["model"], int(row["resolution"])) for row, _, _ in valid},
        key=lambda item: (FULL_MODELS.index(item[0]), item[1]),
    )
    for model_id, resolution in labels:
        points = sorted(
            (x, y)
            for row, x, y in valid
            if row["model"] == model_id and int(row["resolution"]) == resolution
        )
        ax.plot(
            [x for x, _ in points],
            [y for _, y in points],
            marker="o",
            label=f"{MODEL_REGISTRY[model_id].display_name}@{resolution}",
        )
    ax.set_title(title)
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_memory_by_model(plt, output_path: Path, rows: list[dict[str, Any]]) -> None:
    valid = _valid_xy(rows, "resolution", "peak_allocated_memory_mb")
    if not valid:
        _plot_placeholder(
            plt,
            output_path,
            "Deployment memory vs resolution",
            "No successful measurements available.",
        )
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for model_id in FULL_MODELS:
        points = sorted((x, y) for row, x, y in valid if row["model"] == model_id)
        if points:
            ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                marker="o",
                label=f"{MODEL_REGISTRY[model_id].display_name} peak",
            )
    ax.set_title("Deployment peak allocated memory vs resolution")
    ax.set_xlabel("resolution")
    ax.set_ylabel("peak allocated memory (MB)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(
    plt,
    output_path: Path,
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    valid = _valid_xy(rows, x_key, y_key)
    if not valid:
        _plot_placeholder(plt, output_path, title, "No successful quality measurements available.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for row, x, y in valid:
        ax.scatter([x], [y])
        ax.annotate(
            f"{row['display_name']}@{row['resolution']}",
            (x, y),
            fontsize=8,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pareto(
    plt,
    output_path: Path,
    rows: list[dict[str, Any]],
    recommendations: dict[str, Any],
) -> None:
    records_by_id: dict[str, RecommendationRecord] = {}
    valid_ids = set(recommendations.get("valid_configuration_ids", []))
    for row in rows:
        record = _normalize_recommendation_row(row)
        if record is not None and record.config_id in valid_ids:
            records_by_id.setdefault(record.config_id, record)
    ordered_valid = [
        records_by_id[config_id]
        for config_id in recommendations.get("valid_configuration_ids", [])
        if config_id in records_by_id
    ]
    if not ordered_valid:
        _plot_placeholder(
            plt,
            output_path,
            "Quality-efficiency Pareto front",
            "No successful quality measurements available.",
        )
        return
    pareto_ids = recommendations.get("pareto_configuration_ids", [])
    pareto_id_set = set(pareto_ids)
    pareto = [records_by_id[config_id] for config_id in pareto_ids if config_id in records_by_id]
    dominated = [record for record in ordered_valid if record.config_id not in pareto_id_set]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    if dominated:
        ax.scatter(
            [record.pipeline_ms_per_frame_median for record in dominated],
            [record.sampled_crop_j_mean for record in dominated],
            marker="x",
            label="dominated configurations",
        )
    if pareto:
        ax.scatter(
            [record.pipeline_ms_per_frame_median for record in pareto],
            [record.sampled_crop_j_mean for record in pareto],
            marker="o",
            label="pareto configurations",
        )
        ax.plot(
            [record.pipeline_ms_per_frame_median for record in pareto],
            [record.sampled_crop_j_mean for record in pareto],
            label="pareto front",
        )
    for record in ordered_valid:
        ax.annotate(
            f"{record.display_name}@{record.resolution}",
            (record.pipeline_ms_per_frame_median, record.sampled_crop_j_mean),
            fontsize=8,
            xytext=(4, 3),
            textcoords="offset points",
        )

    role_labels = {
        "low_latency": "low latency",
        "balanced": "balanced",
        "high_quality": "high quality",
    }
    role_markers = {
        "low_latency": "s",
        "balanced": "D",
        "high_quality": "*",
    }
    roles_by_config: dict[str, list[str]] = {}
    for role in role_labels:
        candidate = recommendations.get(role)
        if candidate is not None:
            roles_by_config.setdefault(str(candidate["config_id"]), []).append(role)
    for config_id, roles in roles_by_config.items():
        record = records_by_id.get(config_id)
        if record is None:
            continue
        combined_label = " / ".join(role_labels[role] for role in roles)
        marker = role_markers[roles[0]] if len(roles) == 1 else "P"
        ax.scatter(
            [record.pipeline_ms_per_frame_median],
            [record.sampled_crop_j_mean],
            marker=marker,
            s=90,
            label=combined_label,
        )
        ax.annotate(
            combined_label,
            (record.pipeline_ms_per_frame_median, record.sampled_crop_j_mean),
            fontsize=8,
            xytext=(4, -12),
            textcoords="offset points",
        )
    ax.set_title("Quality-efficiency Pareto front")
    ax.set_xlabel("batch=1 median online pipeline latency (ms/frame)")
    ax.set_ylabel("sampled crop J mean")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _status_table(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    value_label: str,
) -> str:
    if not rows:
        return "无记录。"
    lines = [
        f"| model | resolution | batch | status | {value_label} | peak MB |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows[:40]:
        lines.append(
            f"| {row.get('display_name', row.get('model', ''))} | {row.get('resolution', '')} | "
            f"{row.get('batch_size', '')} | {row.get('status', '')} | {_cell(row.get(value_key))} | "
            f"{_cell(row.get('peak_allocated_memory_mb'))} |"
        )
    if len(rows) > 40:
        lines.append("| ... | ... | ... | ... | ... | ... |")
    return "\n".join(lines)


def _quality_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "质量评估未产生记录。"
    lines = [
        "| model | resolution | status | J mean | J min | pipeline ms/frame |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('display_name', row.get('model', ''))} | {row.get('resolution', '')} | "
            f"{row.get('status', '')} | {_cell(row.get('sampled_crop_j_mean'))} | "
            f"{_cell(row.get('sampled_crop_j_min'))} | "
            f"{_cell(row.get('pipeline_ms_per_frame_median'))} |"
        )
    return "\n".join(lines)


def _realtime_summary(rows: list[dict[str, Any]]) -> str:
    valid = [
        row for row in rows
        if row.get("status") == "success" and _is_finite_number(row.get("pipeline_fps"))
    ]
    if not valid:
        return "有效 canonical batch=1 pipeline 数据不足，无法实时性分级。"
    lines = []
    for row in sorted(valid, key=lambda item: (FULL_MODELS.index(item["model"]), int(item["resolution"]))):
        fps = float(row["pipeline_fps"])
        if fps >= 30:
            level = "实时视频"
        elif fps >= 10:
            level = "准实时"
        elif fps >= 1:
            level = "低频在线处理"
        else:
            level = "离线/云端分析"
        lines.append(
            f"- {row['display_name']} (`{row['model']}@{row['resolution']}`)：{fps:.2f} FPS，{level}。"
        )
    return "\n".join(lines)


def cleanup_cuda_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)


def cleanup_cuda_state(device: torch.device) -> None:
    gc.collect()
    release_cuda(device=device)


def reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)


def _allocated_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_allocated(device) / 1024**2)


def _reserved_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_reserved(device) / 1024**2)


def _peak_allocated_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def _peak_reserved_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_reserved(device) / 1024**2)


def _total_gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.get_device_properties(device).total_memory / 1024**2)


def handle_cuda_failure(error: BaseException, device: torch.device) -> None:
    del error
    if device.type == "cuda":
        try:
            synchronize(device)
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass


def status_from_exception(error: BaseException) -> str:
    if isinstance(error, ModelUnavailableError):
        return "unavailable"
    if is_oom_error(error):
        return "oom"
    message = compact_error(error).lower()
    if "has no attribute" in message or "unknown model" in message or "not found in hubconf" in message:
        return "unavailable"
    return "error"


def is_oom_error(error: BaseException) -> bool:
    message = compact_error(error).lower()
    return any(
        needle in message
        for needle in (
            "out of memory",
            "cuda oom",
            "cublas_status_alloc_failed",
            "cuda error: out of memory",
            "memory allocation",
        )
    )


def compact_error(error: BaseException) -> str:
    message = str(error).replace("\n", " ").strip()
    return message if len(message) <= 500 else message[:497] + "..."


def safe_divide(numerator: float, denominator: float) -> float:
    return math.nan if denominator <= 0 else float(numerator / denominator)


def expected_grid_for_resolution(resolution: int, patch_size: int = 16) -> tuple[str, int]:
    if resolution % patch_size:
        return "", 0
    grid = resolution // patch_size
    return f"{grid}x{grid}", int(grid * grid)


def format_grid(grid_shape: tuple[int, int] | None) -> str:
    return "" if grid_shape is None else f"{int(grid_shape[0])}x{int(grid_shape[1])}"


def format_low_iou(value: dict[str, list[int]]) -> str:
    chunks = [
        f"{sequence}:{','.join(str(int(frame)) for frame in frames)}"
        for sequence, frames in value.items()
        if frames
    ]
    return "; ".join(chunks) if chunks else "none"


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.6f}"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return "" if value is None else value


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "" if not math.isfinite(number) else f"{number:.{digits}f}"


def _nanmean(values: list[float]) -> float:
    clean = [float(value) for value in values if _is_finite_number(value)]
    return float(np.mean(clean)) if clean else math.nan


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _display_number(value: Any, *, suffix: str = "") -> str:
    number = _to_float(value)
    return "unknown" if number is None else f"{number:.2f}{suffix}"


def _cell(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return str(value) if value not in (None, "") else ""
    return f"{number:.4f}"


def _seconds(value: float) -> str:
    return f"{float(value):.2f}s"
