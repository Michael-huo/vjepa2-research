from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from research.scripts.common.phase2_data import (
    DAVIS_DOWNLOAD_COMMAND,
    DEFAULT_SEQUENCES,
    FRAME_SAMPLE_COUNT,
    PHASE2_NAME,
    PHASE2_OUTPUT_DIR,
    PHASE2_STAGED_OUTPUT_DIR,
    PREPARED_MANIFEST_PATH,
    PreparedDavis,
    prepare_davis_dataset,
    repo_relative,
    uniform_sample_indices,
)


EXPECTED_OUTPUT_FILES = {
    "image_dense_pca/image_pca_summary.png",
    "image_dense_pca/metrics.json",
    "video_dense_pca/video_pca_summary.png",
    "video_dense_pca/metrics.json",
    "vos_label_propagation/vos_summary.png",
    "vos_label_propagation/metrics.json",
    "phase2_dense_summary.png",
    "summary.csv",
    "metrics.json",
    "report.md",
    "manifest.json",
}

PHASE2_SOURCE_RELATIVE_PATHS = (
    "research/scripts/run_phase2_dense.py",
    "research/scripts/common/phase2_data.py",
    "research/scripts/common/dense_pca.py",
    "research/scripts/common/vos.py",
    "research/scripts/common/visualization.py",
)


def _seconds(value: float) -> str:
    return f"{value:.2f}s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 dense DAVIS experiments with V-JEPA 2.1 framewise image features."
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only validate DAVIS and write/reuse the prepared dataset manifest.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Rewrite the prepared DAVIS manifest even if the structure fingerprint matches.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=list(DEFAULT_SEQUENCES),
        help="DAVIS sequences to use. Defaults to dog car-shadow parkour.",
    )
    return parser


def _sequence_tuple(values: list[str]) -> tuple[str, ...]:
    output: list[str] = []
    for value in values:
        if value not in output:
            output.append(value)
    if not output:
        raise ValueError("At least one DAVIS sequence must be selected.")
    return tuple(output)


def _prepare_staged_output(staged_dir: Path = PHASE2_STAGED_OUTPUT_DIR) -> Path:
    staged_dir.parent.mkdir(parents=True, exist_ok=True)
    if staged_dir.exists():
        print(f"[Setup] Removing stale staged output directory: {repo_relative(staged_dir)}")
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=False)
    return staged_dir


def _cleanup_staged_output(staged_dir: Path) -> None:
    if staged_dir.exists():
        shutil.rmtree(staged_dir)


def _output_file_list(output_dir: Path) -> list[str]:
    if not output_dir.is_dir():
        return []
    return sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    )


def _verify_phase2_output(output_dir: Path) -> None:
    actual = set(_output_file_list(output_dir))
    missing = EXPECTED_OUTPUT_FILES - actual
    unexpected = actual - EXPECTED_OUTPUT_FILES
    if missing or unexpected:
        raise RuntimeError(
            "Phase 2 output directory has an unexpected file set. "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _replace_final_output(staged_dir: Path, final_dir: Path = PHASE2_OUTPUT_DIR) -> None:
    _verify_phase2_output(staged_dir)
    if final_dir.exists():
        print(f"[Setup] Removing previous Phase 2 outputs: {repo_relative(final_dir)}")
        shutil.rmtree(final_dir)
    staged_dir.rename(final_dir)
    print("[Done] Replaced Phase 2 output directory:")
    print(f"       {repo_relative(final_dir)}/")


def _require_phase2_cuda(torch_module):
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "Phase 2 requires CUDA. Activate the CUDA-enabled vjepa environment "
            "and verify torch.cuda.is_available()."
        )
    torch_module.cuda.set_device(0)
    device = torch_module.device("cuda")
    gpu_name = torch_module.cuda.get_device_name(0)
    return device, gpu_name


def _collect_phase2_source_provenance(*, repo_root: Path, sha256_file, sha256_bytes) -> dict[str, Any]:
    records = []
    for relative_path in PHASE2_SOURCE_RELATIVE_PATHS:
        source_path = repo_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Required Phase 2 source file is missing: {relative_path}")
        records.append({"path": relative_path, "sha256": sha256_file(source_path)})
    combined = bytearray()
    for record in records:
        combined.extend(f"{record['path']}\n{record['sha256']}\n".encode("utf-8"))
    return {
        "phase_source_files": records,
        "phase_source_fingerprint": sha256_bytes(bytes(combined)),
    }


def _run_image_dense_pca(
    *,
    output_dir: Path,
    prepared: PreparedDavis,
    sequences: tuple[str, ...],
    encoder,
    device,
    write_json,
    relative_to_repo,
) -> dict[str, Any]:
    from research.scripts.common.dense_pca import (
        FEATURE_MODE,
        fit_pca_rgb,
        model_metadata,
        pca_metric_payload,
        project_pca_rgb,
        extract_frame_features,
    )
    from research.scripts.common.visualization import save_phase2_image_pca_summary

    stage_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = [prepared.sequences[sequence].frame_path_for_index(0) for sequence in sequences]
    batch = extract_frame_features(
        frame_paths=frame_paths,
        frame_indices=[0 for _ in frame_paths],
        encoder=encoder,
        device=device,
    )
    pca_start = time.perf_counter()
    pca = fit_pca_rgb(batch.features.reshape(-1, batch.feature_dim))
    pca_maps = project_pca_rgb(batch.features, pca)
    pca_seconds = time.perf_counter() - pca_start

    records = []
    frame_metrics = []
    for index, sequence in enumerate(sequences):
        records.append(
            {
                "sequence": sequence,
                "frame_index": 0,
                "frame": batch.frames[index],
                "pca_map": pca_maps[index],
            }
        )
        frame_metrics.append(
            {
                "sequence": sequence,
                "input_frame_path": relative_to_repo(frame_paths[index]),
                "processed_resolution_hw": list(batch.processed_resolution),
                "original_size_hw": list(batch.original_sizes[index]),
                "crop_metadata": batch.crop_metadata[index],
            }
        )

    save_phase2_image_pca_summary(output_dir / "image_pca_summary.png", records=records)
    metrics = {
        "stage": "image_dense_pca",
        **model_metadata(),
        "selected_sequences": list(sequences),
        "frames": frame_metrics,
        "processed_resolution_hw": list(batch.processed_resolution),
        "feature_grid_shape": list(batch.grid_shape),
        "feature_dimension": int(batch.feature_dim),
        "patch_size": int(batch.patch_size),
        "pca": pca_metric_payload(pca),
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - stage_start),
            "feature_extraction": batch.runtime,
            "pca_seconds": float(pca_seconds),
            "peak_gpu_memory_mb": float(batch.runtime["peak_gpu_memory_mb"]),
        },
        "notes": [
            "PCA maps are dense feature visualizations, not semantic segmentation.",
            f"Feature mode: {FEATURE_MODE}.",
        ],
    }
    write_json(output_dir / "metrics.json", metrics)
    return {"metrics": metrics, "records": records}


def _run_video_dense_pca(
    *,
    output_dir: Path,
    prepared: PreparedDavis,
    sequences: tuple[str, ...],
    encoder,
    device,
    write_json,
    relative_to_repo,
) -> dict[str, Any]:
    from research.scripts.common.dense_pca import (
        adjacent_patch_cosine,
        adjacent_pca_drift,
        extract_frame_features,
        fit_pca_rgb,
        model_metadata,
        pca_metric_payload,
        project_pca_rgb,
    )
    from research.scripts.common.visualization import save_phase2_video_pca_summary

    stage_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_records: list[dict[str, Any]] = []
    sequence_metrics: dict[str, Any] = {}
    feature_batches: dict[str, Any] = {}
    peak_memory = 0.0

    for sequence in sequences:
        record = prepared.sequences[sequence]
        frame_indices = uniform_sample_indices(record.frame_count, FRAME_SAMPLE_COUNT)
        frame_paths = [record.frame_path_for_index(index) for index in frame_indices]
        batch = extract_frame_features(
            frame_paths=frame_paths,
            frame_indices=frame_indices,
            encoder=encoder,
            device=device,
        )
        pca_start = time.perf_counter()
        pca = fit_pca_rgb(batch.features.reshape(-1, batch.feature_dim))
        pca_maps = project_pca_rgb(batch.features, pca)
        pca_seconds = time.perf_counter() - pca_start
        adjacent_cosine = adjacent_patch_cosine(batch.features)
        pca_drift = adjacent_pca_drift(pca_maps)
        peak_memory = max(peak_memory, float(batch.runtime["peak_gpu_memory_mb"]))
        feature_batches[sequence] = batch
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
            "sampled_frame_paths": [relative_to_repo(path) for path in frame_paths],
            "processed_resolution_hw": list(batch.processed_resolution),
            "feature_grid_shape": list(batch.grid_shape),
            "feature_dimension": int(batch.feature_dim),
            "patch_size": int(batch.patch_size),
            "pca": pca_metric_payload(pca),
            "adjacent_frame_feature_cosine": adjacent_cosine,
            "mean_adjacent_patch_cosine": float(np.mean(adjacent_cosine)) if adjacent_cosine else float("nan"),
            "temporal_pca_color_drift_proxy": pca_drift,
            "mean_temporal_pca_color_drift_proxy": float(np.mean(pca_drift)) if pca_drift else float("nan"),
            "runtime": {
                "feature_extraction": batch.runtime,
                "pca_seconds": float(pca_seconds),
            },
        }

    save_phase2_video_pca_summary(output_dir / "video_pca_summary.png", records=visual_records)
    metrics = {
        "stage": "video_dense_pca",
        **model_metadata(),
        "selected_sequences": list(sequences),
        "frames_per_sequence": int(FRAME_SAMPLE_COUNT),
        "sequence_metrics": sequence_metrics,
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - stage_start),
            "peak_gpu_memory_mb": float(peak_memory),
        },
        "notes": [
            "Each sequence fits one PCA transform over all sampled frame patches.",
            "PCA RGB colors are comparable within a sequence, not across independently fitted sequences.",
        ],
    }
    write_json(output_dir / "metrics.json", metrics)
    return {"metrics": metrics, "records": visual_records, "feature_batches": feature_batches}


def _run_vos_label_propagation(
    *,
    output_dir: Path,
    prepared: PreparedDavis,
    sequences: tuple[str, ...],
    feature_batches: dict[str, Any],
    write_json,
) -> dict[str, Any]:
    from research.scripts.common.vos import aggregate_vos_metrics, run_vos_for_sequence
    from research.scripts.common.visualization import save_phase2_vos_summary

    stage_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_results = []
    visual_records = []
    for sequence in sequences:
        result = run_vos_for_sequence(
            sequence=prepared.sequences[sequence],
            feature_batch=feature_batches[sequence],
        )
        sequence_results.append(result)
        visual_records.append(
            {
                "sequence": sequence,
                "frame_indices": feature_batches[sequence].frame_indices,
                "frames": result["frames"],
                "gt_masks": result["gt_masks"],
                "predicted_masks": result["predicted_masks"],
                "metrics": result["metrics"],
            }
        )

    metrics = aggregate_vos_metrics(sequence_results)
    metrics["stage"] = "vos_label_propagation"
    metrics["runtime"] = {"elapsed_seconds": float(time.perf_counter() - stage_start)}
    save_phase2_vos_summary(output_dir / "vos_summary.png", records=visual_records)
    write_json(output_dir / "metrics.json", metrics)
    return {"metrics": metrics, "records": visual_records}


def _runtime_summary(
    *,
    phase_start_time: float,
    prepare_seconds: float,
    model_load_seconds: float,
    image_metrics: dict[str, Any],
    video_metrics: dict[str, Any],
    vos_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "end_to_end_total_seconds": float(time.perf_counter() - phase_start_time),
        "prepare_seconds": float(prepare_seconds),
        "model_load_seconds": float(model_load_seconds),
        "image_dense_pca_seconds": float(image_metrics["runtime"]["elapsed_seconds"]),
        "video_dense_pca_seconds": float(video_metrics["runtime"]["elapsed_seconds"]),
        "vos_label_propagation_seconds": float(vos_metrics["runtime"]["elapsed_seconds"]),
        "peak_gpu_memory_mb": float(
            max(
                image_metrics["runtime"].get("peak_gpu_memory_mb", 0.0),
                video_metrics["runtime"].get("peak_gpu_memory_mb", 0.0),
            )
        ),
    }


def _build_root_metrics(
    *,
    prepared: PreparedDavis,
    sequences: tuple[str, ...],
    image_metrics: dict[str, Any],
    video_metrics: dict[str, Any],
    vos_metrics: dict[str, Any],
    runtime_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": PHASE2_NAME,
        "dataset": "DAVIS 2017 TrainVal 480p",
        "davis_root": repo_relative(prepared.davis_root),
        "prepared_manifest": repo_relative(prepared.manifest_path),
        "prepared_structure_fingerprint": prepared.manifest.get("structure_fingerprint"),
        "selected_sequences": list(sequences),
        "model_name": image_metrics["model_name"],
        "feature_mode": image_metrics["feature_mode"],
        "image_dense_pca": image_metrics,
        "video_dense_pca": video_metrics,
        "vos_label_propagation": vos_metrics,
        "runtime_summary": runtime_summary,
    }


def _frame_list_label(values: list[int] | list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(str(value) for value in values)


def _sequence_note(sequence: str, *, video_metric: dict[str, Any], vos_metric: dict[str, Any]) -> str:
    aggregate = vos_metric["aggregate"]
    mean_j = aggregate["sampled_crop_j_mean"]
    min_j = aggregate.get("sampled_crop_j_min", aggregate.get("sampled_crop_iou_min"))
    mean_cosine = video_metric["mean_adjacent_patch_cosine"]
    low_iou_frames = aggregate["low_iou_frames"]
    if sequence == "dog":
        return (
            f"`dog` 是稳定动物目标序列，mean sampled-crop J={mean_j:.4f}，"
            f"min J={min_j:.4f}，相邻帧 dense feature cosine={mean_cosine:.4f}。"
            "当前结果显示一帧标注可以较稳定地传播到后续采样帧，是对象级时序一致性的正向证据。"
        )
    if sequence == "car-shadow":
        return (
            f"`car-shadow` 是街景刚体目标序列，mean sampled-crop J={mean_j:.4f}，"
            f"min J={min_j:.4f}，相邻帧 dense feature cosine={mean_cosine:.4f}。"
            "传播整体稳定，说明该固定 dense baseline 对导航/街景类刚体目标有可用的局部表征连续性。"
        )
    if sequence == "parkour":
        if low_iou_frames:
            boundary = f"低于 0.6 阈值的采样帧为 {_frame_list_label(low_iou_frames)}"
        else:
            boundary = "本次采样帧未低于 0.6 阈值"
        return (
            f"`parkour` 是快速非刚体人体运动序列，mean sampled-crop J={mean_j:.4f}，"
            f"min J={min_j:.4f}，相邻帧 dense feature cosine={mean_cosine:.4f}，{boundary}。"
            "它清楚暴露了当前方法对大姿态变化、尺度变化和快速运动的失败边界。"
        )
    return (
        f"`{sequence}`：mean sampled-crop J={mean_j:.4f}，min J={min_j:.4f}，"
        f"相邻帧 dense feature cosine={mean_cosine:.4f}。"
    )


def _csv_note(sequence: str, low_iou_frames: list[int]) -> str:
    if sequence == "dog":
        return "stable animal target; strong propagation; object-level temporal consistency"
    if sequence == "car-shadow":
        return "street rigid object; stable propagation; useful street-scene dense features"
    if sequence == "parkour":
        if low_iou_frames:
            return "fast non-rigid human motion; failure boundary under pose/scale/fast motion"
        return "fast non-rigid human motion; inspect for pose/scale sensitivity"
    return "custom selected sequence"


def _write_summary_csv(path: Path, *, metrics: dict[str, Any]) -> None:
    video = metrics["video_dense_pca"]["sequence_metrics"]
    vos = metrics["vos_label_propagation"]["sequence_metrics"]
    fieldnames = [
        "sequence",
        "mean_adjacent_cosine",
        "mean_pca_color_drift",
        "sampled_crop_j_mean",
        "sampled_crop_j_min",
        "low_iou_frames",
        "num_sampled_vos_frames",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sequence in metrics["selected_sequences"]:
            video_item = video[sequence]
            aggregate = vos[sequence]["aggregate"]
            low_iou_frames = aggregate["low_iou_frames"]
            writer.writerow(
                {
                    "sequence": sequence,
                    "mean_adjacent_cosine": f"{video_item['mean_adjacent_patch_cosine']:.6f}",
                    "mean_pca_color_drift": f"{video_item['mean_temporal_pca_color_drift_proxy']:.6f}",
                    "sampled_crop_j_mean": f"{aggregate['sampled_crop_j_mean']:.6f}",
                    "sampled_crop_j_min": f"{aggregate.get('sampled_crop_j_min', aggregate['sampled_crop_iou_min']):.6f}",
                    "low_iou_frames": _frame_list_label(low_iou_frames),
                    "num_sampled_vos_frames": aggregate["evaluated_frame_count_excluding_source"],
                    "notes": _csv_note(sequence, low_iou_frames),
                }
            )


def _write_report(path: Path, *, metrics: dict[str, Any], output_files: list[str]) -> None:
    video = metrics["video_dense_pca"]
    vos = metrics["vos_label_propagation"]
    low_iou_threshold = vos.get("low_iou_threshold", 0.6)
    lines = [
        "# Phase 2：V-JEPA 2.1 稠密特征 DAVIS 验证报告",
        "",
        "## 实验目的",
        "",
        "本实验用于快速检查 V-JEPA 2.1 ViT-B 在 DAVIS 2017 视频帧上的稠密 patch 特征是否具有可视化结构和基本时序一致性。"
        "当前 Phase 2 是固定 dense-feature baseline，不新增模型尺度、输入分辨率或效率实验。",
        "",
        "## 实验设置",
        "",
        f"- 数据集：DAVIS 2017 TrainVal 480p，直接读取 `JPEGImages/480p` 与 `Annotations/480p`。",
        f"- 序列：{', '.join(metrics['selected_sequences'])}。",
        f"- 模型：{metrics['model_name']}。",
        f"- 特征模式：`{metrics['feature_mode']}`，即逐帧 image-tokenizer dense features。",
        "- 输入对齐：固定 384x384 resize + center crop，RGB 图像与 mask 使用同一裁剪策略。",
        "- VOS 前景定义：DAVIS multi-instance mask 统一使用 `mask > 0` 合并为二值 foreground。",
        "- 边界说明：PCA 图是特征可视化，不是语义分割；VOS 指标是采样 aligned-crop IoU/J，不是官方 DAVIS J&F。",
        "- 结果不能解读为 zero-shot semantic segmentation、depth estimation 或官方 VOS benchmark 复现。",
        "",
        "## 单图 Dense PCA 结果",
        "",
        "单图阶段使用每个默认序列的 `00000.jpg`，在所有首帧 patch feature 上拟合一个共享 PCA，前三个主成分映射为 RGB。"
        "这些颜色只表示 V-JEPA dense feature 的主变化方向，不能当作类别或实例 mask。",
        "",
        f"- feature grid shape：{metrics['image_dense_pca']['feature_grid_shape']}。",
        f"- feature dimension：{metrics['image_dense_pca']['feature_dimension']}。",
        f"- PCA explained variance ratio：{metrics['image_dense_pca']['pca']['explained_variance_ratio'][:3]}。",
        "",
        "## 视频 Dense PCA 时序一致性结果",
        "",
        "视频阶段每个序列均匀采样 8 帧，并在该序列全部采样帧 patch feature 上拟合一次 PCA。"
        "因此同一序列内 PCA 颜色可作时序对比，跨序列颜色不直接可比。",
        "",
        "| sequence | mean adjacent feature cosine | mean PCA color drift | sampled frames |",
        "| --- | ---: | ---: | --- |",
    ]
    for sequence, item in video["sequence_metrics"].items():
        lines.append(
            f"| `{sequence}` | {item['mean_adjacent_patch_cosine']:.4f} | "
            f"{item['mean_temporal_pca_color_drift_proxy']:.4f} | "
            f"{_frame_list_label(item['sampled_frame_indices'])} |"
        )

    lines.extend(
        [
            "",
            "## VOS Label Propagation 结果",
            "",
            "VOS 阶段使用第一帧 `00000.png` 作为二值 foreground 标注，并通过 patch feature cosine similarity 做简单 top-k label propagation。"
            "这里的 `sampled_crop_j_mean` 和 `sampled_crop_j_min` 是采样帧、对齐裁剪后的 IoU/J；source frame 不参与聚合指标。"
            f"`low_iou_frames` 使用 sampled/cropped J < {low_iou_threshold:.1f} 判定。",
            "",
            "| sequence | sampled_crop_j_mean | sampled_crop_j_min | low_iou_frames | evaluated frames |",
            "| --- | ---: | ---: | --- | ---: |",
        ]
    )
    for sequence, item in vos["sequence_metrics"].items():
        aggregate = item["aggregate"]
        low_iou = _frame_list_label(aggregate["low_iou_frames"])
        lines.append(
            f"| `{sequence}` | {aggregate['sampled_crop_j_mean']:.4f} | "
            f"{aggregate.get('sampled_crop_j_min', aggregate['sampled_crop_iou_min']):.4f} | "
            f"{low_iou} | {aggregate['evaluated_frame_count_excluding_source']} |"
        )
    lines.extend(
        [
            "",
            f"所有序列平均 sampled-crop J mean：{vos['mean_sampled_crop_j_mean']:.4f}。",
            "注意：per-frame `mean_confidence` 仍保留在 JSON 中作为调试值，但当前不把 low-confidence frame list 作为主结论，"
            "因为这个绝对置信度在现有实现中区分力不足。",
            "",
            "## 序列级结论",
            "",
        ]
    )
    for sequence in metrics["selected_sequences"]:
        lines.append(
            "- "
            + _sequence_note(
                sequence,
                video_metric=video["sequence_metrics"][sequence],
                vos_metric=vos["sequence_metrics"][sequence],
            )
        )
    lines.extend(
        [
            "",
            "## 当前限制",
            "",
            "- 当前只评估 DAVIS 2017 TrainVal 480p、ViT-B、384x384、image-tokenizer-framewise 这一固定设置。",
            "- 当前 VOS 是简单 first-frame binary foreground propagation，不处理官方 multi-object DAVIS 协议。",
            "- 当前指标不是官方 DAVIS J&F，不能与论文或排行榜分数直接比较。",
            "- 当前结果不覆盖输入分辨率、运行时间、显存、吞吐、边缘设备可行性或 video-tokenizer 模式。",
            "",
            "## Phase 3 后续计划",
            "",
            "Phase 3 将单独评估不同输入分辨率、计算成本、显存占用、推理时间和 edge-computing 约束下的可行性。"
            "这些问题不在当前 Phase 2 范围内，避免把表征验证和系统效率分析混在一起。",
            "",
            "## 输出文件",
            "",
        ]
    )
    for output_file in output_files:
        lines.append(f"- `{output_file}`")
    lines.extend(
        [
            "",
            "## 运行时间",
            "",
            f"- prepare：{_seconds(metrics['runtime_summary']['prepare_seconds'])}",
            f"- model load：{_seconds(metrics['runtime_summary']['model_load_seconds'])}",
            f"- image dense PCA：{_seconds(metrics['runtime_summary']['image_dense_pca_seconds'])}",
            f"- video dense PCA：{_seconds(metrics['runtime_summary']['video_dense_pca_seconds'])}",
            f"- VOS label propagation：{_seconds(metrics['runtime_summary']['vos_label_propagation_seconds'])}",
            f"- end to end：{_seconds(metrics['runtime_summary']['end_to_end_total_seconds'])}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    path: Path,
    *,
    prepared: PreparedDavis,
    sequences: tuple[str, ...],
    environment: dict[str, Any],
    runtime_summary: dict[str, Any],
    output_files: list[str],
    model_name: str,
    feature_mode: str,
    atomic: bool,
) -> None:
    from research.scripts.common.runtime import (
        REPO_ROOT,
        collect_git_provenance,
        now_utc,
        sha256_bytes,
        sha256_file,
        write_json,
        write_json_atomic,
    )

    payload = {
        "phase_name": PHASE2_NAME,
        "timestamp": now_utc(),
        **collect_git_provenance(),
        **_collect_phase2_source_provenance(
            repo_root=REPO_ROOT,
            sha256_file=sha256_file,
            sha256_bytes=sha256_bytes,
        ),
        "model_name": model_name,
        "feature_mode": feature_mode,
        "davis_root": repo_relative(prepared.davis_root),
        "prepared_dataset_manifest": repo_relative(prepared.manifest_path),
        "prepared_structure_fingerprint": prepared.manifest.get("structure_fingerprint"),
        "selected_sequences": list(sequences),
        "device": environment["device"],
        "gpu_name": environment["gpu_name"],
        "pytorch_version": environment["pytorch_version"],
        "cuda_version": environment["torch_cuda_version"],
        "runtime_summary": runtime_summary,
        "output_file_list": output_files,
    }
    if atomic:
        write_json_atomic(path, payload)
    else:
        write_json(path, payload)


def run_phase2_dense(
    *,
    prepared: PreparedDavis,
    sequences: tuple[str, ...],
    prepare_seconds: float,
    phase_start_time: float,
) -> None:
    import torch

    from research.scripts.common.dense_pca import load_phase2_encoder
    from research.scripts.common.runtime import runtime_environment, write_json
    from research.scripts.common.video_models import release_cuda
    from research.scripts.common.visualization import save_phase2_overview

    device, gpu_name = _require_phase2_cuda(torch)
    environment = runtime_environment("cuda:0", gpu_name)
    print(f"[Setup] Device: cuda:0 | GPU: {gpu_name} | CUDA: {torch.version.cuda}")

    output_dir = _prepare_staged_output()
    encoder = None
    try:
        print()
        print("[Setup] Loading V-JEPA 2.1 ViT-B encoder")
        model_start = time.perf_counter()
        encoder = load_phase2_encoder(device)
        model_load_seconds = time.perf_counter() - model_start
        print(f"      model load elapsed: {_seconds(model_load_seconds)}")

        print()
        print("[1/4] Image dense PCA")
        image_result = _run_image_dense_pca(
            output_dir=output_dir / "image_dense_pca",
            prepared=prepared,
            sequences=sequences,
            encoder=encoder,
            device=device,
            write_json=write_json,
            relative_to_repo=repo_relative,
        )
        print(
            "      feature grid: "
            f"{image_result['metrics']['feature_grid_shape']} x {image_result['metrics']['feature_dimension']}"
        )
        print(f"      elapsed: {_seconds(image_result['metrics']['runtime']['elapsed_seconds'])}")

        print()
        print("[2/4] Video dense PCA temporal consistency")
        video_result = _run_video_dense_pca(
            output_dir=output_dir / "video_dense_pca",
            prepared=prepared,
            sequences=sequences,
            encoder=encoder,
            device=device,
            write_json=write_json,
            relative_to_repo=repo_relative,
        )
        for sequence, item in video_result["metrics"]["sequence_metrics"].items():
            print(f"      {sequence}: mean adjacent cosine={item['mean_adjacent_patch_cosine']:.4f}")
        print(f"      elapsed: {_seconds(video_result['metrics']['runtime']['elapsed_seconds'])}")

        print()
        print("[3/4] VOS label propagation")
        vos_result = _run_vos_label_propagation(
            output_dir=output_dir / "vos_label_propagation",
            prepared=prepared,
            sequences=sequences,
            feature_batches=video_result["feature_batches"],
            write_json=write_json,
        )
        for sequence, item in vos_result["metrics"]["sequence_metrics"].items():
            print(f"      {sequence}: sampled crop J={item['aggregate']['sampled_crop_j_mean']:.4f}")
        print(f"      elapsed: {_seconds(vos_result['metrics']['runtime']['elapsed_seconds'])}")

        print()
        print("[4/4] Writing Phase 2 report outputs")
        output_files = sorted(EXPECTED_OUTPUT_FILES)
        runtime_summary = _runtime_summary(
            phase_start_time=phase_start_time,
            prepare_seconds=prepare_seconds,
            model_load_seconds=model_load_seconds,
            image_metrics=image_result["metrics"],
            video_metrics=video_result["metrics"],
            vos_metrics=vos_result["metrics"],
        )
        root_metrics = _build_root_metrics(
            prepared=prepared,
            sequences=sequences,
            image_metrics=image_result["metrics"],
            video_metrics=video_result["metrics"],
            vos_metrics=vos_result["metrics"],
            runtime_summary=runtime_summary,
        )
        save_phase2_overview(
            output_dir / "phase2_dense_summary.png",
            image_summary_path=output_dir / "image_dense_pca" / "image_pca_summary.png",
            video_summary_path=output_dir / "video_dense_pca" / "video_pca_summary.png",
            vos_summary_path=output_dir / "vos_label_propagation" / "vos_summary.png",
            metrics=root_metrics,
        )
        _write_summary_csv(output_dir / "summary.csv", metrics=root_metrics)
        write_json(output_dir / "metrics.json", root_metrics)
        _write_report(output_dir / "report.md", metrics=root_metrics, output_files=output_files)
        _write_manifest(
            output_dir / "manifest.json",
            prepared=prepared,
            sequences=sequences,
            environment=environment,
            runtime_summary=runtime_summary,
            output_files=output_files,
            model_name=root_metrics["model_name"],
            feature_mode=root_metrics["feature_mode"],
            atomic=False,
        )
        _verify_phase2_output(output_dir)
        _replace_final_output(output_dir)

        final_runtime_summary = _runtime_summary(
            phase_start_time=phase_start_time,
            prepare_seconds=prepare_seconds,
            model_load_seconds=model_load_seconds,
            image_metrics=image_result["metrics"],
            video_metrics=video_result["metrics"],
            vos_metrics=vos_result["metrics"],
        )
        root_metrics["runtime_summary"] = final_runtime_summary
        _write_summary_csv(PHASE2_OUTPUT_DIR / "summary.csv", metrics=root_metrics)
        write_json(PHASE2_OUTPUT_DIR / "metrics.json", root_metrics)
        _write_report(PHASE2_OUTPUT_DIR / "report.md", metrics=root_metrics, output_files=output_files)
        _write_manifest(
            PHASE2_OUTPUT_DIR / "manifest.json",
            prepared=prepared,
            sequences=sequences,
            environment=environment,
            runtime_summary=final_runtime_summary,
            output_files=output_files,
            model_name=root_metrics["model_name"],
            feature_mode=root_metrics["feature_mode"],
            atomic=True,
        )
        _verify_phase2_output(PHASE2_OUTPUT_DIR)
        print("[Done] Phase 2 outputs:")
        print(f"       {repo_relative(PHASE2_OUTPUT_DIR)}/")
    except Exception:
        _cleanup_staged_output(output_dir)
        raise
    finally:
        if encoder is not None:
            del encoder
        release_cuda(device=device)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sequences = _sequence_tuple(args.sequences)
    phase_start_time = time.perf_counter()

    print("[Phase 2 / Dense DAVIS] V-JEPA 2.1 framewise dense baseline")
    print("[0/4] Preparing DAVIS dataset")
    prepare_start = time.perf_counter()
    try:
        prepared = prepare_davis_dataset(
            manifest_path=PREPARED_MANIFEST_PATH,
            sequences=sequences,
            force=bool(args.force_prepare),
        )
    except FileNotFoundError as error:
        print(str(error))
        raise
    prepare_seconds = time.perf_counter() - prepare_start
    status = "reused" if prepared.skipped else "wrote"
    print(f"      prepared manifest {status}: {repo_relative(prepared.manifest_path)}")
    for sequence in sequences:
        record = prepared.sequences[sequence]
        print(f"      {sequence}: {record.frame_count} frames, {record.mask_count} masks")
    print(f"      elapsed: {_seconds(prepare_seconds)}")

    if args.prepare_only:
        print("[Done] Prepare-only check complete.")
        print("       Full Phase 2 run command: python research/scripts/run_phase2_dense.py")
        return

    run_phase2_dense(
        prepared=prepared,
        sequences=sequences,
        prepare_seconds=prepare_seconds,
        phase_start_time=phase_start_time,
    )


if __name__ == "__main__":
    main()
