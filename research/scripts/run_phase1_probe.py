from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from research.scripts.common.analysis import (
    analyse_completion,
    analyse_correspondence,
    analyse_representation,
    compact_mode_metric,
)
from research.scripts.common.runtime import (
    EXPECTED_OUTPUT_FILES,
    PhaseConfig,
    build_parser,
    cleanup_staged_output,
    collect_phase_source_provenance,
    now_utc,
    prepare_staged_output,
    relative_to_repo,
    replace_final_output,
    require_cuda,
    require_sample_video,
    runtime_environment,
    sha256_file,
    verify_compact_output,
    write_json,
    write_json_atomic,
)
from research.scripts.common.video_models import (
    MODEL_NAME,
    PREDICTOR_MODEL_NAME,
    TEACHER_MODEL_NAME,
    VJEPA21_PREPROCESSOR,
    VJEPA21_VITB_MODEL,
    VJEPA21_VITG_TEACHER_MODEL,
    extract_dense_features,
    save_feature_npz,
)
from research.scripts.common.visualization import (
    save_completion_figure,
    save_correspondence_figure,
    save_representation_figure,
)


def _seconds(value: float) -> str:
    return f"{value:.2f}s"


def _mode_line(mode: str, metric: dict) -> str:
    return (
        f"{mode}: cos={metric['mean_target_cosine']:.4f}, "
        f"nonmatch={metric['mean_nonmatching_cosine']:.4f}, "
        f"top5={metric['top5']:.4f}, MRR={metric['mrr']:.4f}"
    )


def _write_metrics(
    path: Path,
    *,
    phase_config: PhaseConfig,
    representation,
    correspondence,
    completion,
    dense_runtime: dict,
) -> dict:
    completion_metrics = {
        "target_block": {
            "token": phase_config.completion.target_token,
            "row": phase_config.completion.target_row,
            "col": phase_config.completion.target_col,
            "height": phase_config.completion.target_height,
            "width": phase_config.completion.target_width,
        },
        "runtime": completion.runtime,
    }
    for mode in phase_config.completion.modes:
        metric = compact_mode_metric(completion.mode_metrics[mode])
        metric["teacher_forward_seconds"] = completion.teacher_runtime["teacher_forward_seconds"]
        metric["teacher_peak_gpu_memory_mb"] = completion.teacher_runtime["teacher_peak_gpu_memory_mb"]
        completion_metrics[mode] = metric

    metrics = {
        "phase": phase_config.phase_name,
        "representation": {
            **representation.metrics,
            "patch_tokens_shape": list(dense_runtime["patch_token_shape"]),
            "temporal_features_shape": list(dense_runtime["temporal_features_shape"]),
            "video_feature_shape": list(dense_runtime["video_feature_shape"]),
            "temporal_similarity_shape": list(dense_runtime["temporal_similarity_shape"]),
            "dense_feature_runtime": dense_runtime,
        },
        "correspondence": correspondence.metrics,
        "completion": completion_metrics,
    }
    write_json(path, metrics)
    return metrics


def _write_report(
    path: Path,
    *,
    phase_config: PhaseConfig,
    video_path: Path,
    environment: dict,
    metrics: dict,
    output_files: list[str],
    runtime_summary: dict,
) -> None:
    completion = metrics["completion"]
    lines = [
        "# Phase 1 - V-JEPA 2.1 核心能力探测",
        "",
        f"- 阶段名：`{phase_config.phase_name}`",
        f"- 输入视频：`{relative_to_repo(video_path)}`",
        f"- 模型：{MODEL_NAME}；{TEACHER_MODEL_NAME}；{PREDICTOR_MODEL_NAME}",
        f"- 设备：{environment['device']}；GPU：{environment['gpu_name']}",
        f"- PyTorch：{environment['pytorch_version']}；CUDA：{environment['torch_cuda_version']}",
        "",
        "## Representation",
        "",
        (
            "V-JEPA dense token 在该样例中保持较高相邻时间一致性："
            f"mean consecutive temporal similarity = "
            f"{metrics['representation']['mean_consecutive_temporal_similarity']:.4f}。"
        ),
        "token 12-19 的 PCA 只用于 latent token 伪彩色可视化，不是 RGB 重建。",
        (
            "local latent change 定义为同一空间位置 "
            "`1 - cosine(F_t(h,w), F_(t+1)(h,w))`，不是光流、像素差或对象跟踪。"
        ),
        "",
        "## Correspondence",
        "",
        (
            f"reference ROI = {metrics['correspondence']['reference_roi']}，"
            f"reference token = {metrics['correspondence']['reference_token']}，"
            f"reference patches = {metrics['correspondence']['reference_patch_count']}。"
        ),
        (
            "红色轮廓是由 patch latent similarity、memory bank、spatial prior 与连通区域约束"
            "得到的粗粒度对象候选区域；patch centroid 只是 patch-level 重心，不是像素级分割或可靠物理轨迹。"
        ),
        (
            "聚合 uncertainty flags："
            f"{metrics['correspondence']['aggregate_uncertainty_flags'] or ['none']}。"
        ),
        "",
        "## Completion",
        "",
        "四种上下文模式的核心指标：",
        "",
        "| mode | mean cosine | non-match cosine | Top-1 | Top-5 | MRR | visible tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in phase_config.completion.modes:
        metric = completion[mode]
        lines.append(
            f"| `{mode}` | {metric['mean_target_cosine']:.4f} | "
            f"{metric['mean_nonmatching_cosine']:.4f} | {metric['top1']:.4f} | "
            f"{metric['top5']:.4f} | {metric['mrr']:.4f} | "
            f"{metric['visible_context_token_count']} |"
        )
    lines.extend(
        [
            "",
            "这是 masked latent completion 诊断，不是 RGB 视频生成。`full` 不是纯未来预测；"
            "`past_only` 是因果式诊断，不证明严格因果世界模型。",
            "",
            "## 输出文件",
            "",
        ]
    )
    for name in output_files:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "## 运行时间",
            "",
            f"- dense feature extraction：{_seconds(runtime_summary['dense_feature_extraction_seconds'])}",
            f"- representation analysis：{_seconds(runtime_summary['representation_seconds'])}",
            f"- correspondence analysis：{_seconds(runtime_summary['correspondence_seconds'])}",
            f"- completion analysis：{_seconds(runtime_summary['completion_seconds'])}",
            f"- setup and output overhead：{_seconds(runtime_summary['setup_and_output_seconds'])}",
            f"- end-to-end total：{_seconds(runtime_summary['end_to_end_total_seconds'])}",
            "",
            "端到端总耗时包含固定输入和 CUDA 环境检查、模型加载、图像保存、"
            "JSON/Markdown 写入、GPU 清理和 staged-output 发布等开销；"
            "四个核心分析阶段耗时不需要与端到端总耗时完全相加一致。",
            "",
            "## 能力边界",
            "",
            "本阶段是能力探测，不是论文 benchmark 复现，不证明视频对象分割、RGB 生成、控制或规划能力。"
            "24x24 patch grid 只提供 patch-level 空间分辨率；遮挡、模糊和手-物体接触会增加对应与传播不确定性。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    path: Path,
    *,
    phase_config: PhaseConfig,
    video_path: Path,
    environment: dict,
    output_files: list[str],
    runtime_summary: dict,
    atomic: bool = False,
) -> None:
    provenance = collect_phase_source_provenance()
    manifest = {
        "phase_name": phase_config.phase_name,
        "timestamp": now_utc(),
        **provenance,
        "source_video_path": relative_to_repo(video_path),
        "source_video_hash": sha256_file(video_path),
        "device": environment["device"],
        "gpu_name": environment["gpu_name"],
        "pytorch_version": environment["pytorch_version"],
        "cuda_version": environment["torch_cuda_version"],
        "model_names": {
            "preprocessor": VJEPA21_PREPROCESSOR,
            "vit_b_encoder_and_predictor": VJEPA21_VITB_MODEL,
            "vit_g_teacher": VJEPA21_VITG_TEACHER_MODEL,
        },
        "fixed_phase_configuration": phase_config,
        "output_file_list": output_files,
        "runtime_summary": runtime_summary,
    }
    if atomic:
        write_json_atomic(path, manifest)
    else:
        write_json(path, manifest)


def _build_runtime_summary(
    *,
    end_to_end_total_seconds: float,
    dense_runtime: dict,
    representation,
    correspondence,
    completion,
    phase_config: PhaseConfig,
) -> dict:
    dense_seconds = float(dense_runtime["elapsed_seconds"])
    representation_seconds = float(representation.runtime["elapsed_seconds"])
    correspondence_seconds = float(correspondence.runtime["elapsed_seconds"])
    completion_seconds = float(completion.runtime["elapsed_seconds"])
    core_stage_seconds = (
        dense_seconds
        + representation_seconds
        + correspondence_seconds
        + completion_seconds
    )
    return {
        "end_to_end_total_seconds": float(end_to_end_total_seconds),
        "setup_and_output_seconds": float(max(0.0, end_to_end_total_seconds - core_stage_seconds)),
        "dense_feature_extraction_seconds": dense_seconds,
        "representation_seconds": representation_seconds,
        "correspondence_seconds": correspondence_seconds,
        "completion_seconds": completion_seconds,
        "dense_feature_peak_gpu_memory_mb": dense_runtime["peak_gpu_memory_mb"],
        "teacher_peak_gpu_memory_mb": completion.teacher_runtime["teacher_peak_gpu_memory_mb"],
        "student_predictor_peak_gpu_memory_mb": {
            mode: completion.mode_metrics[mode]["peak_gpu_mb"]
            for mode in phase_config.completion.modes
        },
    }


def run_phase1_probe(
    config: PhaseConfig,
    *,
    device: torch.device,
    gpu_name: str,
    phase_start_time: float,
    environment: dict,
) -> None:
    output_dir = prepare_staged_output(config.staged_output_dir)

    try:
        print()
        print("[1/4] Dense feature extraction")
        dense = extract_dense_features(
            video_path=config.video_path,
            device=device,
            phase_name=config.phase_name,
            num_frames=config.representation.num_frames,
            crop_size=config.representation.crop_size,
        )
        save_feature_npz(
            output_dir / "features.npz",
            sampled_frame_indices=dense.sampled_frame_indices,
            fps=dense.fps,
            patch_tokens=dense.patch_tokens,
            temporal_features=dense.temporal_features,
            video_feature=dense.video_feature,
            temporal_similarity=dense.temporal_similarity,
            metadata=dense.metadata,
        )
        print(f"      patch tokens: {tuple(dense.patch_tokens.shape)}")
        print(
            "      mean consecutive temporal similarity: "
            f"{dense.runtime['mean_consecutive_temporal_similarity']:.4f}"
        )
        print(f"      elapsed: {_seconds(dense.runtime['elapsed_seconds'])}")
        print(f"      peak GPU memory: {dense.runtime['peak_gpu_memory_mb']:.1f} MB")

        print()
        print("[2/4] Representation analysis")
        representation = analyse_representation(dense, config.representation)
        save_representation_figure(
            output_dir / "representation.png",
            model_frames=dense.model_frames,
            sampled_frame_indices=dense.sampled_frame_indices,
            fps=dense.fps,
            temporal_similarity=dense.temporal_similarity,
            consecutive_similarity=dense.consecutive_similarity,
            representation=representation,
        )
        top_changes = ", ".join(
            f"{item['token']}->{item['next_token']}:{item['similarity']:.4f}"
            for item in representation.top_temporal_changes
        )
        print(f"      selected token range: {config.representation.start_token}-{config.representation.end_token}")
        print(f"      top temporal changes: {top_changes}")
        print(f"      elapsed: {_seconds(representation.runtime['elapsed_seconds'])}")

        print()
        print("[3/4] Correspondence analysis")
        correspondence = analyse_correspondence(
            representation.token_grid,
            dense,
            config.correspondence,
        )
        save_correspondence_figure(
            output_dir / "correspondence.png",
            model_frames=dense.model_frames,
            sampled_frame_indices=dense.sampled_frame_indices,
            fps=dense.fps,
            correspondence=correspondence,
            roi=config.correspondence.roi,
        )
        aggregate_flags = correspondence.metrics["aggregate_uncertainty_flags"] or ["none"]
        print(f"      reference patches: {correspondence.metrics['reference_patch_count']}")
        print(
            "      selected tokens: "
            + ", ".join(str(token) for token in config.correspondence.selected_tokens)
        )
        print(f"      uncertainty flags: {aggregate_flags}")
        print(f"      elapsed: {_seconds(correspondence.runtime['elapsed_seconds'])}")

        print()
        print("[4/4] Latent completion analysis")
        completion = analyse_completion(features=dense, device=device, config=config.completion)
        target_frame = dense.model_frames[2 * config.completion.target_token]
        save_completion_figure(
            output_dir / "completion.png",
            frame=target_frame,
            fps=dense.fps,
            completion=completion,
            mode_order=config.completion.modes,
        )
        for mode in config.completion.modes:
            print(f"      {_mode_line(mode, completion.mode_metrics[mode])}")
        print(f"      elapsed: {_seconds(completion.runtime['elapsed_seconds'])}")

        print()
        print("[Done] Writing compact Phase 1 outputs")
        metrics = _write_metrics(
            output_dir / "metrics.json",
            phase_config=config,
            representation=representation,
            correspondence=correspondence,
            completion=completion,
            dense_runtime=dense.runtime,
        )
        output_files = sorted(EXPECTED_OUTPUT_FILES)
        provisional_runtime_summary = _build_runtime_summary(
            end_to_end_total_seconds=time.perf_counter() - phase_start_time,
            dense_runtime=dense.runtime,
            representation=representation,
            correspondence=correspondence,
            completion=completion,
            phase_config=config,
        )
        _write_report(
            output_dir / "report.md",
            phase_config=config,
            video_path=config.video_path,
            environment=environment,
            metrics=metrics,
            output_files=output_files,
            runtime_summary=provisional_runtime_summary,
        )
        _write_manifest(
            output_dir / "manifest.json",
            phase_config=config,
            video_path=config.video_path,
            environment=environment,
            output_files=output_files,
            runtime_summary=provisional_runtime_summary,
        )
        verify_compact_output(output_dir)
        replace_final_output(output_dir, config.final_output_dir)
        phase_publish_complete_time = time.perf_counter()
        final_runtime_summary = _build_runtime_summary(
            end_to_end_total_seconds=phase_publish_complete_time - phase_start_time,
            dense_runtime=dense.runtime,
            representation=representation,
            correspondence=correspondence,
            completion=completion,
            phase_config=config,
        )
        _write_report(
            config.final_output_dir / "report.md",
            phase_config=config,
            video_path=config.video_path,
            environment=environment,
            metrics=metrics,
            output_files=output_files,
            runtime_summary=final_runtime_summary,
        )
        _write_manifest(
            config.final_output_dir / "manifest.json",
            phase_config=config,
            video_path=config.video_path,
            environment=environment,
            output_files=output_files,
            runtime_summary=final_runtime_summary,
            atomic=True,
        )
        print("[Done] Phase 1 outputs:")
        print(f"       {relative_to_repo(config.final_output_dir)}/")
    except Exception:
        cleanup_staged_output(output_dir)
        raise


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    config = PhaseConfig()

    print("[Phase 1 / Probe] V-JEPA 2.1 capability probe")
    print("[Setup] Validating fixed input and CUDA environment")
    phase_start_time = time.perf_counter()
    video_path = require_sample_video(config.video_path)
    device, gpu_name = require_cuda()
    environment = runtime_environment("cuda:0", gpu_name)
    print(f"[Setup] Device: cuda:0 | GPU: {gpu_name} | CUDA: {torch.version.cuda}")
    if video_path != config.video_path:
        raise RuntimeError("Unexpected fixed video path resolution.")
    run_phase1_probe(
        config,
        device=device,
        gpu_name=gpu_name,
        phase_start_time=phase_start_time,
        environment=environment,
    )


if __name__ == "__main__":
    main()
