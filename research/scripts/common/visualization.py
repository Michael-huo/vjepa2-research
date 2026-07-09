"""Compact non-interactive figures for the Phase 1 probe."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from research.scripts.common.analysis import CompletionResult, CorrespondenceResult, RepresentationResult
from research.scripts.common.video_models import CROP_SIZE, GRID_HEIGHT, GRID_WIDTH, PATCH_SIZE


MODE_LABELS = {
    "full": "full",
    "spatial_only": "spatial only",
    "temporal_bi": "temporal bi",
    "past_only": "past only",
}


def frame_time(frame_index: int, fps: float) -> float:
    if fps <= 0:
        return 0.0
    return float(frame_index) / float(fps)


def time_label(frame_index: int, fps: float) -> str:
    return f"{frame_time(frame_index, fps):.2f}s"


def resize_patch_map(values: np.ndarray, size: int = CROP_SIZE, interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    return cv2.resize(values.astype(np.float32), (size, size), interpolation=interpolation)


def similarity_overlay(frame: np.ndarray, scores: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    scaled = np.clip((scores - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    heatmap = plt.get_cmap("turbo")(scaled)[..., :3]
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(0.52 * frame + 0.48 * heatmap, 0.0, 1.0)


def draw_mask_and_centroid(ax, mask: np.ndarray, centroid: np.ndarray, *, contour_color: str = "red") -> None:
    mask_up = resize_patch_map(mask.astype(np.float32))
    if mask.any():
        ax.contour(mask_up, levels=[0.5], colors=contour_color, linewidths=1.6)
    if np.isfinite(centroid).all():
        cx = centroid[1] * PATCH_SIZE + PATCH_SIZE / 2
        cy = centroid[0] * PATCH_SIZE + PATCH_SIZE / 2
        ax.plot(cx, cy, marker="x", color="lime", markersize=8, markeredgewidth=2)


def save_representation_figure(
    output_path: Path,
    *,
    model_frames: np.ndarray,
    sampled_frame_indices: np.ndarray,
    fps: float,
    temporal_similarity: np.ndarray,
    consecutive_similarity: np.ndarray,
    representation: RepresentationResult,
) -> None:
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    grid = fig.add_gridspec(4, 4, height_ratios=[1.1, 1.0, 1.0, 1.0])

    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(temporal_similarity, vmin=-1.0, vmax=1.0, cmap="viridis")
    ax.set_title("Temporal token cosine similarity")
    ax.set_xlabel("token")
    ax.set_ylabel("token")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="cosine")

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(np.arange(len(consecutive_similarity)), consecutive_similarity, marker="o", linewidth=1.3)
    ax.set_title("Adjacent temporal consistency")
    ax.set_xlabel("token t")
    ax.set_ylabel("cosine(t, t+1)")
    ax.grid(alpha=0.25)

    for col, item in enumerate(representation.top_temporal_changes[:2], start=2):
        ax = fig.add_subplot(grid[0, col])
        token = int(item["token"])
        frame_a = model_frames[2 * token]
        frame_b = model_frames[2 * (token + 1)]
        pair = np.concatenate([frame_a, frame_b], axis=1)
        ax.imshow(pair)
        ax.axis("off")
        ax.set_title(
            f"Temporal change rank {item['rank']}: {token}->{token + 1}\n"
            f"cos={item['similarity']:.4f}"
        )

    pca_offset = representation.selected_tokens[0] - 12
    for col, token in enumerate(representation.selected_tokens):
        frame_slot = 2 * token
        frame_index = int(sampled_frame_indices[frame_slot])
        ax = fig.add_subplot(grid[1, col])
        ax.imshow(model_frames[frame_slot])
        ax.axis("off")
        ax.set_title(f"token {token} | frame {frame_index} | {time_label(frame_index, fps)}")

        ax = fig.add_subplot(grid[2, col])
        pca_index = token - 12 + pca_offset
        if 0 <= pca_index < representation.pca_maps.shape[0]:
            pca_up = cv2.resize(
                representation.pca_maps[pca_index],
                (CROP_SIZE, CROP_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            ax.imshow(pca_up)
        ax.axis("off")
        ax.set_title("PCA pseudo-color latent features\nnot RGB reconstruction")

    for col, item in enumerate(representation.local_changes[:2]):
        ax = fig.add_subplot(grid[3, col * 2 : col * 2 + 2])
        token = int(item["token"])
        frame = model_frames[2 * (token + 1)]
        change_up = resize_patch_map(item["change_map"], interpolation=cv2.INTER_CUBIC)
        ax.imshow(frame)
        ax.imshow(change_up, cmap="jet", alpha=0.48)
        ax.axis("off")
        ax.set_title(
            f"Same-position latent state change {token}->{token + 1}\n"
            f"mean={item['mean_change']:.4f}, max={item['max_change']:.4f}; "
            "not optical flow or pixel difference"
        )

    fig.suptitle(
        "Phase 1 representation: dense local V-JEPA states and temporal consistency",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_correspondence_figure(
    output_path: Path,
    *,
    model_frames: np.ndarray,
    sampled_frame_indices: np.ndarray,
    fps: float,
    correspondence: CorrespondenceResult,
    roi: tuple[int, int, int, int],
) -> None:
    selected = correspondence.selected_indices
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, len(selected), height_ratios=[1.0, 1.0, 0.72])
    selected_scores = [correspondence.appearance_maps[index] for index in selected]
    all_scores = np.concatenate([scores.reshape(-1) for scores in selected_scores])
    vmin = float(np.percentile(all_scores, 2))
    vmax = float(np.percentile(all_scores, 98))

    trajectory = np.array(
        [[record["centroid_col"], record["centroid_row"]] for record in correspondence.records],
        dtype=np.float32,
    )
    token_to_record_index = {record["token"]: index for index, record in enumerate(correspondence.records)}

    for col, index in enumerate(selected):
        record = correspondence.records[index]
        token = int(record["token"])
        frame_slot = 2 * token
        frame_index = int(sampled_frame_indices[frame_slot])
        frame = model_frames[frame_slot]
        mask = correspondence.masks[index]
        centroid = correspondence.centroids[index]

        ax = fig.add_subplot(grid[0, col])
        ax.imshow(frame)
        if token == correspondence.metrics["reference_token"]:
            ax.add_patch(
                plt.Rectangle(
                    (roi[0], roi[1]),
                    roi[2],
                    roi[3],
                    fill=False,
                    edgecolor="yellow",
                    linewidth=2.2,
                    label="reference ROI",
                )
            )
        draw_mask_and_centroid(ax, mask, centroid)
        current_record_index = token_to_record_index[token]
        history = trajectory[: current_record_index + 1]
        ax.plot(
            history[:, 0] * PATCH_SIZE + PATCH_SIZE / 2,
            history[:, 1] * PATCH_SIZE + PATCH_SIZE / 2,
            color="cyan",
            linewidth=1.6,
            marker="o",
            markersize=3,
        )
        ax.axis("off")
        ax.set_title(f"token {token} | frame {frame_index} | {time_label(frame_index, fps)}")

        ax = fig.add_subplot(grid[1, col])
        appearance = correspondence.appearance_maps[index]
        im = ax.imshow(appearance, cmap="turbo", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.contour(mask.astype(np.float32), levels=[0.5], colors="white", linewidths=1.2)
        ax.plot(record["centroid_col"], record["centroid_row"], marker="x", color="white", markersize=8)
        ax.set_title("appearance response\nred/white: coarse candidate")
        ax.set_xlabel("patch column")
        ax.set_ylabel("patch row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(grid[2, col])
        ax.axis("off")
        flags = ", ".join(record["uncertainty_flags"]) if record["uncertainty_flags"] else "none"
        text = (
            f"token: {token}\n"
            f"mask patches: {record['mask_patch_count']}\n"
            f"mask mean similarity: {record['mask_mean_similarity']:.4f}\n"
            f"peak similarity: {record['peak_similarity']:.4f}\n"
            f"patch centroid: ({record['centroid_row']:.2f}, {record['centroid_col']:.2f})\n"
            f"uncertainty: {flags}"
        )
        ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle(
        "Phase 1 correspondence: patch-level object candidates, response diffusion, and trajectory uncertainty",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_completion_figure(
    output_path: Path,
    *,
    frame: np.ndarray,
    fps: float,
    completion: CompletionResult,
    mode_order: tuple[str, ...],
) -> None:
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.78])
    x0, y0, width_px, height_px = completion.target_rect

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(frame)
    ax.add_patch(
        plt.Rectangle((x0, y0), width_px, height_px, fill=False, edgecolor="red", linewidth=2.5)
    )
    ax.set_title(
        f"Masked latent target block\n"
        f"token 16 | frame {completion.target_frame_index} | "
        f"{time_label(completion.target_frame_index, fps)}"
    )
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 1])
    full_grid = completion.cosine_grids["full"]
    im = ax.imshow(full_grid, cmap="viridis", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_title("full context: per-patch latent cosine")
    ax.set_xlabel("target block patch column")
    ax.set_ylabel("target block patch row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine")

    ax = fig.add_subplot(grid[0, 2])
    retrieval = completion.retrieval_matrices["full"]
    matrix = ax.imshow(retrieval, cmap="viridis", vmin=-1.0, vmax=1.0, aspect="auto")
    n = retrieval.shape[0]
    ax.plot(np.arange(n), np.arange(n), color="white", linewidth=0.9, label="ideal correspondence reference")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("full context retrieval matrix")
    ax.set_xlabel("teacher target index")
    ax.set_ylabel("predicted target index")
    fig.colorbar(matrix, ax=ax, fraction=0.046, pad=0.04, label="cosine")

    ax = fig.add_subplot(grid[1, :2])
    ax.axis("off")
    header = "mode             mean cos  non-match  top1   top5   MRR    visible"
    lines = [header]
    for mode in mode_order:
        metric = completion.mode_metrics[mode]
        lines.append(
            f"{mode:16s} {metric['mean_target_cosine']:8.4f} "
            f"{metric['mean_nonmatching_cosine']:9.4f} "
            f"{metric['top1']:5.3f} {metric['top5']:6.3f} "
            f"{metric['mrr']:6.3f} {metric['visible_tokens']:8d}"
        )
    ax.text(0.01, 0.96, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    ax.set_title("Context ablation summary")

    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    notes = (
        "Interpretation\n"
        "- masked latent completion, not RGB generation\n"
        "- full uses spatial and temporal context\n"
        "- spatial_only isolates same-frame context\n"
        "- temporal_bi hides the whole current frame plane\n"
        "- past_only is a causal-style diagnostic,\n"
        "  not proof of a strict causal world model\n"
        "- white diagonal: ideal correspondence reference"
    )
    ax.text(0.02, 0.96, notes, va="top", ha="left", fontsize=10)

    fig.suptitle("Phase 1 completion: masked V-JEPA latent prediction and context ablation", fontsize=15)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _imshow_no_axes(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(np.clip(image, 0.0, 1.0))
    ax.set_title(title)
    ax.axis("off")


def _upsample_dense_rgb(pca_map: np.ndarray, output_size: int = CROP_SIZE) -> np.ndarray:
    return cv2.resize(
        pca_map.astype(np.float32),
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST,
    )


def _choose_display_indices(count: int, max_items: int = 4) -> list[int]:
    if count <= max_items:
        return list(range(count))
    return sorted({int(round(i * (count - 1) / (max_items - 1))) for i in range(max_items)})


def save_phase2_image_pca_summary(
    output_path: Path,
    *,
    records: list[dict[str, Any]],
) -> None:
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, len(records))
    for col, record in enumerate(records):
        _imshow_no_axes(
            fig.add_subplot(grid[0, col]),
            record["frame"],
            f"{record['sequence']} | frame {record['frame_index']:05d}",
        )
        _imshow_no_axes(
            fig.add_subplot(grid[1, col]),
            _upsample_dense_rgb(record["pca_map"]),
            "shared PCA RGB\nfeature visualization",
        )
    fig.suptitle("Phase 2 image dense PCA: first DAVIS frame per sequence", fontsize=15)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_phase2_video_pca_summary(
    output_path: Path,
    *,
    records: list[dict[str, Any]],
) -> None:
    max_frames = max(len(record["frame_indices"]) for record in records)
    fig = plt.figure(figsize=(max(16, max_frames * 2.2), len(records) * 4.2), constrained_layout=True)
    grid = fig.add_gridspec(len(records) * 2, max_frames)
    for row, record in enumerate(records):
        frames = record["frames"]
        pca_maps = record["pca_maps"]
        for col, frame_index in enumerate(record["frame_indices"]):
            _imshow_no_axes(
                fig.add_subplot(grid[row * 2, col]),
                frames[col],
                f"{record['sequence']}\n{frame_index:05d}.jpg",
            )
            _imshow_no_axes(
                fig.add_subplot(grid[row * 2 + 1, col]),
                _upsample_dense_rgb(pca_maps[col]),
                "sequence PCA",
            )
        for col in range(len(record["frame_indices"]), max_frames):
            fig.add_subplot(grid[row * 2, col]).axis("off")
            fig.add_subplot(grid[row * 2 + 1, col]).axis("off")
    fig.suptitle(
        "Phase 2 video dense PCA: one PCA fit per sequence, applied across sampled frames",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _mask_overlay(frame: np.ndarray, predicted: np.ndarray, gt: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    pred_color = np.zeros_like(frame)
    pred_color[..., 0] = 1.0
    gt_color = np.zeros_like(frame)
    gt_color[..., 1] = 1.0
    overlay = np.where(predicted[..., None], 0.62 * overlay + 0.38 * pred_color, overlay)
    overlay = np.where(gt[..., None], 0.78 * overlay + 0.22 * gt_color, overlay)
    return np.clip(overlay, 0.0, 1.0)


def save_phase2_vos_summary(
    output_path: Path,
    *,
    records: list[dict[str, Any]],
) -> None:
    display_count = 4
    fig = plt.figure(figsize=(display_count * 4.0, len(records) * 4.4), constrained_layout=True)
    grid = fig.add_gridspec(len(records), display_count)
    for row, record in enumerate(records):
        indices = _choose_display_indices(len(record["frame_indices"]), max_items=display_count)
        metrics_by_frame = {
            item["frame_index"]: item
            for item in record["metrics"]["per_frame"]
        }
        for col, sample_index in enumerate(indices):
            frame_index = int(record["frame_indices"][sample_index])
            metric = metrics_by_frame[frame_index]
            overlay = _mask_overlay(
                record["frames"][sample_index],
                record["predicted_masks"][sample_index],
                record["gt_masks"][sample_index],
            )
            title = (
                f"{record['sequence']} | {frame_index:05d}\n"
                f"J={metric['sampled_crop_j']:.3f}, conf={metric['mean_confidence']:.3f}"
            )
            if metric["source_frame"]:
                title += " | source"
            _imshow_no_axes(fig.add_subplot(grid[row, col]), overlay, title)
        for col in range(len(indices), display_count):
            fig.add_subplot(grid[row, col]).axis("off")
    fig.suptitle(
        "Phase 2 VOS: red predicted foreground, green DAVIS crop GT, yellow overlap",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_phase2_overview(
    output_path: Path,
    *,
    image_summary_path: Path,
    video_summary_path: Path,
    vos_summary_path: Path,
    metrics: dict[str, Any],
) -> None:
    summary_images = [
        ("Image dense PCA", plt.imread(image_summary_path)),
        ("Video dense PCA", plt.imread(video_summary_path)),
        ("VOS propagation", plt.imread(vos_summary_path)),
    ]
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, width_ratios=[1.2, 1.2, 0.9])
    for row, (title, image) in enumerate(summary_images):
        ax = fig.add_subplot(grid[row, :2])
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    ax = fig.add_subplot(grid[:, 2])
    ax.axis("off")
    vos = metrics["vos_label_propagation"]
    video = metrics["video_dense_pca"]
    lines = [
        "Phase 2 Dense DAVIS",
        "",
        f"Sequences: {', '.join(metrics['selected_sequences'])}",
        f"Feature mode: {metrics['feature_mode']}",
        "",
        "Video PCA adjacent cosine:",
    ]
    for sequence, item in video["sequence_metrics"].items():
        value = item["mean_adjacent_patch_cosine"]
        lines.append(f"- {sequence}: {value:.4f}")
    lines.extend(["", "VOS sampled crop J mean:"])
    for sequence, item in vos["sequence_metrics"].items():
        value = item["aggregate"]["sampled_crop_j_mean"]
        lines.append(f"- {sequence}: {value:.4f}")
    lines.extend(
        [
            "",
            f"Mean VOS sampled crop J: {vos['mean_sampled_crop_j_mean']:.4f}",
            "",
            "Notes:",
            "- PCA maps visualize dense features.",
            "- VOS merges all DAVIS instances.",
            "- Metrics are aligned-crop samples, not official DAVIS J&F.",
        ]
    )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
