"""Simple first-frame label propagation for Phase 2 dense DAVIS experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from research.scripts.common.dense_pca import FrameFeatureBatch, normalize_features, preprocess_mask
from research.scripts.common.phase2_data import DavisSequence, repo_relative


DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.2
LOW_IOU_THRESHOLD = 0.6


def mask_to_patch_labels(mask: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    grid_height, grid_width = grid_shape
    height, width = mask.shape
    if height % grid_height != 0 or width % grid_width != 0:
        raise ValueError(
            f"Mask shape {mask.shape} is not divisible by feature grid {grid_shape}."
        )
    patch_height = height // grid_height
    patch_width = width // grid_width
    pooled = mask.reshape(grid_height, patch_height, grid_width, patch_width).mean(axis=(1, 3))
    return pooled.astype(np.float32)


def patch_probabilities_to_mask(probabilities: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    grid_height, grid_width = probabilities.shape
    height, width = output_shape
    if height % grid_height != 0 or width % grid_width != 0:
        raise ValueError(
            f"Output shape {output_shape} is not divisible by probability grid {probabilities.shape}."
        )
    patch_height = height // grid_height
    patch_width = width // grid_width
    return np.repeat(np.repeat(probabilities >= 0.5, patch_height, axis=0), patch_width, axis=1)


def binary_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = predicted.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(predicted, target).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(predicted, target).sum()
    return float(intersection / union)


def probability_centroid(probabilities: np.ndarray) -> tuple[float, float]:
    rows, cols = np.indices(probabilities.shape)
    weights = probabilities.astype(np.float64)
    total = float(weights.sum())
    if total <= 1e-8:
        return (float("nan"), float("nan"))
    return (
        float((rows * weights).sum() / total),
        float((cols * weights).sum() / total),
    )


def topk_softmax(values: np.ndarray, *, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    scaled = values / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_values = np.exp(scaled)
    return exp_values / (exp_values.sum(axis=1, keepdims=True) + 1e-8)


def propagate_from_source(
    *,
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[np.ndarray, float]:
    source_flat = normalize_features(source_features).reshape(-1, source_features.shape[-1])
    target_flat = normalize_features(target_features).reshape(-1, target_features.shape[-1])
    labels_flat = source_labels.reshape(-1)
    k = min(max(int(top_k), 1), source_flat.shape[0])
    similarities = target_flat @ source_flat.T
    top_indices = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(similarities, top_indices, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_indices = np.take_along_axis(top_indices, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)
    weights = topk_softmax(top_scores, temperature=temperature)
    probabilities = np.sum(weights * labels_flat[top_indices], axis=1)
    mean_confidence = float(weights.max(axis=1).mean())
    return probabilities.reshape(source_labels.shape).astype(np.float32), mean_confidence


def _foreground_area_ratio(mask_or_prob: np.ndarray) -> float:
    return float(np.mean(mask_or_prob > 0.5))


def _record_flags(*, iou: float | None, area_ratio: float) -> list[str]:
    flags: list[str] = []
    if area_ratio < 0.005:
        flags.append("empty_or_tiny_foreground")
    if area_ratio > 0.80:
        flags.append("overexpanded_foreground")
    if iou is not None and iou < LOW_IOU_THRESHOLD:
        flags.append("low_sampled_crop_iou")
    return flags


def run_vos_for_sequence(
    *,
    sequence: DavisSequence,
    feature_batch: FrameFeatureBatch,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    if not feature_batch.frame_indices:
        raise ValueError("feature_batch has no sampled frames.")
    if feature_batch.frame_indices[0] != 0:
        raise ValueError("VOS propagation expects the source frame 00000 to be included first.")

    stage_start = time.perf_counter()
    output_shape = feature_batch.processed_resolution
    grid_shape = feature_batch.grid_shape
    masks: list[np.ndarray] = []
    mask_metadata: list[dict[str, Any]] = []
    mask_paths: list[Path] = []
    for frame_index in feature_batch.frame_indices:
        mask_path = sequence.mask_path_for_index(frame_index)
        mask, metadata = preprocess_mask(mask_path, crop_size=output_shape[0])
        masks.append(mask)
        mask_metadata.append(metadata)
        mask_paths.append(mask_path)

    source_patch_labels = mask_to_patch_labels(masks[0], grid_shape)
    source_centroid = probability_centroid(source_patch_labels)
    predicted_probabilities: list[np.ndarray] = []
    predicted_masks: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    for sample_index, frame_index in enumerate(feature_batch.frame_indices):
        if sample_index == 0:
            patch_probabilities = source_patch_labels.astype(np.float32)
            mean_confidence = 1.0
        else:
            patch_probabilities, mean_confidence = propagate_from_source(
                source_features=feature_batch.features[0],
                target_features=feature_batch.features[sample_index],
                source_labels=source_patch_labels,
                top_k=top_k,
                temperature=temperature,
            )
        predicted_mask = patch_probabilities_to_mask(patch_probabilities, output_shape)
        iou = binary_iou(predicted_mask, masks[sample_index])
        centroid = probability_centroid(patch_probabilities)
        if np.isfinite(source_centroid).all() and np.isfinite(centroid).all():
            centroid_drift = float(
                np.sqrt((centroid[0] - source_centroid[0]) ** 2 + (centroid[1] - source_centroid[1]) ** 2)
            )
        else:
            centroid_drift = float("nan")
        area_ratio = _foreground_area_ratio(patch_probabilities)
        flags = _record_flags(
            iou=None if sample_index == 0 else iou,
            area_ratio=area_ratio,
        )
        record = {
            "frame_index": int(frame_index),
            "frame_path": repo_relative(feature_batch.frame_paths[sample_index]),
            "mask_path": repo_relative(mask_paths[sample_index]),
            "source_frame": bool(sample_index == 0),
            "foreground_area_ratio": float(np.mean(masks[sample_index])),
            "propagated_foreground_area_ratio": area_ratio,
            "mean_confidence": float(mean_confidence),
            "centroid_row": float(centroid[0]),
            "centroid_col": float(centroid[1]),
            "centroid_drift_patches": centroid_drift,
            "sampled_crop_iou": float(iou),
            "sampled_crop_j": float(iou),
            "flags": flags,
        }
        predicted_probabilities.append(patch_probabilities)
        predicted_masks.append(predicted_mask)
        records.append(record)

    target_records = [record for record in records if not record["source_frame"]]
    target_ious = [record["sampled_crop_iou"] for record in target_records]
    aggregate = {
        "sampled_crop_iou_mean": float(np.mean(target_ious)) if target_ious else float("nan"),
        "sampled_crop_j_mean": float(np.mean(target_ious)) if target_ious else float("nan"),
        "sampled_crop_iou_min": float(np.min(target_ious)) if target_ious else float("nan"),
        "sampled_crop_j_min": float(np.min(target_ious)) if target_ious else float("nan"),
        "low_iou_threshold": float(LOW_IOU_THRESHOLD),
        "evaluated_frame_count_excluding_source": int(len(target_records)),
        "low_iou_frames": [
            record["frame_index"]
            for record in target_records
            if record["sampled_crop_j"] < LOW_IOU_THRESHOLD
        ],
    }
    metrics = {
        "sequence": sequence.name,
        "sampled_frame_indices": [int(index) for index in feature_batch.frame_indices],
        "grid_shape": list(grid_shape),
        "top_k": int(top_k),
        "temperature": float(temperature),
        "binary_foreground_definition": "DAVIS annotation mask > 0; all instances merged.",
        "source_foreground_patch_area_ratio": float(np.mean(source_patch_labels > 0.5)),
        "source_foreground_crop_area_ratio": float(np.mean(masks[0])),
        "per_frame": records,
        "aggregate": aggregate,
        "runtime": {"elapsed_seconds": float(time.perf_counter() - stage_start)},
    }
    return {
        "sequence": sequence.name,
        "frames": feature_batch.frames,
        "gt_masks": np.stack(masks, axis=0),
        "predicted_masks": np.stack(predicted_masks, axis=0),
        "predicted_patch_probabilities": np.stack(predicted_probabilities, axis=0),
        "records": records,
        "metrics": metrics,
        "mask_metadata": mask_metadata,
    }


def aggregate_vos_metrics(sequence_results: list[dict[str, Any]]) -> dict[str, Any]:
    sequence_metrics = {item["sequence"]: item["metrics"] for item in sequence_results}
    means = [
        item["metrics"]["aggregate"]["sampled_crop_j_mean"]
        for item in sequence_results
        if np.isfinite(item["metrics"]["aggregate"]["sampled_crop_j_mean"])
    ]
    low_confidence: dict[str, list[int]] = {}
    low_iou: dict[str, list[int]] = {}
    for item in sequence_results:
        aggregate = item["metrics"]["aggregate"]
        low_confidence[item["sequence"]] = [
            record["frame_index"]
            for record in item["metrics"]["per_frame"]
            if (not record["source_frame"]) and record["mean_confidence"] < 0.35
        ]
        low_iou[item["sequence"]] = aggregate["low_iou_frames"]
    return {
        "task": "first_frame_binary_foreground_label_propagation",
        "metric_scope": "sampled aligned 384x384 crop; not official DAVIS J&F",
        "binary_foreground_definition": "mask > 0, merging all DAVIS instances",
        "top_k": DEFAULT_TOP_K,
        "temperature": DEFAULT_TEMPERATURE,
        "low_iou_threshold": float(LOW_IOU_THRESHOLD),
        "sequence_metrics": sequence_metrics,
        "mean_sampled_crop_j_mean": float(np.mean(means)) if means else float("nan"),
        "debug_note": "Per-frame mean_confidence is retained, but low-confidence frame lists are not used as primary conclusions.",
        "debug_low_confidence_frames": low_confidence,
        "low_iou_frames": low_iou,
    }
