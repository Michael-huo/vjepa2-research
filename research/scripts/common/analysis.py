"""Analysis routines for representation, correspondence, and completion."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any

import numpy as np
import torch

from research.scripts.common.runtime import CompletionConfig, CorrespondenceConfig, RepresentationConfig
from research.scripts.common.video_models import (
    FEATURE_DIM,
    GRID_HEIGHT,
    GRID_WIDTH,
    PATCH_SIZE,
    TEACHER_FEATURE_DIM,
    TOKENS_PER_TIME,
    VJEPA21_VITB_MODEL,
    VJEPA21_VITG_TEACHER_MODEL,
    DenseFeatureResult,
    autocast_context,
    load_vitb_encoder_predictor,
    load_vitg_teacher_encoder,
    peak_memory_mb,
    release_cuda,
    reset_peak_memory,
    synchronize,
)


@dataclass
class RepresentationResult:
    token_grid: np.ndarray
    selected_tokens: list[int]
    pca_maps: np.ndarray
    top_temporal_changes: list[dict[str, Any]]
    local_changes: list[dict[str, Any]]
    metrics: dict[str, Any]
    runtime: dict[str, Any]


@dataclass
class CorrespondenceResult:
    reference_mask: np.ndarray
    patch_bounds: tuple[int, int, int, int]
    similarity_maps: list[np.ndarray]
    appearance_maps: list[np.ndarray]
    masks: list[np.ndarray]
    centroids: list[np.ndarray]
    records: list[dict[str, Any]]
    selected_records: list[dict[str, Any]]
    selected_indices: list[int]
    uncertainty_flags: list[dict[str, Any]]
    metrics: dict[str, Any]
    runtime: dict[str, Any]


@dataclass
class CompletionResult:
    target_indices: torch.Tensor
    target_rows: np.ndarray
    target_cols: np.ndarray
    target_rect: tuple[int, int, int, int]
    target_frame_index: int
    cosine_grids: dict[str, np.ndarray]
    retrieval_matrices: dict[str, np.ndarray]
    mode_metrics: dict[str, dict[str, Any]]
    teacher_runtime: dict[str, Any]
    runtime: dict[str, Any]


def normalize_np(features: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return features / (np.linalg.norm(features, axis=axis, keepdims=True) + eps)


def normalize_torch(features: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return features / (torch.linalg.norm(features, dim=dim, keepdim=True) + eps)


def token_grid_from_patch_tokens(patch_tokens: np.ndarray) -> np.ndarray:
    if patch_tokens.ndim != 3 or patch_tokens.shape[0] != 1 or patch_tokens.shape[2] != FEATURE_DIM:
        raise ValueError(f"Unexpected patch_tokens shape: {patch_tokens.shape}")
    token_count = patch_tokens.shape[1] // TOKENS_PER_TIME
    if token_count * TOKENS_PER_TIME != patch_tokens.shape[1]:
        raise ValueError("patch_tokens length is not divisible by 24x24.")
    return patch_tokens.reshape(token_count, GRID_HEIGHT, GRID_WIDTH, FEATURE_DIM)


def validate_token_range(start_token: int, end_token: int, temporal_tokens: int) -> None:
    if not (0 <= start_token <= end_token < temporal_tokens):
        raise ValueError(
            f"Invalid token range [{start_token}, {end_token}] for temporal_tokens={temporal_tokens}."
        )


def cosine_change_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = normalize_np(a)
    b_norm = normalize_np(b)
    return np.clip(1.0 - np.sum(a_norm * b_norm, axis=-1), 0.0, 2.0)


def pca_project_range(token_grid: np.ndarray, start_token: int, end_token: int) -> np.ndarray:
    selected = token_grid[start_token : end_token + 1].reshape(-1, FEATURE_DIM)
    mean = selected.mean(axis=0, keepdims=True)
    centered = selected - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:3].T
    projected = centered @ basis
    lo = np.percentile(projected, 1, axis=0, keepdims=True)
    hi = np.percentile(projected, 99, axis=0, keepdims=True)
    projected = np.clip((projected - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return projected.reshape(end_token - start_token + 1, GRID_HEIGHT, GRID_WIDTH, 3)


def roi_to_patch_mask(roi: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive.")
    row_start = max(0, int(np.floor(y / PATCH_SIZE)))
    row_end = min(GRID_HEIGHT, int(np.ceil((y + h) / PATCH_SIZE)))
    col_start = max(0, int(np.floor(x / PATCH_SIZE)))
    col_end = min(GRID_WIDTH, int(np.ceil((x + w) / PATCH_SIZE)))
    if row_end <= row_start or col_end <= col_start:
        raise ValueError("ROI does not cover any valid 24x24 patch.")
    mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
    mask[row_start:row_end, col_start:col_end] = True
    return mask, (row_start, row_end, col_start, col_end)


def analyse_representation(features: DenseFeatureResult, config: RepresentationConfig) -> RepresentationResult:
    stage_start = time.perf_counter()
    temporal_tokens = int(features.metadata["temporal_tokens"])
    validate_token_range(config.start_token, config.end_token, temporal_tokens)
    token_grid = token_grid_from_patch_tokens(features.patch_tokens)
    pca_maps = pca_project_range(token_grid, config.start_token, config.end_token)

    ranked_pairs = np.argsort(features.consecutive_similarity)
    top_temporal_changes: list[dict[str, Any]] = []
    for rank, token_index in enumerate(ranked_pairs[: config.top_temporal_changes], start=1):
        token_index = int(token_index)
        top_temporal_changes.append(
            {
                "rank": rank,
                "token": token_index,
                "next_token": token_index + 1,
                "similarity": float(features.consecutive_similarity[token_index]),
                "frame_indices": [
                    int(features.sampled_frame_indices[2 * token_index]),
                    int(features.sampled_frame_indices[2 * (token_index + 1)]),
                ],
            }
        )

    local_rows: list[dict[str, Any]] = []
    for token_index in range(config.start_token, config.end_token):
        change_map = cosine_change_map(token_grid[token_index], token_grid[token_index + 1])
        local_rows.append(
            {
                "token": token_index,
                "next_token": token_index + 1,
                "change_map": change_map,
                "mean_change": float(change_map.mean()),
                "max_change": float(change_map.max()),
                "temporal_similarity": float(features.temporal_similarity[token_index, token_index + 1]),
                "frame_indices": [
                    int(features.sampled_frame_indices[2 * token_index]),
                    int(features.sampled_frame_indices[2 * (token_index + 1)]),
                ],
            }
        )
    local_changes = sorted(local_rows, key=lambda item: item["mean_change"], reverse=True)[: config.top_local_changes]

    metrics = {
        "mean_consecutive_temporal_similarity": float(features.consecutive_similarity.mean()),
        "top_temporal_changes": [
            {
                "rank": item["rank"],
                "token_pair": [item["token"], item["next_token"]],
                "cosine_similarity": item["similarity"],
                "frame_indices": item["frame_indices"],
            }
            for item in top_temporal_changes
        ],
        "local_latent_change_summary": [
            {
                "token_pair": [item["token"], item["next_token"]],
                "mean_change": item["mean_change"],
                "max_change": item["max_change"],
                "temporal_similarity": item["temporal_similarity"],
                "frame_indices": item["frame_indices"],
            }
            for item in local_changes
        ],
        "runtime": {},
    }
    runtime = {"elapsed_seconds": time.perf_counter() - stage_start}
    metrics["runtime"] = runtime
    return RepresentationResult(
        token_grid=token_grid,
        selected_tokens=list(config.selected_tokens),
        pca_maps=pca_maps,
        top_temporal_changes=top_temporal_changes,
        local_changes=local_changes,
        metrics=metrics,
        runtime=runtime,
    )


def connected_component(binary_mask: np.ndarray, seed_row: int, seed_col: int) -> np.ndarray:
    if not binary_mask[seed_row, seed_col]:
        return np.zeros_like(binary_mask, dtype=bool)
    output = np.zeros_like(binary_mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque([(int(seed_row), int(seed_col))])
    output[seed_row, seed_col] = True
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < GRID_HEIGHT
                and 0 <= nc < GRID_WIDTH
                and binary_mask[nr, nc]
                and not output[nr, nc]
            ):
                output[nr, nc] = True
                queue.append((nr, nc))
    return output


def crop_connected_by_peak(
    component: np.ndarray,
    scores: np.ndarray,
    peak_row: int,
    peak_col: int,
    max_patches: int,
) -> np.ndarray:
    if int(component.sum()) <= max_patches:
        return component
    selected = np.zeros_like(component, dtype=bool)
    selected[peak_row, peak_col] = True
    visited = np.zeros_like(component, dtype=bool)
    heap: list[tuple[float, float, int, int]] = []

    def push_neighbors(row: int, col: int) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < GRID_HEIGHT
                and 0 <= nc < GRID_WIDTH
                and component[nr, nc]
                and not selected[nr, nc]
                and not visited[nr, nc]
            ):
                distance = abs(nr - peak_row) + abs(nc - peak_col)
                heappush(heap, (float(distance), -float(scores[nr, nc]), int(nr), int(nc)))
                visited[nr, nc] = True

    push_neighbors(peak_row, peak_col)
    while heap and int(selected.sum()) < max_patches:
        _, _, row, col = heappop(heap)
        if selected[row, col]:
            continue
        selected[row, col] = True
        push_neighbors(row, col)
    return selected


def build_mask_from_scores(
    appearance_scores: np.ndarray,
    valid_region: np.ndarray,
    peak_row: int,
    peak_col: int,
    *,
    quantile: float,
    peak_margin: float,
    min_patches: int,
    max_patches: int,
) -> tuple[np.ndarray, list[str]]:
    flags: list[str] = []
    valid_scores = appearance_scores[valid_region]
    if valid_scores.size == 0:
        raise RuntimeError("No valid patches remain in the spatial search region.")
    peak_score = float(appearance_scores[peak_row, peak_col])
    chosen = np.zeros_like(appearance_scores, dtype=bool)
    relaxed = False
    for relaxation in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10):
        threshold = max(
            float(np.quantile(valid_scores, quantile)) - relaxation,
            peak_score - peak_margin - relaxation,
        )
        candidates = (appearance_scores >= threshold) & valid_region
        component = connected_component(candidates, peak_row, peak_col)
        if int(component.sum()) >= min_patches:
            chosen = component
            relaxed = relaxation > 0
            break
    if not chosen.any():
        chosen[peak_row, peak_col] = True
        flags.append("low_evidence_single_peak")
    if relaxed:
        flags.append("threshold_relaxed")
    if int(chosen.sum()) > max_patches:
        chosen = crop_connected_by_peak(chosen, appearance_scores, peak_row, peak_col, max_patches)
        flags.append("area_capped_connected")
    if int(chosen.sum()) < min_patches:
        flags.append("below_min_mask_patches")
    if int(chosen.sum()) >= max_patches:
        flags.append("mask_reached_max_patches")
    return chosen, flags


def weighted_centroid(mask: np.ndarray, scores: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(mask.shape)
    if not mask.any():
        return np.array([np.nan, np.nan], dtype=np.float32)
    weights = np.where(mask, scores, 0.0)
    selected = weights[mask]
    weights = np.where(mask, weights - float(selected.min()) + 1e-4, 0.0)
    total = float(weights.sum())
    if total <= 0:
        points = np.argwhere(mask)
        return points.mean(axis=0).astype(np.float32)
    return np.array(
        [
            float((rows * weights).sum() / total),
            float((cols * weights).sum() / total),
        ],
        dtype=np.float32,
    )


def max_reference_similarity(target_grid: np.ndarray, reference_features: np.ndarray) -> np.ndarray:
    flat = target_grid.reshape(-1, FEATURE_DIM)
    return np.max(flat @ reference_features.T, axis=1).reshape(GRID_HEIGHT, GRID_WIDTH)


def normalize_uncertainty_flags(flags: list[str]) -> list[str]:
    normalized: set[str] = set()
    if any(flag in flags for flag in ("threshold_relaxed", "area_capped_connected", "mask_reached_max_patches")):
        normalized.add("area_expanded")
    if "diffuse_response" in flags:
        normalized.add("response_diffuse")
    if "large_peak_jump" in flags:
        normalized.add("trajectory_jump")
    if any(
        flag in flags
        for flag in (
            "low_peak_similarity",
            "low_mask_similarity",
            "low_evidence_single_peak",
            "below_min_mask_patches",
        )
    ):
        normalized.add("low_confidence")
    return sorted(normalized)


def analyse_correspondence(
    token_grid: np.ndarray,
    features: DenseFeatureResult,
    config: CorrespondenceConfig,
) -> CorrespondenceResult:
    stage_start = time.perf_counter()
    temporal_tokens = int(features.metadata["temporal_tokens"])
    validate_token_range(config.start_token, config.end_token, temporal_tokens)
    if not (config.start_token <= config.reference_token <= config.end_token):
        raise ValueError("reference_token must lie within the configured token range.")

    reference_mask, patch_bounds = roi_to_patch_mask(config.roi)
    token_grid_norm = normalize_np(token_grid)
    reference_features = normalize_np(token_grid_norm[config.reference_token][reference_mask])
    reference_centroid = weighted_centroid(reference_mask, np.ones_like(reference_mask, dtype=np.float32))
    row_coords, col_coords = np.indices((GRID_HEIGHT, GRID_WIDTH))
    token_indices = list(range(config.start_token, config.end_token + 1))
    memory_bank: deque[np.ndarray] = deque(maxlen=config.memory_frames)
    memory_bank.append(reference_features)

    similarity_maps: list[np.ndarray] = []
    appearance_maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    uncertainty_flags: list[dict[str, Any]] = []
    previous_centroid = reference_centroid.copy()

    for token_index in token_indices:
        target = token_grid_norm[token_index]
        reference_similarity = max_reference_similarity(target, reference_features)
        memory_features = np.concatenate(list(memory_bank), axis=0)
        memory_similarity = max_reference_similarity(target, memory_features)
        appearance = (
            config.reference_weight * reference_similarity
            + (1.0 - config.reference_weight) * memory_similarity
        )
        flags: list[str] = []
        if token_index == config.reference_token:
            mask = reference_mask.copy()
            centroid = reference_centroid.copy()
            valid_region = np.ones_like(mask, dtype=bool)
            peak_row, peak_col = np.unravel_index(
                np.argmax(np.where(mask, appearance, -np.inf)),
                appearance.shape,
            )
        else:
            distance = np.sqrt(
                (row_coords - previous_centroid[0]) ** 2
                + (col_coords - previous_centroid[1]) ** 2
            )
            valid_region = distance <= config.max_jump
            spatial_prior = np.exp(-(distance**2) / (2.0 * config.spatial_sigma**2))
            fused = appearance + config.spatial_weight * spatial_prior
            gated_fused = np.where(valid_region, fused, -np.inf)
            peak_row, peak_col = np.unravel_index(np.argmax(gated_fused), gated_fused.shape)
            peak_distance = float(distance[peak_row, peak_col])
            if peak_distance > config.max_jump * 0.75:
                flags.append("large_peak_jump")
            mask, mask_flags = build_mask_from_scores(
                appearance,
                valid_region,
                int(peak_row),
                int(peak_col),
                quantile=config.mask_quantile,
                peak_margin=config.peak_margin,
                min_patches=config.min_mask_patches,
                max_patches=config.max_mask_patches,
            )
            flags.extend(mask_flags)
            centroid = weighted_centroid(mask, appearance)

            selected_indices = np.argwhere(mask)
            if len(selected_indices) > 0:
                selected_scores = appearance[mask]
                top_count = min(config.top_k_update, len(selected_indices))
                top_order = np.argsort(selected_scores)[-top_count:]
                update_indices = selected_indices[top_order]
                update_features = target[update_indices[:, 0], update_indices[:, 1]]
                memory_bank.append(normalize_np(update_features))

        peak_app = float(appearance[peak_row, peak_col])
        peak_ref = float(reference_similarity[peak_row, peak_col])
        mask_mean = float(appearance[mask].mean()) if mask.any() else float("nan")
        response_std = float(appearance[valid_region].std()) if valid_region.any() else 0.0
        if response_std < 0.03:
            flags.append("diffuse_response")
        if peak_app < 0.15:
            flags.append("low_peak_similarity")
        if mask_mean < 0.10:
            flags.append("low_mask_similarity")

        normalized_flags = normalize_uncertainty_flags(flags)
        frame_index = int(features.sampled_frame_indices[2 * token_index])
        record = {
            "token": int(token_index),
            "frame_index": frame_index,
            "peak_row": int(peak_row),
            "peak_col": int(peak_col),
            "peak_similarity": peak_app,
            "peak_reference_similarity": peak_ref,
            "mask_patch_count": int(mask.sum()),
            "mask_mean_similarity": mask_mean,
            "centroid_row": float(centroid[0]),
            "centroid_col": float(centroid[1]),
            "response_std": response_std,
            "uncertainty_flags": normalized_flags,
        }
        similarity_maps.append(reference_similarity)
        appearance_maps.append(appearance)
        masks.append(mask)
        centroids.append(centroid)
        records.append(record)
        uncertainty_flags.append({"token": int(token_index), "flags": normalized_flags})
        previous_centroid = centroid.copy()

    selected_indices = [token_indices.index(token) for token in config.selected_tokens]
    selected_records = [records[index] for index in selected_indices]
    all_flags = sorted({flag for item in uncertainty_flags for flag in item["flags"]})
    metrics = {
        "reference_roi": list(config.roi),
        "reference_token": int(config.reference_token),
        "reference_patch_count": int(reference_mask.sum()),
        "token_range": [int(config.start_token), int(config.end_token)],
        "selected_tokens": list(config.selected_tokens),
        "trajectory": [
            {
                "token": record["token"],
                "centroid_row": record["centroid_row"],
                "centroid_col": record["centroid_col"],
                "mask_patch_count": record["mask_patch_count"],
                "mask_mean_similarity": record["mask_mean_similarity"],
                "peak_similarity": record["peak_similarity"],
                "uncertainty_flags": record["uncertainty_flags"],
            }
            for record in records
        ],
        "uncertainty_flags": uncertainty_flags,
        "aggregate_uncertainty_flags": all_flags,
        "runtime": {},
    }
    runtime = {"elapsed_seconds": time.perf_counter() - stage_start}
    metrics["runtime"] = runtime
    return CorrespondenceResult(
        reference_mask=reference_mask,
        patch_bounds=patch_bounds,
        similarity_maps=similarity_maps,
        appearance_maps=appearance_maps,
        masks=masks,
        centroids=centroids,
        records=records,
        selected_records=selected_records,
        selected_indices=selected_indices,
        uncertainty_flags=uncertainty_flags,
        metrics=metrics,
        runtime=runtime,
    )


def flat_index(token: int, row: int, col: int) -> int:
    return token * TOKENS_PER_TIME + row * GRID_WIDTH + col


def build_target_indices(target_token: int, row: int, height: int, col: int, width: int) -> torch.Tensor:
    indices = []
    for patch_row in range(row, row + height):
        for patch_col in range(col, col + width):
            indices.append(flat_index(target_token, patch_row, patch_col))
    return torch.tensor(indices, dtype=torch.long)


def validate_target_block(config: CompletionConfig, temporal_tokens: int) -> None:
    if not (0 <= config.target_token < temporal_tokens):
        raise ValueError(f"target_token must be in [0, {temporal_tokens - 1}].")
    if config.target_height <= 0 or config.target_width <= 0:
        raise ValueError("target block height and width must be positive.")
    if config.target_row < 0 or config.target_row + config.target_height > GRID_HEIGHT:
        raise ValueError("Target row block is outside the 24x24 patch grid.")
    if config.target_col < 0 or config.target_col + config.target_width > GRID_WIDTH:
        raise ValueError("Target column block is outside the 24x24 patch grid.")


def build_context_indices(
    mode: str,
    total_tokens: int,
    target_token: int,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    all_indices = torch.arange(total_tokens, dtype=torch.long)
    temporal_index = all_indices // TOKENS_PER_TIME
    target_mask = torch.zeros(total_tokens, dtype=torch.bool)
    target_mask[target_indices] = True
    if mode == "full":
        keep = ~target_mask
    elif mode == "spatial_only":
        keep = (temporal_index == target_token) & (~target_mask)
    elif mode == "temporal_bi":
        keep = temporal_index != target_token
    elif mode == "past_only":
        keep = temporal_index < target_token
    else:
        raise ValueError(f"Unsupported context mode: {mode}")
    context_indices = all_indices[keep]
    if len(context_indices) == 0:
        raise ValueError(f"Mode {mode!r} has no visible context tokens.")
    return context_indices


def target_coordinates(row: int, height: int, col: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    for patch_row in range(row, row + height):
        for patch_col in range(col, col + width):
            rows.append(patch_row)
            cols.append(patch_col)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def evaluate_latent_prediction(
    predicted_target: torch.Tensor,
    teacher_target: torch.Tensor,
    target_rows: np.ndarray,
    target_cols: np.ndarray,
) -> dict[str, Any]:
    predicted_norm = normalize_torch(predicted_target)
    teacher_norm = normalize_torch(teacher_target)
    pair_cosine = (predicted_norm * teacher_norm).sum(dim=-1)[0]
    retrieval = predicted_norm[0] @ teacher_norm[0].T
    correct_similarity = torch.diag(retrieval)
    ranks = (retrieval > correct_similarity.unsqueeze(1)).sum(dim=1) + 1
    n_targets = retrieval.shape[0]
    if n_targets > 1:
        nonmatching_mask = ~torch.eye(n_targets, dtype=torch.bool, device=retrieval.device)
        per_token_nonmatching = retrieval[nonmatching_mask].reshape(n_targets, n_targets - 1).mean(dim=1)
        mean_nonmatching = float(per_token_nonmatching.mean())
    else:
        per_token_nonmatching = torch.full((n_targets,), float("nan"), device=retrieval.device)
        mean_nonmatching = float("nan")

    best_match = retrieval.argmax(dim=1).cpu().numpy()
    row_error = target_rows[best_match] - target_rows
    col_error = target_cols[best_match] - target_cols
    chebyshev_error = np.maximum(np.abs(row_error), np.abs(col_error))
    euclidean_error = np.sqrt(row_error**2 + col_error**2)
    return {
        "pair_cosine": pair_cosine.cpu().numpy(),
        "per_token_nonmatching_cosine": per_token_nonmatching.cpu().numpy(),
        "mean_nonmatching_cosine": mean_nonmatching,
        "retrieval": retrieval.cpu().numpy(),
        "ranks": ranks.cpu().numpy(),
        "top1": float((ranks == 1).float().mean()),
        "top5": float((ranks <= min(5, n_targets)).float().mean()),
        "mean_rank": float(ranks.float().mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
        "neighbor_at_1": float((chebyshev_error <= 1).mean()),
        "mean_spatial_error": float(euclidean_error.mean()),
        "best_match_indices": best_match.astype(np.int64),
        "best_match_spatial_error": euclidean_error.astype(np.float32),
        "mean_target_cosine": float(pair_cosine.mean()),
        "median_target_cosine": float(pair_cosine.median()),
        "min_target_cosine": float(pair_cosine.min()),
        "max_target_cosine": float(pair_cosine.max()),
    }


def compact_mode_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_target_cosine": metric["mean_target_cosine"],
        "median_target_cosine": metric["median_target_cosine"],
        "min_target_cosine": metric["min_target_cosine"],
        "max_target_cosine": metric["max_target_cosine"],
        "mean_nonmatching_cosine": metric["mean_nonmatching_cosine"],
        "top1": metric["top1"],
        "top5": metric["top5"],
        "mean_rank": metric["mean_rank"],
        "mrr": metric["mrr"],
        "top1_within_1_patch_neighborhood": metric["neighbor_at_1"],
        "mean_best_match_spatial_error": metric["mean_spatial_error"],
        "visible_context_token_count": metric["visible_tokens"],
        "student_predictor_forward_seconds": metric["elapsed_seconds"],
        "student_predictor_peak_gpu_memory_mb": metric["peak_gpu_mb"],
    }


def analyse_completion(
    *,
    features: DenseFeatureResult,
    device: torch.device,
    config: CompletionConfig,
) -> CompletionResult:
    stage_start = time.perf_counter()
    temporal_tokens = int(features.metadata["temporal_tokens"])
    validate_target_block(config, temporal_tokens)
    total_tokens = temporal_tokens * TOKENS_PER_TIME
    target_indices_cpu = build_target_indices(
        config.target_token,
        config.target_row,
        config.target_height,
        config.target_col,
        config.target_width,
    )
    target_rows, target_cols = target_coordinates(
        config.target_row,
        config.target_height,
        config.target_col,
        config.target_width,
    )
    target_count = int(len(target_indices_cpu))
    video = features.processed_video.unsqueeze(0).to(device, non_blocking=True)
    target_indices_gpu = target_indices_cpu.unsqueeze(0).to(device)
    target_frame_slot = 2 * config.target_token
    target_frame_index = int(features.sampled_frame_indices[target_frame_slot])

    teacher_encoder = load_vitg_teacher_encoder(device)
    reset_peak_memory(device)
    teacher_start = time.perf_counter()
    with torch.inference_mode(), autocast_context(device):
        teacher_all = teacher_encoder(video)
    synchronize(device)
    teacher_seconds = time.perf_counter() - teacher_start
    teacher_peak_mb = peak_memory_mb(device)
    teacher_target = torch.index_select(teacher_all, 1, target_indices_gpu.reshape(-1)).float().cpu()
    expected_shape = (1, target_count, TEACHER_FEATURE_DIM)
    if teacher_target.shape != expected_shape:
        raise RuntimeError(
            "Teacher target shape mismatch. "
            f"model={VJEPA21_VITG_TEACHER_MODEL}, video_shape={tuple(video.shape)}, "
            f"target_indices_shape={tuple(target_indices_gpu.shape)}, "
            f"teacher_target_shape={tuple(teacher_target.shape)}, expected={expected_shape}."
        )
    del teacher_all
    del teacher_encoder
    release_cuda(device=device)

    student_encoder, predictor = load_vitb_encoder_predictor(device)
    mode_metrics: dict[str, dict[str, Any]] = {}
    cosine_grids: dict[str, np.ndarray] = {}
    retrieval_matrices: dict[str, np.ndarray] = {}
    for mode in config.modes:
        context_indices_cpu = build_context_indices(mode, total_tokens, config.target_token, target_indices_cpu)
        context_indices_gpu = context_indices_cpu.unsqueeze(0).to(device)
        reset_peak_memory(device)
        mode_start = time.perf_counter()
        with torch.inference_mode(), autocast_context(device):
            context_tokens = student_encoder(
                video,
                masks=[context_indices_gpu],
                training=False,
            )
            predicted_target, predicted_context = predictor(
                context_tokens,
                [context_indices_gpu],
                [target_indices_gpu],
                mod="video",
                mask_index=0,
            )
        synchronize(device)
        elapsed = time.perf_counter() - mode_start
        peak_mb = peak_memory_mb(device)
        predicted_target_cpu = predicted_target.float().cpu()
        if predicted_target_cpu.shape != expected_shape:
            raise RuntimeError(
                "Predictor target shape mismatch. "
                f"student_model={VJEPA21_VITB_MODEL}, teacher_model={VJEPA21_VITG_TEACHER_MODEL}, "
                f"mode={mode}, video_shape={tuple(video.shape)}, "
                f"context_indices_shape={tuple(context_indices_gpu.shape)}, "
                f"target_indices_shape={tuple(target_indices_gpu.shape)}, "
                f"context_tokens_shape={tuple(context_tokens.shape)}, "
                f"predictor_target_shape={tuple(predicted_target_cpu.shape)}, "
                f"teacher_target_shape={tuple(teacher_target.shape)}. "
                "Check that student_encoder uses training=False and that the input uses 64 frames at 384 resolution."
            )
        metric = evaluate_latent_prediction(predicted_target_cpu, teacher_target, target_rows, target_cols)
        metric["visible_tokens"] = int(len(context_indices_cpu))
        metric["elapsed_seconds"] = float(elapsed)
        metric["peak_gpu_mb"] = float(peak_mb)
        metric["context_shape"] = tuple(context_tokens.shape)
        metric["prediction_shape"] = tuple(predicted_target_cpu.shape)
        mode_metrics[mode] = metric
        cosine_grids[mode] = metric["pair_cosine"].reshape(config.target_height, config.target_width)
        retrieval_matrices[mode] = metric["retrieval"]
        del context_tokens
        del predicted_target
        del predicted_context
        release_cuda(device=device)

    del student_encoder
    del predictor
    release_cuda(device=device)
    teacher_runtime = {
        "teacher_forward_seconds": float(teacher_seconds),
        "teacher_peak_gpu_memory_mb": float(teacher_peak_mb),
    }
    for metric in mode_metrics.values():
        metric["teacher_forward_seconds"] = float(teacher_seconds)
        metric["teacher_peak_gpu_memory_mb"] = float(teacher_peak_mb)

    target_rect = (
        config.target_col * PATCH_SIZE,
        config.target_row * PATCH_SIZE,
        config.target_width * PATCH_SIZE,
        config.target_height * PATCH_SIZE,
    )
    runtime = {"elapsed_seconds": time.perf_counter() - stage_start, **teacher_runtime}
    return CompletionResult(
        target_indices=target_indices_cpu,
        target_rows=target_rows,
        target_cols=target_cols,
        target_rect=target_rect,
        target_frame_index=target_frame_index,
        cosine_grids=cosine_grids,
        retrieval_matrices=retrieval_matrices,
        mode_metrics=mode_metrics,
        teacher_runtime=teacher_runtime,
        runtime=runtime,
    )
