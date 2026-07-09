"""Framewise V-JEPA dense feature extraction and PCA helpers for Phase 2."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from research.scripts.common.video_models import (
    CROP_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_NAME,
    autocast_context,
    load_vitb_encoder,
    peak_memory_mb,
    reset_peak_memory,
    synchronize,
)


FEATURE_MODE = "image_tokenizer_framewise"
SHORT_SIDE_SCALE = 256 / 224


@dataclass(frozen=True)
class CropMetadata:
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    crop_size: int


@dataclass
class ProcessedFrame:
    path: Path
    frame_index: int
    image: np.ndarray
    tensor: torch.Tensor
    metadata: CropMetadata


@dataclass
class FrameFeatureBatch:
    frame_paths: list[Path]
    frame_indices: list[int]
    frames: np.ndarray
    features: np.ndarray
    processed_resolution: tuple[int, int]
    original_sizes: list[tuple[int, int]]
    crop_metadata: list[dict[str, Any]]
    grid_shape: tuple[int, int]
    feature_dim: int
    patch_size: int
    runtime: dict[str, Any]


@dataclass(frozen=True)
class PcaFit:
    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_value_ratio: np.ndarray
    robust_min: np.ndarray
    robust_max: np.ndarray


def _resampling(name: str) -> int:
    try:
        resampling = Image.Resampling
    except AttributeError:
        resampling = Image
    if name == "nearest":
        return int(resampling.NEAREST)
    if name == "bilinear":
        return int(resampling.BILINEAR)
    raise ValueError(f"Unsupported resampling mode: {name}")


def resize_center_crop_pil(
    image: Image.Image,
    *,
    crop_size: int = CROP_SIZE,
    interpolation: str = "bilinear",
) -> tuple[Image.Image, CropMetadata]:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")

    short_side = int(crop_size * SHORT_SIDE_SCALE)
    if width < height:
        resized_width = short_side
        resized_height = int(round(height * short_side / width))
    else:
        resized_height = short_side
        resized_width = int(round(width * short_side / height))
    resized_width = max(resized_width, crop_size)
    resized_height = max(resized_height, crop_size)

    resized = image.resize((resized_width, resized_height), _resampling(interpolation))
    left = max(0, (resized_width - crop_size) // 2)
    top = max(0, (resized_height - crop_size) // 2)
    crop_box = (left, top, left + crop_size, top + crop_size)
    cropped = resized.crop(crop_box)
    metadata = CropMetadata(
        original_size=(height, width),
        resized_size=(resized_height, resized_width),
        crop_box=crop_box,
        crop_size=crop_size,
    )
    return cropped, metadata


def crop_metadata_json(metadata: CropMetadata) -> dict[str, Any]:
    return {
        "original_size_hw": list(metadata.original_size),
        "resized_size_hw": list(metadata.resized_size),
        "crop_box_xyxy": list(metadata.crop_box),
        "crop_size": int(metadata.crop_size),
    }


def preprocess_rgb_frame(path: Path, *, crop_size: int = CROP_SIZE) -> ProcessedFrame:
    image = Image.open(path).convert("RGB")
    cropped, metadata = resize_center_crop_pil(image, crop_size=crop_size, interpolation="bilinear")
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    normalized = (array - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).contiguous()
    return ProcessedFrame(
        path=path,
        frame_index=int(path.stem),
        image=array,
        tensor=tensor,
        metadata=metadata,
    )


def preprocess_mask(path: Path, *, crop_size: int = CROP_SIZE) -> tuple[np.ndarray, dict[str, Any]]:
    image = Image.open(path)
    cropped, metadata = resize_center_crop_pil(image, crop_size=crop_size, interpolation="nearest")
    mask = np.asarray(cropped)
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = mask > 0
    return binary.astype(bool), crop_metadata_json(metadata)


def load_phase2_encoder(device: torch.device):
    return load_vitb_encoder(device)


def infer_feature_grid(
    *,
    token_count: int,
    processed_resolution: tuple[int, int],
    encoder: torch.nn.Module,
) -> tuple[int, int, int]:
    height, width = processed_resolution
    patch_size = int(getattr(encoder, "patch_size", 0) or 0)
    if patch_size > 0:
        grid_height = height // patch_size
        grid_width = width // patch_size
        if grid_height * grid_width == token_count:
            return grid_height, grid_width, patch_size

    target_aspect = height / max(width, 1)
    best_pair: tuple[int, int] | None = None
    best_error = float("inf")
    for grid_height in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % grid_height != 0:
            continue
        grid_width = token_count // grid_height
        aspect = grid_height / max(grid_width, 1)
        error = abs(aspect - target_aspect)
        if error < best_error:
            best_pair = (grid_height, grid_width)
            best_error = error
    if best_pair is None:
        raise RuntimeError(f"Cannot infer dense feature grid for token_count={token_count}.")
    grid_height, grid_width = best_pair
    inferred_patch = height // grid_height if height % grid_height == 0 else 0
    return grid_height, grid_width, inferred_patch


def extract_frame_features(
    *,
    frame_paths: list[Path],
    encoder: torch.nn.Module,
    device: torch.device,
    frame_indices: list[int] | None = None,
    crop_size: int = CROP_SIZE,
    batch_size: int = 4,
) -> FrameFeatureBatch:
    if not frame_paths:
        raise ValueError("frame_paths must not be empty.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if frame_indices is None:
        frame_indices = [int(path.stem) for path in frame_paths]
    if len(frame_indices) != len(frame_paths):
        raise ValueError("frame_indices and frame_paths must have the same length.")

    stage_start = time.perf_counter()
    processed = [preprocess_rgb_frame(path, crop_size=crop_size) for path in frame_paths]
    frames = np.stack([item.image for item in processed], axis=0).astype(np.float32)
    tensors = [item.tensor for item in processed]
    processed_resolution = tuple(int(value) for value in frames.shape[1:3])

    reset_peak_memory(device)
    forward_start = time.perf_counter()
    feature_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size], dim=0).unsqueeze(2)
            batch = batch.to(device=device, non_blocking=True)
            with autocast_context(device):
                tokens = encoder(batch)
            synchronize(device)
            if tokens.ndim != 3:
                raise RuntimeError(f"Unexpected encoder output shape: {tuple(tokens.shape)}")
            feature_batches.append(tokens.float().cpu().numpy())
            del batch, tokens

    forward_seconds = time.perf_counter() - forward_start
    peak_mb = peak_memory_mb(device)
    token_array = np.concatenate(feature_batches, axis=0)
    if token_array.shape[0] != len(frame_paths):
        raise RuntimeError(
            f"Feature batch size mismatch: got {token_array.shape[0]}, expected {len(frame_paths)}."
        )
    token_count = int(token_array.shape[1])
    feature_dim = int(token_array.shape[2])
    grid_height, grid_width, patch_size = infer_feature_grid(
        token_count=token_count,
        processed_resolution=processed_resolution,
        encoder=encoder,
    )
    features = token_array.reshape(len(frame_paths), grid_height, grid_width, feature_dim)

    runtime = {
        "elapsed_seconds": float(time.perf_counter() - stage_start),
        "encoder_forward_seconds": float(forward_seconds),
        "peak_gpu_memory_mb": float(peak_mb),
        "frame_count": int(len(frame_paths)),
        "batch_size": int(batch_size),
    }
    return FrameFeatureBatch(
        frame_paths=frame_paths,
        frame_indices=[int(index) for index in frame_indices],
        frames=frames,
        features=features.astype(np.float32),
        processed_resolution=processed_resolution,
        original_sizes=[item.metadata.original_size for item in processed],
        crop_metadata=[crop_metadata_json(item.metadata) for item in processed],
        grid_shape=(grid_height, grid_width),
        feature_dim=feature_dim,
        patch_size=int(patch_size),
        runtime=runtime,
    )


def normalize_features(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return features / (np.linalg.norm(features, axis=-1, keepdims=True) + eps)


def fit_pca_rgb(features: np.ndarray, *, robust_percentiles: tuple[float, float] = (1.0, 99.0)) -> PcaFit:
    if features.ndim != 2:
        raise ValueError(f"PCA expects a 2D array, got shape {features.shape}.")
    if features.shape[0] < 3 or features.shape[1] < 3:
        raise ValueError(f"PCA needs at least 3 samples and 3 dimensions, got {features.shape}.")
    matrix = features.astype(np.float32, copy=False)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3].T.astype(np.float32)

    for component_index in range(components.shape[1]):
        anchor = int(np.argmax(np.abs(components[:, component_index])))
        if components[anchor, component_index] < 0:
            components[:, component_index] *= -1.0

    projected = centered @ components
    robust_min = np.percentile(projected, robust_percentiles[0], axis=0, keepdims=True)
    robust_max = np.percentile(projected, robust_percentiles[1], axis=0, keepdims=True)
    variance = singular_values**2
    variance_total = float(variance.sum())
    singular_total = float(singular_values.sum())
    explained = variance[:3] / variance_total if variance_total > 0 else np.zeros(3, dtype=np.float32)
    singular_ratio = (
        singular_values[:3] / singular_total if singular_total > 0 else np.zeros(3, dtype=np.float32)
    )
    return PcaFit(
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        singular_values=singular_values[:3].astype(np.float32),
        explained_variance_ratio=explained.astype(np.float32),
        singular_value_ratio=singular_ratio.astype(np.float32),
        robust_min=robust_min.astype(np.float32),
        robust_max=robust_max.astype(np.float32),
    )


def project_pca_rgb(features: np.ndarray, pca: PcaFit) -> np.ndarray:
    original_shape = features.shape[:-1]
    flat = features.reshape(-1, features.shape[-1]).astype(np.float32, copy=False)
    projected = (flat - pca.mean) @ pca.components
    rgb = (projected - pca.robust_min) / (pca.robust_max - pca.robust_min + 1e-8)
    return np.clip(rgb, 0.0, 1.0).reshape(*original_shape, 3).astype(np.float32)


def pca_metric_payload(pca: PcaFit) -> dict[str, Any]:
    return {
        "singular_values": pca.singular_values.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio.tolist(),
        "singular_value_ratio": pca.singular_value_ratio.tolist(),
        "robust_normalization_percentiles": [1.0, 99.0],
    }


def adjacent_patch_cosine(features: np.ndarray) -> list[float]:
    if features.shape[0] < 2:
        return []
    normalized = normalize_features(features)
    values = []
    for index in range(features.shape[0] - 1):
        cosine = np.sum(normalized[index] * normalized[index + 1], axis=-1)
        values.append(float(cosine.mean()))
    return values


def adjacent_pca_drift(pca_maps: np.ndarray) -> list[float]:
    if pca_maps.shape[0] < 2:
        return []
    return [
        float(np.mean(np.abs(pca_maps[index + 1] - pca_maps[index])))
        for index in range(pca_maps.shape[0] - 1)
    ]


def foreground_feature_similarity(source: np.ndarray, target: np.ndarray) -> float:
    source_tensor = torch.from_numpy(source.reshape(-1, source.shape[-1]))
    target_tensor = torch.from_numpy(target.reshape(-1, target.shape[-1]))
    source_norm = F.normalize(source_tensor.float(), dim=-1)
    target_norm = F.normalize(target_tensor.float(), dim=-1)
    return float((source_norm * target_norm).sum(dim=-1).mean())


def model_metadata() -> dict[str, str]:
    return {
        "model_name": MODEL_NAME,
        "feature_mode": FEATURE_MODE,
    }
