"""Video IO and V-JEPA model helpers for the Phase 1 probe."""

from __future__ import annotations

import gc
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

from research.scripts.common.data_paths import REPO_ROOT


VJEPA21_VITB_MODEL = "vjepa2_1_vit_base_384"
VJEPA21_VITG_TEACHER_MODEL = "vjepa2_1_vit_gigantic_384"
VJEPA21_PREPROCESSOR = "vjepa2_preprocessor"
MODEL_NAME = "V-JEPA 2.1 ViT-B/16 384"
TEACHER_MODEL_NAME = "V-JEPA 2.1 ViT-G/16 384 teacher"
PREDICTOR_MODEL_NAME = "V-JEPA 2.1 ViT-B predictor"

SCHEMA_VERSION = "1.0"
CROP_SIZE = 384
PATCH_SIZE = 16
TUBELET_SIZE = 2
GRID_HEIGHT = 24
GRID_WIDTH = 24
TOKENS_PER_TIME = GRID_HEIGHT * GRID_WIDTH
FEATURE_DIM = 768
TEACHER_FEATURE_DIM = 1664
IMAGE_SIZE = CROP_SIZE

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FEATURE_METADATA_KEYS = [
    "schema_version",
    "phase_name",
    "model_name",
    "crop_size",
    "patch_size",
    "tubelet_size",
    "num_sampled_frames",
    "temporal_tokens",
    "grid_height",
    "grid_width",
    "feature_dim",
    "source_video_name",
]


@dataclass
class DenseFeatureResult:
    sampled_frame_indices: np.ndarray
    raw_frames: np.ndarray
    processed_video: torch.Tensor
    model_frames: np.ndarray
    fps: float
    patch_tokens: np.ndarray
    temporal_features: np.ndarray
    video_feature: np.ndarray
    temporal_similarity: np.ndarray
    consecutive_similarity: np.ndarray
    metadata: dict[str, Any]
    runtime: dict[str, Any]


def open_video(path: Path) -> VideoReader:
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    return VideoReader(str(path), ctx=cpu(0))


def sample_video_frames(video_path: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray, float]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    reader = open_video(video_path)
    total_frames = len(reader)
    if total_frames < 1:
        raise RuntimeError(f"Cannot decode frames from: {video_path}")
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=np.int64)
    frames = reader.get_batch(frame_indices).asnumpy()
    return frames, frame_indices, float(reader.get_avg_fps())


def load_preprocessor(crop_size: int):
    return torch.hub.load(
        str(REPO_ROOT),
        VJEPA21_PREPROCESSOR,
        source="local",
        crop_size=crop_size,
    )


def preprocess_frames(frames: np.ndarray, crop_size: int) -> torch.Tensor:
    preprocessor = load_preprocessor(crop_size)
    processed = preprocessor(frames)[0]
    if processed.ndim != 4 or processed.shape[0] != 3:
        raise RuntimeError(
            "Unexpected preprocessor output shape. Expected (3, T, H, W), "
            f"got {tuple(processed.shape)}."
        )
    return processed


def denormalize_imagenet(video_tensor: torch.Tensor) -> np.ndarray:
    frames = video_tensor.detach().cpu().float().permute(1, 2, 3, 0).numpy()
    frames = frames * IMAGENET_STD[None, None, None, :] + IMAGENET_MEAN[None, None, None, :]
    return np.clip(frames, 0.0, 1.0)


def load_vitb_encoder_predictor(device: torch.device, *, dtype: torch.dtype = torch.bfloat16):
    encoder, predictor = torch.hub.load(
        str(REPO_ROOT),
        VJEPA21_VITB_MODEL,
        source="local",
        pretrained=True,
    )
    encoder = encoder.to(device=device, dtype=dtype).eval()
    predictor = predictor.to(device=device, dtype=dtype).eval()
    return encoder, predictor


def load_vitb_encoder(device: torch.device, *, dtype: torch.dtype = torch.bfloat16):
    encoder, predictor = load_vitb_encoder_predictor(device, dtype=dtype)
    del predictor
    release_cuda(device=device)
    return encoder


def load_vitg_teacher_encoder(device: torch.device, *, dtype: torch.dtype = torch.bfloat16):
    encoder, predictor = torch.hub.load(
        str(REPO_ROOT),
        VJEPA21_VITG_TEACHER_MODEL,
        source="local",
        pretrained=True,
    )
    del predictor
    release_cuda(device=device)
    return encoder.to(device=device, dtype=dtype).eval()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated() / 1024**2)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def release_cuda(*objects: Any, device: torch.device | None = None) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()


def build_feature_metadata(
    *,
    phase_name: str,
    source_video: Path,
    crop_size: int,
    num_sampled_frames: int,
    temporal_tokens: int,
    model_name: str = MODEL_NAME,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_name": phase_name,
        "model_name": model_name,
        "crop_size": int(crop_size),
        "patch_size": PATCH_SIZE,
        "tubelet_size": TUBELET_SIZE,
        "num_sampled_frames": int(num_sampled_frames),
        "temporal_tokens": int(temporal_tokens),
        "grid_height": GRID_HEIGHT,
        "grid_width": GRID_WIDTH,
        "feature_dim": FEATURE_DIM,
        "source_video_name": Path(source_video).name,
    }


def save_feature_npz(
    output_path: Path,
    *,
    sampled_frame_indices: np.ndarray,
    fps: float,
    patch_tokens: np.ndarray,
    temporal_features: np.ndarray,
    video_feature: np.ndarray,
    temporal_similarity: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "sampled_frame_indices": sampled_frame_indices.astype(np.int64),
        "fps": np.array([fps], dtype=np.float32),
        "patch_tokens": patch_tokens.astype(np.float32),
        "temporal_features": temporal_features.astype(np.float32),
        "video_feature": video_feature.astype(np.float32),
        "temporal_similarity": temporal_similarity.astype(np.float32),
    }
    for key in FEATURE_METADATA_KEYS:
        if key not in metadata:
            raise ValueError(f"Missing feature metadata key: {key}")
        payload[key] = np.array(metadata[key])
    np.savez_compressed(output_path, **payload)


def extract_dense_features(
    *,
    video_path: Path,
    device: torch.device,
    phase_name: str,
    num_frames: int,
    crop_size: int,
) -> DenseFeatureResult:
    if crop_size != CROP_SIZE:
        raise ValueError(f"Phase 1 expects crop_size={CROP_SIZE}; got {crop_size}.")
    if num_frames % TUBELET_SIZE != 0:
        raise ValueError("num_frames must be divisible by tubelet_size=2.")

    stage_start = time.perf_counter()
    frames, sampled_indices, fps = sample_video_frames(video_path, num_frames)
    processed = preprocess_frames(frames, crop_size)
    model_frames = denormalize_imagenet(processed)
    video = processed.unsqueeze(0).to(device, non_blocking=True)

    encoder = load_vitb_encoder(device)
    reset_peak_memory(device)
    forward_start = time.perf_counter()
    with torch.inference_mode(), autocast_context(device):
        tokens = encoder(video)
    synchronize(device)
    encoder_seconds = time.perf_counter() - forward_start
    encoder_peak_mb = peak_memory_mb(device)
    del encoder
    release_cuda(device=device)

    tokens = tokens.float().cpu()
    batch_size, token_count, embedding_dim = tokens.shape
    temporal_tokens = num_frames // TUBELET_SIZE
    expected_token_count = temporal_tokens * TOKENS_PER_TIME
    expected_shape = (1, expected_token_count, FEATURE_DIM)
    if (batch_size, token_count, embedding_dim) != expected_shape:
        raise RuntimeError(
            f"Unexpected encoder output shape {tuple(tokens.shape)}; expected {expected_shape}."
        )

    token_grid_torch = tokens.reshape(1, temporal_tokens, GRID_HEIGHT, GRID_WIDTH, FEATURE_DIM)
    temporal_features = token_grid_torch.mean(dim=(2, 3)).squeeze(0)
    video_feature = temporal_features.mean(dim=0)
    normalized_features = F.normalize(temporal_features, dim=-1)
    temporal_similarity = normalized_features @ normalized_features.T
    consecutive_similarity = torch.diagonal(temporal_similarity, offset=1).numpy()

    patch_tokens_np = tokens.numpy()
    temporal_features_np = temporal_features.numpy()
    video_feature_np = video_feature.numpy()
    temporal_similarity_np = temporal_similarity.numpy()

    metadata = build_feature_metadata(
        phase_name=phase_name,
        source_video=video_path,
        crop_size=crop_size,
        num_sampled_frames=num_frames,
        temporal_tokens=temporal_tokens,
    )
    runtime = {
        "elapsed_seconds": time.perf_counter() - stage_start,
        "encoder_forward_seconds": encoder_seconds,
        "peak_gpu_memory_mb": encoder_peak_mb,
        "patch_token_shape": tuple(patch_tokens_np.shape),
        "temporal_features_shape": tuple(temporal_features_np.shape),
        "video_feature_shape": tuple(video_feature_np.shape),
        "temporal_similarity_shape": tuple(temporal_similarity_np.shape),
        "mean_consecutive_temporal_similarity": float(consecutive_similarity.mean()),
    }
    return DenseFeatureResult(
        sampled_frame_indices=sampled_indices,
        raw_frames=frames,
        processed_video=processed,
        model_frames=model_frames,
        fps=fps,
        patch_tokens=patch_tokens_np,
        temporal_features=temporal_features_np,
        video_feature=video_feature_np,
        temporal_similarity=temporal_similarity_np,
        consecutive_similarity=consecutive_similarity,
        metadata=metadata,
        runtime=runtime,
    )
