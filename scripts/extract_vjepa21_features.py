
import argparse

import os

import time



import matplotlib.pyplot as plt

import numpy as np

import torch

import torch.nn.functional as F

from decord import VideoReader, cpu





def parse_args():

    parser = argparse.ArgumentParser(

        description="Extract V-JEPA 2.1 ViT-B/16 features from one video."

    )

    parser.add_argument("--video", required=True, help="Path to an input video.")

    parser.add_argument(

        "--output-dir",

        default="outputs",

        help="Directory for extracted features and figures.",

    )

    parser.add_argument(

        "--num-frames",

        type=int,

        default=64,

        help="Number of uniformly sampled frames. Keep this at 64 for the default model.",

    )

    return parser.parse_args()





def sample_video(video_path: str, num_frames: int) -> tuple[np.ndarray, np.ndarray, float]:

    reader = VideoReader(video_path, ctx=cpu(0))

    total_frames = len(reader)

    fps = float(reader.get_avg_fps())



    if total_frames < 1:

        raise RuntimeError(f"Cannot decode frames from: {video_path}")



    frame_indices = np.linspace(

        0,

        total_frames - 1,

        num=num_frames,

        dtype=np.int64,

    )



    frames = reader.get_batch(frame_indices).asnumpy()

    return frames, frame_indices, fps





def main():

    args = parse_args()



    if not os.path.isfile(args.video):

        raise FileNotFoundError(f"Video not found: {args.video}")



    if args.num_frames % 2 != 0:

        raise ValueError("num_frames must be even because V-JEPA 2.1 uses tubelet_size=2.")



    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0")



    print(f"Reading video: {args.video}")

    frames, frame_indices, fps = sample_video(args.video, args.num_frames)

    print(f"Decoded frames: {frames.shape}")

    print(f"Average FPS: {fps:.3f}")

    print(f"Sampled frame indices: {frame_indices.tolist()}")



    print("Loading V-JEPA 2.1 ViT-B/16...")

    processor = torch.hub.load(

        ".",

        "vjepa2_preprocessor",

        source="local",

        crop_size=384,

    )

    encoder, _ = torch.hub.load(

        ".",

        "vjepa2_1_vit_base_384",

        source="local",

        pretrained=True,

    )

    encoder = encoder.to(device).eval()



    processed = processor(frames)[0]

    video = processed.unsqueeze(0).to(device, non_blocking=True)



    print(f"Model input shape: {tuple(video.shape)}")



    torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.synchronize()

    start = time.perf_counter()



    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):

        tokens = encoder(video)



    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start



    tokens = tokens.float().cpu()

    batch_size, token_count, embedding_dim = tokens.shape



    temporal_tokens = args.num_frames // 2

    spatial_tokens = 24 * 24



    if token_count != temporal_tokens * spatial_tokens:

        raise RuntimeError(

            f"Unexpected token count {token_count}; expected "

            f"{temporal_tokens} x {spatial_tokens} = {temporal_tokens * spatial_tokens}."

        )



    token_grid = tokens.reshape(

        batch_size,

        temporal_tokens,

        24,

        24,

        embedding_dim,

    )



    temporal_features = token_grid.mean(dim=(2, 3)).squeeze(0)

    video_feature = temporal_features.mean(dim=0)



    normalized_features = F.normalize(temporal_features, dim=-1)

    similarity = normalized_features @ normalized_features.T



    stem = os.path.splitext(os.path.basename(args.video))[0]

    output_npz = os.path.join(args.output_dir, f"{stem}_vjepa21_features.npz")

    output_png = os.path.join(args.output_dir, f"{stem}_vjepa21_temporal_similarity.png")



    np.savez_compressed(

        output_npz,

        sampled_frame_indices=frame_indices,

        fps=np.array([fps], dtype=np.float32),

        patch_tokens=tokens.numpy(),

        temporal_features=temporal_features.numpy(),

        video_feature=video_feature.numpy(),

        temporal_similarity=similarity.numpy(),

    )



    plt.figure(figsize=(7, 6))

    plt.imshow(similarity.numpy(), vmin=-1.0, vmax=1.0)

    plt.colorbar(label="Cosine similarity")

    plt.xlabel("Temporal token index")

    plt.ylabel("Temporal token index")

    plt.title("V-JEPA 2.1 temporal feature similarity")

    plt.tight_layout()

    plt.savefig(output_png, dpi=180)

    plt.close()



    consecutive_similarity = torch.diagonal(similarity, offset=1).numpy()

    print()

    print("Feature extraction: success")

    print(f"Patch token shape: {tuple(tokens.shape)}")

    print(f"Token grid shape: {tuple(token_grid.shape)}")

    print(f"Temporal feature shape: {tuple(temporal_features.shape)}")

    print(f"Video feature shape: {tuple(video_feature.shape)}")

    print(f"Elapsed seconds: {elapsed:.3f}")

    print(f"Peak GPU memory MB: {torch.cuda.max_memory_allocated(device) / 1024**2:.1f}")

    print(f"Mean consecutive temporal similarity: {consecutive_similarity.mean():.4f}")

    print(f"Saved features: {output_npz}")

    print(f"Saved similarity figure: {output_png}")





if __name__ == "__main__":

    main()

