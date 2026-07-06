
import argparse

import gc

import os

import time



import matplotlib.pyplot as plt

import numpy as np

import torch

from decord import VideoReader, cpu





GRID = 24

PATCH = 16

TOKENS_PER_TIME = GRID * GRID





def parse_args():

    parser = argparse.ArgumentParser(

        description="Evaluate V-JEPA 2.1 masked latent prediction on a video block."

    )

    parser.add_argument("--features", required=True, help="Existing .npz feature file.")

    parser.add_argument("--video", required=True, help="Original source video.")

    parser.add_argument("--output-dir", default="outputs")

    parser.add_argument("--target-token", type=int, default=16)

    parser.add_argument("--target-temporal", type=int, default=1)

    parser.add_argument("--target-row", type=int, default=16)

    parser.add_argument("--target-height", type=int, default=6)

    parser.add_argument("--target-col", type=int, default=2)

    parser.add_argument("--target-width", type=int, default=8)

    return parser.parse_args()





def normalize(x):

    return x / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-8)





def denormalize_imagenet(video_tensor):

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)

    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)



    frames = video_tensor.detach().cpu().float().permute(1, 2, 3, 0).numpy()

    frames = frames * std[None, None, None, :] + mean[None, None, None, :]

    return np.clip(frames, 0.0, 1.0)





def build_target_indices(t0, temporal, row0, height, col0, width):

    indices = []



    for t in range(t0, t0 + temporal):

        for row in range(row0, row0 + height):

            for col in range(col0, col0 + width):

                indices.append(t * TOKENS_PER_TIME + row * GRID + col)



    return torch.tensor(indices, dtype=torch.long)





def ensure_valid_block(args, temporal_tokens):

    if args.target_token < 0 or args.target_token + args.target_temporal > temporal_tokens:

        raise ValueError("Target temporal block is outside the video token grid.")



    if args.target_row < 0 or args.target_row + args.target_height > GRID:

        raise ValueError("Target row block is outside the 24x24 patch grid.")



    if args.target_col < 0 or args.target_col + args.target_width > GRID:

        raise ValueError("Target column block is outside the 24x24 patch grid.")





def gather_tokens(tokens, indices):

    expanded = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])

    return torch.gather(tokens, dim=1, index=expanded)





def main():

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)



    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0")



    if not os.path.isfile(args.features):

        raise FileNotFoundError(f"Feature file not found: {args.features}")



    if not os.path.isfile(args.video):

        raise FileNotFoundError(f"Video file not found: {args.video}")



    with np.load(args.features) as data:

        sampled_indices = data["sampled_frame_indices"].astype(np.int64)

        fps = float(data["fps"][0])



    temporal_tokens = len(sampled_indices) // 2

    ensure_valid_block(args, temporal_tokens)



    target_indices_cpu = build_target_indices(

        args.target_token,

        args.target_temporal,

        args.target_row,

        args.target_height,

        args.target_col,

        args.target_width,

    )



    total_tokens = temporal_tokens * TOKENS_PER_TIME

    all_indices_cpu = torch.arange(total_tokens, dtype=torch.long)



    context_mask_cpu = torch.ones(total_tokens, dtype=torch.bool)

    context_mask_cpu[target_indices_cpu] = False

    context_indices_cpu = all_indices_cpu[context_mask_cpu]



    target_indices = target_indices_cpu.unsqueeze(0).to(device)

    context_indices = context_indices_cpu.unsqueeze(0).to(device)



    print(f"Temporal token grid: {temporal_tokens} x {GRID} x {GRID}")

    print(f"Total patch tokens: {total_tokens}")

    print(f"Masked target tokens: {target_indices.shape[1]}")

    print(f"Visible context tokens: {context_indices.shape[1]}")



    reader = VideoReader(args.video, ctx=cpu(0))

    raw_frames = reader.get_batch(sampled_indices).asnumpy()



    print("Loading official V-JEPA preprocessor...")

    processor = torch.hub.load(

        ".",

        "vjepa2_preprocessor",

        source="local",

        crop_size=384,

    )



    processed = processor(raw_frames)[0]

    video_visual = denormalize_imagenet(processed)

    video = processed.unsqueeze(0).to(device, non_blocking=True)



    target_frame_slot = 2 * args.target_token

    target_frame_index = int(sampled_indices[target_frame_slot])

    target_time_seconds = target_frame_index / fps



    print()

    print("Stage 1/2: loading ViT-G/16 reference encoder...")

    teacher_encoder, unused_teacher_predictor = torch.hub.load(

        ".",

        "vjepa2_1_vit_gigantic_384",

        source="local",

        pretrained=True,

    )



    del unused_teacher_predictor

    gc.collect()



    teacher_encoder = teacher_encoder.to(device=device, dtype=torch.bfloat16).eval()



    torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.synchronize()

    teacher_start = time.perf_counter()



    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):

        teacher_all = teacher_encoder(video)



    torch.cuda.synchronize()

    teacher_seconds = time.perf_counter() - teacher_start



    teacher_target = gather_tokens(

        teacher_all.float().cpu(),

        target_indices_cpu.unsqueeze(0),

    )



    teacher_context = gather_tokens(

        teacher_all.float().cpu(),

        context_indices_cpu.unsqueeze(0),

    )



    print(f"Teacher full-token shape: {tuple(teacher_all.shape)}")

    print(f"Teacher target shape: {tuple(teacher_target.shape)}")

    print(f"Teacher forward seconds: {teacher_seconds:.3f}")

    print(f"Teacher peak GPU MB: {torch.cuda.max_memory_allocated(device) / 1024**2:.1f}")



    del teacher_all

    del teacher_encoder

    gc.collect()

    torch.cuda.empty_cache()



    print()

    print("Stage 2/2: loading ViT-B/16 student encoder and predictor...")

    student_encoder, predictor = torch.hub.load(

        ".",

        "vjepa2_1_vit_base_384",

        source="local",

        pretrained=True,

    )



    student_encoder = student_encoder.to(device=device, dtype=torch.bfloat16).eval()

    predictor = predictor.to(device=device, dtype=torch.bfloat16).eval()



    torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.synchronize()

    student_start = time.perf_counter()



    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):

        context_tokens = student_encoder(

            video,

            masks=[context_indices],

            training=False,

        )



        predicted_target, predicted_context = predictor(

            context_tokens,

            [context_indices],

            [target_indices],

            mod="video",

            mask_index=0,

        )



    torch.cuda.synchronize()

    student_seconds = time.perf_counter() - student_start



    predicted_target = predicted_target.float().cpu()

    predicted_context = predicted_context.float().cpu()



    print(f"Student context-token shape: {tuple(context_tokens.shape)}")

    print(f"Predicted target shape: {tuple(predicted_target.shape)}")

    print(f"Predicted context shape: {tuple(predicted_context.shape)}")

    print(f"Student/predictor seconds: {student_seconds:.3f}")

    print(f"Student/predictor peak GPU MB: {torch.cuda.max_memory_allocated(device) / 1024**2:.1f}")



    if predicted_target.shape != teacher_target.shape:

        raise RuntimeError(

            f"Target shape mismatch: predictor {tuple(predicted_target.shape)} "

            f"vs reference {tuple(teacher_target.shape)}"

        )



    if predicted_context.shape != teacher_context.shape:

        raise RuntimeError(

            f"Context shape mismatch: predictor {tuple(predicted_context.shape)} "

            f"vs reference {tuple(teacher_context.shape)}"

        )



    pred_norm = normalize(predicted_target)

    teacher_target_norm = normalize(teacher_target)

    teacher_context_norm = normalize(teacher_context)

    predicted_context_norm = normalize(predicted_context)



    target_pair_cosine = (pred_norm * teacher_target_norm).sum(dim=-1)[0]

    context_pair_cosine = (predicted_context_norm * teacher_context_norm).sum(dim=-1)[0]



    retrieval = pred_norm[0] @ teacher_target_norm[0].T

    correct_similarity = torch.diag(retrieval)



    ranks = (retrieval > correct_similarity.unsqueeze(1)).sum(dim=1) + 1

    top1 = float((ranks == 1).float().mean())

    top5 = float((ranks <= min(5, retrieval.shape[1])).float().mean())

    mean_rank = float(ranks.float().mean())

    mean_reciprocal_rank = float((1.0 / ranks.float()).mean())



    permutation = torch.randperm(teacher_target_norm.shape[1])

    random_pair_cosine = (

        pred_norm[0] * teacher_target_norm[0, permutation]

    ).sum(dim=-1)



    h = args.target_height

    w = args.target_width

    d = args.target_temporal



    cosine_grid = target_pair_cosine.numpy().reshape(d, h, w)

    rank_grid = ranks.numpy().reshape(d, h, w)



    stem = os.path.splitext(os.path.basename(args.features))[0]

    figure_out = os.path.join(

        args.output_dir,

        f"{stem}_masked_latent_prediction_token_{args.target_token}.png",

    )

    report_out = os.path.join(

        args.output_dir,

        f"{stem}_masked_latent_prediction_token_{args.target_token}.txt",

    )

    npz_out = os.path.join(

        args.output_dir,

        f"{stem}_masked_latent_prediction_token_{args.target_token}.npz",

    )



    frame = video_visual[target_frame_slot]

    x0 = args.target_col * PATCH

    y0 = args.target_row * PATCH

    width_px = args.target_width * PATCH

    height_px = args.target_height * PATCH



    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))



    axes[0].imshow(frame)

    axes[0].add_patch(

        plt.Rectangle(

            (x0, y0),

            width_px,

            height_px,

            fill=False,

            edgecolor="red",

            linewidth=2.5,

        )

    )

    axes[0].set_title(

        f"Masked target block\n"

        f"time token {args.target_token}, frame {target_frame_index}, "

        f"{target_time_seconds:.2f}s",

        fontsize=11,

    )

    axes[0].axis("off")



    image = axes[1].imshow(

        cosine_grid[0],

        cmap="viridis",

        vmin=-1.0,

        vmax=1.0,

        interpolation="nearest",

    )

    axes[1].set_title(

        "Predicted vs reference latent cosine\n"

        "per masked patch",

        fontsize=11,

    )

    axes[1].set_xlabel("Target-block patch column")

    axes[1].set_ylabel("Target-block patch row")

    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)



    matrix = axes[2].imshow(

        retrieval.numpy(),

        cmap="viridis",

        vmin=-1.0,

        vmax=1.0,

        aspect="auto",

    )

    axes[2].plot(

        np.arange(retrieval.shape[1]),

        np.arange(retrieval.shape[0]),

        color="white",

        linewidth=0.7,

    )

    axes[2].set_title(

        f"Target-token retrieval matrix\n"

        f"Top-1={top1:.3f}, Top-5={top5:.3f}, MRR={mean_reciprocal_rank:.3f}",

        fontsize=11,

    )

    axes[2].set_xlabel("Reference target-token index")

    axes[2].set_ylabel("Predicted target-token index")

    fig.colorbar(matrix, ax=axes[2], fraction=0.046, pad=0.04)



    fig.suptitle(

        "V-JEPA 2.1 masked latent-feature prediction diagnostic",

        fontsize=15,

        y=0.99,

    )

    fig.tight_layout()

    fig.savefig(figure_out, dpi=180, bbox_inches="tight")

    plt.close(fig)



    report = [

        "V-JEPA 2.1 masked latent-feature prediction diagnostic",

        f"features: {args.features}",

        f"video: {args.video}",

        f"reference model: V-JEPA 2.1 ViT-G/16",

        f"student model: V-JEPA 2.1 ViT-B/16",

        f"fps: {fps:.6f}",

        "",

        f"target temporal token start: {args.target_token}",

        f"target temporal extent: {args.target_temporal}",

        f"target spatial rows: {args.target_row} to {args.target_row + args.target_height - 1}",

        f"target spatial cols: {args.target_col} to {args.target_col + args.target_width - 1}",

        f"target original frame: {target_frame_index}",

        f"target timestamp: {target_time_seconds:.6f}s",

        f"masked target-token count: {target_indices.shape[1]}",

        f"visible context-token count: {context_indices.shape[1]}",

        "",

        f"teacher target shape: {tuple(teacher_target.shape)}",

        f"predictor target shape: {tuple(predicted_target.shape)}",

        f"mean target cosine: {float(target_pair_cosine.mean()):.6f}",

        f"median target cosine: {float(target_pair_cosine.median()):.6f}",

        f"min target cosine: {float(target_pair_cosine.min()):.6f}",

        f"max target cosine: {float(target_pair_cosine.max()):.6f}",

        f"mean random-pair cosine: {float(random_pair_cosine.mean()):.6f}",

        f"mean context cosine: {float(context_pair_cosine.mean()):.6f}",

        f"target retrieval top-1: {top1:.6f}",

        f"target retrieval top-5: {top5:.6f}",

        f"target retrieval mean rank: {mean_rank:.6f}",

        f"target retrieval MRR: {mean_reciprocal_rank:.6f}",

        f"top-1 random chance: {1.0 / retrieval.shape[1]:.6f}",

        f"top-5 random chance: {min(5, retrieval.shape[1]) / retrieval.shape[1]:.6f}",

        "",

        f"teacher forward seconds: {teacher_seconds:.6f}",

        f"student predictor seconds: {student_seconds:.6f}",

    ]



    with open(report_out, "w", encoding="utf-8") as f:

        f.write("\n".join(report))



    np.savez_compressed(

        npz_out,

        target_indices=target_indices_cpu.numpy(),

        cosine_per_target_patch=cosine_grid,

        rank_per_target_patch=rank_grid,

        retrieval_matrix=retrieval.numpy(),

        target_pair_cosine=target_pair_cosine.numpy(),

        random_pair_cosine=random_pair_cosine.numpy(),

        context_pair_cosine=context_pair_cosine.numpy(),

    )



    print()

    print("Masked latent prediction: success")

    print(f"Mean target cosine: {float(target_pair_cosine.mean()):.4f}")

    print(f"Mean random-pair cosine: {float(random_pair_cosine.mean()):.4f}")

    print(f"Mean context cosine: {float(context_pair_cosine.mean()):.4f}")

    print(f"Target retrieval Top-1: {top1:.4f}")

    print(f"Target retrieval Top-5: {top5:.4f}")

    print(f"Target retrieval MRR: {mean_reciprocal_rank:.4f}")

    print(f"Figure: {figure_out}")

    print(f"Report: {report_out}")

    print(f"Raw metrics: {npz_out}")





if __name__ == "__main__":

    main()

