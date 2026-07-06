
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

MODE_ORDER = ["full", "spatial_only", "temporal_bi", "past_only"]

MODE_LABELS = {

    "full": "Full context",

    "spatial_only": "Spatial-only",

    "temporal_bi": "Bidirectional temporal-only",

    "past_only": "Past-only temporal",

}





def parse_args():

    parser = argparse.ArgumentParser(

        description="V-JEPA 2.1 masked latent prediction context ablation."

    )

    parser.add_argument("--features", required=True)

    parser.add_argument("--video", required=True)

    parser.add_argument("--output-dir", default="outputs")

    parser.add_argument("--target-token", type=int, default=16)

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





def build_target_indices(target_token, row, height, col, width):

    indices = []



    for patch_row in range(row, row + height):

        for patch_col in range(col, col + width):

            index = target_token * TOKENS_PER_TIME + patch_row * GRID + patch_col

            indices.append(index)



    return torch.tensor(indices, dtype=torch.long)





def build_context_indices(mode, total_tokens, target_token, target_indices):

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

        raise ValueError(f"Unsupported mode: {mode}")



    context_indices = all_indices[keep]



    if len(context_indices) == 0:

        raise RuntimeError(f"Mode {mode} has no visible context tokens.")



    return context_indices





def target_coordinates(row, height, col, width):

    rows = []

    cols = []



    for patch_row in range(row, row + height):

        for patch_col in range(col, col + width):

            rows.append(patch_row)

            cols.append(patch_col)



    return np.asarray(rows), np.asarray(cols)





def evaluate_prediction(predicted_target, teacher_target, target_rows, target_cols):

    predicted_norm = normalize(predicted_target)

    teacher_norm = normalize(teacher_target)



    pair_cosine = (predicted_norm * teacher_norm).sum(dim=-1)[0]

    retrieval = predicted_norm[0] @ teacher_norm[0].T



    correct_similarity = torch.diag(retrieval)

    ranks = (retrieval > correct_similarity.unsqueeze(1)).sum(dim=1) + 1



    top1 = float((ranks == 1).float().mean())

    top5 = float((ranks <= min(5, retrieval.shape[1])).float().mean())

    mean_rank = float(ranks.float().mean())

    mrr = float((1.0 / ranks.float()).mean())



    shifted_teacher = torch.roll(teacher_norm[0], shifts=1, dims=0)

    random_pair_cosine = (predicted_norm[0] * shifted_teacher).sum(dim=-1)



    best_match = retrieval.argmax(dim=1).cpu().numpy()



    row_error = target_rows[best_match] - target_rows

    col_error = target_cols[best_match] - target_cols



    chebyshev_error = np.maximum(np.abs(row_error), np.abs(col_error))

    euclidean_error = np.sqrt(row_error ** 2 + col_error ** 2)



    neighbor_at_1 = float((chebyshev_error <= 1).mean())

    mean_spatial_error = float(euclidean_error.mean())



    return {

        "pair_cosine": pair_cosine.cpu().numpy(),

        "random_pair_cosine": random_pair_cosine.cpu().numpy(),

        "retrieval": retrieval.cpu().numpy(),

        "ranks": ranks.cpu().numpy(),

        "top1": top1,

        "top5": top5,

        "mean_rank": mean_rank,

        "mrr": mrr,

        "neighbor_at_1": neighbor_at_1,

        "mean_spatial_error": mean_spatial_error,

    }





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

    total_tokens = temporal_tokens * TOKENS_PER_TIME



    if args.target_token < 0 or args.target_token >= temporal_tokens:

        raise ValueError("target-token is outside the temporal token range.")



    if args.target_row < 0 or args.target_row + args.target_height > GRID:

        raise ValueError("Target rows are outside the 24x24 patch grid.")



    if args.target_col < 0 or args.target_col + args.target_width > GRID:

        raise ValueError("Target columns are outside the 24x24 patch grid.")



    target_indices_cpu = build_target_indices(

        args.target_token,

        args.target_row,

        args.target_height,

        args.target_col,

        args.target_width,

    )



    target_rows, target_cols = target_coordinates(

        args.target_row,

        args.target_height,

        args.target_col,

        args.target_width,

    )



    print(f"Temporal token grid: {temporal_tokens} x {GRID} x {GRID}")

    print(f"Total patch tokens: {total_tokens}")

    print(f"Masked target tokens: {len(target_indices_cpu)}")



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

    target_time = target_frame_index / fps



    target_indices_gpu = target_indices_cpu.unsqueeze(0).to(device)



    print()

    print("Stage 1/2: extracting ViT-G teacher targets...")



    teacher_encoder, unused_teacher_predictor = torch.hub.load(

        ".",

        "vjepa2_1_vit_gigantic_384",

        source="local",

        pretrained=True,

    )



    del unused_teacher_predictor

    gc.collect()



    teacher_encoder = teacher_encoder.to(

        device=device,

        dtype=torch.bfloat16,

    ).eval()



    torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.synchronize()

    teacher_start = time.perf_counter()



    with torch.inference_mode(), torch.autocast(

        device_type="cuda",

        dtype=torch.bfloat16,

    ):

        teacher_all = teacher_encoder(video)



    torch.cuda.synchronize()

    teacher_seconds = time.perf_counter() - teacher_start



    teacher_target = teacher_all[:, target_indices_cpu].float().cpu()



    print(f"Teacher full-token shape: {tuple(teacher_all.shape)}")

    print(f"Teacher target shape: {tuple(teacher_target.shape)}")

    print(f"Teacher forward seconds: {teacher_seconds:.3f}")

    print(

        "Teacher peak GPU MB: "

        f"{torch.cuda.max_memory_allocated(device) / 1024**2:.1f}"

    )



    del teacher_all

    del teacher_encoder

    gc.collect()

    torch.cuda.empty_cache()



    print()

    print("Stage 2/2: evaluating context ablations with ViT-B + predictor...")



    student_encoder, predictor = torch.hub.load(

        ".",

        "vjepa2_1_vit_base_384",

        source="local",

        pretrained=True,

    )



    student_encoder = student_encoder.to(

        device=device,

        dtype=torch.bfloat16,

    ).eval()



    predictor = predictor.to(

        device=device,

        dtype=torch.bfloat16,

    ).eval()



    results = {}



    for mode_index, mode in enumerate(MODE_ORDER):

        context_indices_cpu = build_context_indices(

            mode,

            total_tokens,

            args.target_token,

            target_indices_cpu,

        )



        context_indices_gpu = context_indices_cpu.unsqueeze(0).to(device)



        torch.cuda.reset_peak_memory_stats(device)

        torch.cuda.synchronize()

        start = time.perf_counter()



        with torch.inference_mode(), torch.autocast(

            device_type="cuda",

            dtype=torch.bfloat16,

        ):

            context_tokens = student_encoder(

                video,

                masks=[context_indices_gpu],

                training=False,

            )



            predicted_target, _ = predictor(

                context_tokens,

                [context_indices_gpu],

                [target_indices_gpu],

                mod="video",

                mask_index=0,

            )



        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start



        predicted_target = predicted_target.float().cpu()



        metrics = evaluate_prediction(

            predicted_target,

            teacher_target,

            target_rows,

            target_cols,

        )



        metrics["visible_tokens"] = int(len(context_indices_cpu))

        metrics["elapsed_seconds"] = float(elapsed)

        metrics["peak_gpu_mb"] = float(

            torch.cuda.max_memory_allocated(device) / 1024**2

        )

        metrics["context_shape"] = tuple(context_tokens.shape)

        metrics["prediction_shape"] = tuple(predicted_target.shape)



        results[mode] = metrics



        print(

            f"{mode:16s} | visible={metrics['visible_tokens']:5d} | "

            f"cos={metrics['pair_cosine'].mean():.4f} | "

            f"top1={metrics['top1']:.4f} | "

            f"top5={metrics['top5']:.4f} | "

            f"MRR={metrics['mrr']:.4f}"

        )



        del context_tokens

        del predicted_target

        torch.cuda.empty_cache()



    del student_encoder

    del predictor

    gc.collect()

    torch.cuda.empty_cache()



    cosine_grids = {

        mode: results[mode]["pair_cosine"].reshape(

            args.target_height,

            args.target_width,

        )

        for mode in MODE_ORDER

    }



    all_cosines = np.concatenate(

        [cosine_grids[mode].reshape(-1) for mode in MODE_ORDER]

    )



    cosine_vmin = float(np.percentile(all_cosines, 1))

    cosine_vmax = float(np.percentile(all_cosines, 99))



    stem = os.path.splitext(os.path.basename(args.features))[0]



    cosine_figure = os.path.join(

        args.output_dir,

        f"{stem}_context_ablation_cosine_token_{args.target_token}.png",

    )



    retrieval_figure = os.path.join(

        args.output_dir,

        f"{stem}_context_ablation_retrieval_token_{args.target_token}.png",

    )



    report_out = os.path.join(

        args.output_dir,

        f"{stem}_context_ablation_token_{args.target_token}.txt",

    )



    npz_out = os.path.join(

        args.output_dir,

        f"{stem}_context_ablation_token_{args.target_token}.npz",

    )



    frame = video_visual[target_frame_slot]

    x0 = args.target_col * PATCH

    y0 = args.target_row * PATCH

    width_px = args.target_width * PATCH

    height_px = args.target_height * PATCH



    fig, axes = plt.subplots(2, 3, figsize=(17, 10))



    axes[0, 0].imshow(frame)

    axes[0, 0].add_patch(

        plt.Rectangle(

            (x0, y0),

            width_px,

            height_px,

            fill=False,

            edgecolor="red",

            linewidth=2.5,

        )

    )

    axes[0, 0].set_title(

        f"Shared masked target\n"

        f"time token {args.target_token}, frame {target_frame_index}, "

        f"{target_time:.2f}s",

        fontsize=11,

    )

    axes[0, 0].axis("off")



    image = None

    for axis, mode in zip(

        [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]],

        MODE_ORDER,

    ):

        image = axis.imshow(

            cosine_grids[mode],

            cmap="viridis",

            vmin=cosine_vmin,

            vmax=cosine_vmax,

            interpolation="nearest",

        )



        axis.set_title(

            f"{MODE_LABELS[mode]}\n"

            f"mean cosine={results[mode]['pair_cosine'].mean():.4f}",

            fontsize=10,

        )

        axis.set_xlabel("Target-block patch column")

        axis.set_ylabel("Target-block patch row")



    summary_axis = axes[1, 2]

    summary_axis.axis("off")



    summary_lines = [

        "Context ablation summary",

        "",

        "Mode                 Mean cos   Top-1   Top-5   MRR",

    ]



    for mode in MODE_ORDER:

        summary_lines.append(

            f"{mode:20s} "

            f"{results[mode]['pair_cosine'].mean():.4f}    "

            f"{results[mode]['top1']:.3f}   "

            f"{results[mode]['top5']:.3f}   "

            f"{results[mode]['mrr']:.3f}"

        )



    summary_lines.extend(

        [

            "",

            "Top-1 random chance: "

            f"{1.0 / len(target_indices_cpu):.4f}",

            "Top-5 random chance: "

            f"{min(5, len(target_indices_cpu)) / len(target_indices_cpu):.4f}",

        ]

    )



    summary_axis.text(

        0.02,

        0.96,

        "\n".join(summary_lines),

        va="top",

        ha="left",

        fontsize=10,

        family="monospace",

    )



    fig.colorbar(

        image,

        ax=[axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]],

        fraction=0.03,

        pad=0.02,

        label="Predicted vs reference latent cosine",

    )



    fig.suptitle(

        "V-JEPA 2.1 context ablation: masked latent prediction",

        fontsize=16,

        y=0.995,

    )



    fig.tight_layout()

    fig.savefig(cosine_figure, dpi=180, bbox_inches="tight")

    plt.close(fig)



    all_retrieval = np.concatenate(

        [results[mode]["retrieval"].reshape(-1) for mode in MODE_ORDER]

    )



    retrieval_vmin = float(np.percentile(all_retrieval, 1))

    retrieval_vmax = float(np.percentile(all_retrieval, 99))



    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    axes = axes.reshape(-1)



    matrix_image = None



    for axis, mode in zip(axes, MODE_ORDER):

        matrix_image = axis.imshow(

            results[mode]["retrieval"],

            cmap="viridis",

            vmin=retrieval_vmin,

            vmax=retrieval_vmax,

            aspect="auto",

        )



        n = results[mode]["retrieval"].shape[0]



        axis.plot(

            np.arange(n),

            np.arange(n),

            color="white",

            linewidth=0.8,

        )



        axis.set_title(

            f"{MODE_LABELS[mode]}\n"

            f"Top-1={results[mode]['top1']:.3f}, "

            f"Top-5={results[mode]['top5']:.3f}, "

            f"MRR={results[mode]['mrr']:.3f}",

            fontsize=10,

        )



        axis.set_xlabel("Reference target-token index")

        axis.set_ylabel("Predicted target-token index")



    fig.colorbar(

        matrix_image,

        ax=axes.tolist(),

        fraction=0.03,

        pad=0.02,

        label="Predicted-to-reference latent cosine",

    )



    fig.suptitle(

        "V-JEPA 2.1 context ablation: target-token retrieval",

        fontsize=16,

        y=0.995,

    )



    fig.tight_layout()

    fig.savefig(retrieval_figure, dpi=180, bbox_inches="tight")

    plt.close(fig)



    report = [

        "V-JEPA 2.1 context ablation for masked latent prediction",

        f"features: {args.features}",

        f"video: {args.video}",

        f"teacher: V-JEPA 2.1 ViT-G/16",

        f"student: V-JEPA 2.1 ViT-B/16 + predictor",

        f"fps: {fps:.6f}",

        "",

        f"target temporal token: {args.target_token}",

        f"target original frame: {target_frame_index}",

        f"target timestamp: {target_time:.6f}s",

        f"target spatial rows: {args.target_row} to {args.target_row + args.target_height - 1}",

        f"target spatial cols: {args.target_col} to {args.target_col + args.target_width - 1}",

        f"masked target-token count: {len(target_indices_cpu)}",

        "",

        f"teacher forward seconds: {teacher_seconds:.6f}",

        "",

    ]



    for mode in MODE_ORDER:

        metric = results[mode]



        report.extend(

            [

                f"[{mode}]",

                f"  visible context tokens: {metric['visible_tokens']}",

                f"  student context shape: {metric['context_shape']}",

                f"  predictor target shape: {metric['prediction_shape']}",

                f"  mean target cosine: {metric['pair_cosine'].mean():.6f}",

                f"  median target cosine: {np.median(metric['pair_cosine']):.6f}",

                f"  min target cosine: {metric['pair_cosine'].min():.6f}",

                f"  max target cosine: {metric['pair_cosine'].max():.6f}",

                f"  mean random-pair cosine: {metric['random_pair_cosine'].mean():.6f}",

                f"  retrieval top-1: {metric['top1']:.6f}",

                f"  retrieval top-5: {metric['top5']:.6f}",

                f"  retrieval mean rank: {metric['mean_rank']:.6f}",

                f"  retrieval MRR: {metric['mrr']:.6f}",

                f"  top-1 within 1-patch neighborhood: {metric['neighbor_at_1']:.6f}",

                f"  mean best-match spatial error: {metric['mean_spatial_error']:.6f}",

                f"  elapsed seconds: {metric['elapsed_seconds']:.6f}",

                f"  peak GPU MB: {metric['peak_gpu_mb']:.6f}",

                "",

            ]

        )



    with open(report_out, "w", encoding="utf-8") as file:

        file.write("\n".join(report))



    np.savez_compressed(

        npz_out,

        mode_names=np.asarray(MODE_ORDER, dtype="U32"),

        target_indices=target_indices_cpu.numpy(),

        target_patch_rows=target_rows,

        target_patch_cols=target_cols,

        cosine_grids=np.stack([cosine_grids[mode] for mode in MODE_ORDER]),

        retrieval_matrices=np.stack(

            [results[mode]["retrieval"] for mode in MODE_ORDER]

        ),

        ranks=np.stack([results[mode]["ranks"] for mode in MODE_ORDER]),

        mean_target_cosines=np.asarray(

            [results[mode]["pair_cosine"].mean() for mode in MODE_ORDER],

            dtype=np.float32,

        ),

        mean_random_cosines=np.asarray(

            [results[mode]["random_pair_cosine"].mean() for mode in MODE_ORDER],

            dtype=np.float32,

        ),

        top1=np.asarray(

            [results[mode]["top1"] for mode in MODE_ORDER],

            dtype=np.float32,

        ),

        top5=np.asarray(

            [results[mode]["top5"] for mode in MODE_ORDER],

            dtype=np.float32,

        ),

        mean_rank=np.asarray(

            [results[mode]["mean_rank"] for mode in MODE_ORDER],

            dtype=np.float32,

        ),

        mrr=np.asarray(

            [results[mode]["mrr"] for mode in MODE_ORDER],

            dtype=np.float32,

        ),

        neighbor_at_1=np.asarray(

            [results[mode]["neighbor_at_1"] for mode in MODE_ORDER],

            dtype=np.float32,

        ),

        mean_spatial_error=np.asarray(

            [results[mode]["mean_spatial_error"] for mode in MODE_ORDER],

            dtype=np.float32,

        ),

        visible_context_tokens=np.asarray(

            [results[mode]["visible_tokens"] for mode in MODE_ORDER],

            dtype=np.int64,

        ),

    )



    print()

    print("Context ablation: success")

    print()



    for mode in MODE_ORDER:

        metric = results[mode]



        print(

            f"{mode:16s} | "

            f"visible={metric['visible_tokens']:5d} | "

            f"cos={metric['pair_cosine'].mean():.4f} | "

            f"top1={metric['top1']:.4f} | "

            f"top5={metric['top5']:.4f} | "

            f"MRR={metric['mrr']:.4f} | "

            f"near1={metric['neighbor_at_1']:.4f}"

        )



    print()

    print(f"Cosine figure: {cosine_figure}")

    print(f"Retrieval figure: {retrieval_figure}")

    print(f"Report: {report_out}")

    print(f"Raw metrics: {npz_out}")





if __name__ == "__main__":

    main()

