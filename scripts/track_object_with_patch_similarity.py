
import argparse

import os



import cv2

import matplotlib.pyplot as plt

import numpy as np

import torch

from decord import VideoReader, cpu





PATCH_SIZE = 16

GRID_SIZE = 24

IM_SIZE = PATCH_SIZE * GRID_SIZE





def parse_args():

    parser = argparse.ArgumentParser(

        description="Track a reference object with V-JEPA 2.1 patch-feature similarity."

    )

    parser.add_argument("--features", required=True, help="Path to extracted .npz features.")

    parser.add_argument("--video", required=True, help="Path to the original video.")

    parser.add_argument("--output-dir", default="outputs", help="Directory for outputs.")

    parser.add_argument("--reference-token", type=int, default=12, help="Token used to select the reference object.")

    parser.add_argument("--start-token", type=int, default=12, help="First token to visualize.")

    parser.add_argument("--end-token", type=int, default=19, help="Last token to visualize.")

    parser.add_argument(

        "--roi",

        nargs=4,

        type=int,

        metavar=("X", "Y", "W", "H"),

        help="Optional ROI in model-aligned 384x384 coordinates. If omitted, select interactively.",

    )

    parser.add_argument(

        "--peak-radius",

        type=int,

        default=2,

        help="Patch radius around the highest-similarity patch to visualize.",

    )

    return parser.parse_args()





def normalize_features(x):

    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)





def denormalize_imagenet(video_tensor):

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)

    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)



    frames = video_tensor.detach().cpu().float().permute(1, 2, 3, 0).numpy()

    frames = frames * std[None, None, None, :] + mean[None, None, None, :]

    return np.clip(frames, 0.0, 1.0)





def draw_grid_bgr(image_bgr):

    image = image_bgr.copy()



    for x in range(0, IM_SIZE + 1, PATCH_SIZE):

        cv2.line(image, (x, 0), (x, IM_SIZE), (180, 180, 180), 1)



    for y in range(0, IM_SIZE + 1, PATCH_SIZE):

        cv2.line(image, (0, y), (IM_SIZE, y), (180, 180, 180), 1)



    return image





def roi_to_patch_bounds(x, y, w, h):

    row_start = max(0, int(np.floor(y / PATCH_SIZE)))

    row_end = min(GRID_SIZE, int(np.ceil((y + h) / PATCH_SIZE)))

    col_start = max(0, int(np.floor(x / PATCH_SIZE)))

    col_end = min(GRID_SIZE, int(np.ceil((x + w) / PATCH_SIZE)))



    if row_end <= row_start or col_end <= col_start:

        raise ValueError("Selected ROI does not cover any valid patch.")



    return row_start, row_end, col_start, col_end





def make_overlay(frame, similarity, vmin, vmax):

    normalized = np.clip((similarity - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

    heatmap = plt.get_cmap("turbo")(normalized)[..., :3]

    heatmap = cv2.resize(

        heatmap,

        (frame.shape[1], frame.shape[0]),

        interpolation=cv2.INTER_CUBIC,

    )



    return np.clip(0.48 * frame + 0.52 * heatmap, 0.0, 1.0)





def draw_peak_region(ax, peak_row, peak_col, radius):

    left = max(0, peak_col - radius) * PATCH_SIZE

    top = max(0, peak_row - radius) * PATCH_SIZE

    right = min(GRID_SIZE, peak_col + radius + 1) * PATCH_SIZE

    bottom = min(GRID_SIZE, peak_row + radius + 1) * PATCH_SIZE



    rect = plt.Rectangle(

        (left, top),

        right - left,

        bottom - top,

        fill=False,

        edgecolor="lime",

        linewidth=2.0,

    )

    ax.add_patch(rect)



    ax.plot(

        peak_col * PATCH_SIZE + PATCH_SIZE / 2,

        peak_row * PATCH_SIZE + PATCH_SIZE / 2,

        marker="x",

        color="lime",

        markersize=8,

        markeredgewidth=2,

    )





def main():

    args = parse_args()



    if not os.path.isfile(args.features):

        raise FileNotFoundError(f"Feature file not found: {args.features}")



    if not os.path.isfile(args.video):

        raise FileNotFoundError(f"Video file not found: {args.video}")



    os.makedirs(args.output_dir, exist_ok=True)



    with np.load(args.features) as data:

        sampled_frame_indices = data["sampled_frame_indices"].astype(np.int64)

        fps = float(data["fps"][0])

        patch_tokens = data["patch_tokens"].astype(np.float32)



    token_count = len(sampled_frame_indices) // 2



    if patch_tokens.shape != (1, token_count * GRID_SIZE * GRID_SIZE, 768):

        raise ValueError(

            f"Unexpected patch_tokens shape {patch_tokens.shape}; "

            f"expected (1, {token_count * GRID_SIZE * GRID_SIZE}, 768)."

        )



    if not (0 <= args.reference_token < token_count):

        raise ValueError(f"reference-token must be in [0, {token_count - 1}]")



    if not (0 <= args.start_token <= args.end_token < token_count):

        raise ValueError(f"Invalid token range [{args.start_token}, {args.end_token}]")



    token_grid = patch_tokens.reshape(

        token_count,

        GRID_SIZE,

        GRID_SIZE,

        768,

    )



    reader = VideoReader(args.video, ctx=cpu(0))

    raw_frames = reader.get_batch(sampled_frame_indices).asnumpy()



    print("Loading official V-JEPA preprocessor for model-aligned visualization...")

    processor = torch.hub.load(

        ".",

        "vjepa2_preprocessor",

        source="local",

        crop_size=IM_SIZE,

    )



    processed = processor(raw_frames)[0]



    if processed.ndim != 4:

        raise RuntimeError(f"Unexpected preprocessed tensor shape: {tuple(processed.shape)}")



    if processed.shape[0] != 3:

        raise RuntimeError(

            f"Expected preprocessed tensor shape (3, T, H, W), got {tuple(processed.shape)}"

        )



    model_frames = denormalize_imagenet(processed)



    expected_frames = token_count * 2

    if model_frames.shape[0] != expected_frames:

        raise RuntimeError(

            f"Expected {expected_frames} preprocessed frames, got {model_frames.shape[0]}"

        )



    ref_frame_slot = 2 * args.reference_token

    ref_frame = model_frames[ref_frame_slot]



    if args.roi is None:

        ref_bgr = cv2.cvtColor(

            (ref_frame * 255).astype(np.uint8),

            cv2.COLOR_RGB2BGR,

        )

        ref_grid_bgr = draw_grid_bgr(ref_bgr)



        print()

        print("A selection window will open.")

        print("Draw a box tightly around the large lower blue ball.")

        print("Press ENTER or SPACE to confirm. Press C to cancel.")

        print()



        x, y, w, h = cv2.selectROI(

            "Select reference blue ball in 384x384 model view",

            ref_grid_bgr,

            showCrosshair=True,

            fromCenter=False,

        )

        cv2.destroyAllWindows()



        if w == 0 or h == 0:

            raise RuntimeError("ROI selection cancelled or empty.")

    else:

        x, y, w, h = args.roi



    row_start, row_end, col_start, col_end = roi_to_patch_bounds(x, y, w, h)



    reference_patches = token_grid[

        args.reference_token,

        row_start:row_end,

        col_start:col_end,

    ].reshape(-1, 768)



    reference_feature = normalize_features(reference_patches.mean(axis=0))



    token_indices = list(range(args.start_token, args.end_token + 1))

    similarity_maps = []

    peak_records = []



    for token_index in token_indices:

        target_features = normalize_features(token_grid[token_index])

        similarity = target_features @ reference_feature



        peak_row, peak_col = np.unravel_index(np.argmax(similarity), similarity.shape)

        peak_score = float(similarity[peak_row, peak_col])



        similarity_maps.append(similarity)

        peak_records.append(

            {

                "token": token_index,

                "peak_row": int(peak_row),

                "peak_col": int(peak_col),

                "peak_score": peak_score,

                "mean_score": float(similarity.mean()),

                "std_score": float(similarity.std()),

            }

        )



    all_scores = np.concatenate([x.reshape(-1) for x in similarity_maps])

    vmin = float(np.percentile(all_scores, 2))

    vmax = float(np.percentile(all_scores, 98))



    stem = os.path.splitext(os.path.basename(args.features))[0]

    reference_out = os.path.join(

        args.output_dir,

        f"{stem}_reference_object_token_{args.reference_token}.png",

    )

    track_out = os.path.join(

        args.output_dir,

        f"{stem}_object_track_tokens_{args.start_token}_{args.end_token}.png",

    )

    report_out = os.path.join(

        args.output_dir,

        f"{stem}_object_track_tokens_{args.start_token}_{args.end_token}.txt",

    )



    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(ref_frame)

    ax.axis("off")

    ax.set_title(

        f"Reference object | token {args.reference_token} | "

        f"patch rows {row_start}:{row_end - 1}, cols {col_start}:{col_end - 1}",

        fontsize=12,

    )



    ref_rect = plt.Rectangle(

        (x, y),

        w,

        h,

        fill=False,

        edgecolor="red",

        linewidth=2.5,

    )

    ax.add_patch(ref_rect)



    for grid_pos in range(0, IM_SIZE + 1, PATCH_SIZE):

        ax.axvline(grid_pos, color="white", linewidth=0.35, alpha=0.55)

        ax.axhline(grid_pos, color="white", linewidth=0.35, alpha=0.55)



    fig.tight_layout()

    fig.savefig(reference_out, dpi=180, bbox_inches="tight")

    plt.close(fig)



    rows = len(token_indices)

    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.3 * rows))



    if rows == 1:

        axes = np.expand_dims(axes, axis=0)



    report = [

        "V-JEPA 2.1 object-level patch similarity tracking",

        f"features: {args.features}",

        f"video: {args.video}",

        f"fps: {fps:.6f}",

        f"reference token: {args.reference_token}",

        f"reference ROI xywh: [{x}, {y}, {w}, {h}]",

        f"reference patch rows: {row_start} to {row_end - 1}",

        f"reference patch cols: {col_start} to {col_end - 1}",

        f"reference patch count: {len(reference_patches)}",

        f"visualization similarity range: [{vmin:.6f}, {vmax:.6f}]",

        "",

    ]



    for row_idx, (token_index, similarity, record) in enumerate(

        zip(token_indices, similarity_maps, peak_records)

    ):

        frame_slot = 2 * token_index

        frame_idx = int(sampled_frame_indices[frame_slot])

        timestamp = frame_idx / fps

        frame = model_frames[frame_slot]

        overlay = make_overlay(frame, similarity, vmin, vmax)



        axes[row_idx, 0].imshow(frame)

        axes[row_idx, 0].axis("off")

        axes[row_idx, 0].set_title(

            f"token {token_index} | frame {frame_idx} | {timestamp:.2f}s",

            fontsize=10,

        )

        draw_peak_region(

            axes[row_idx, 0],

            record["peak_row"],

            record["peak_col"],

            args.peak_radius,

        )



        if token_index == args.reference_token:

            ref_rect = plt.Rectangle(

                (x, y),

                w,

                h,

                fill=False,

                edgecolor="red",

                linewidth=2.5,

            )

            axes[row_idx, 0].add_patch(ref_rect)



        im = axes[row_idx, 1].imshow(

            similarity,

            cmap="turbo",

            vmin=vmin,

            vmax=vmax,

            interpolation="nearest",

        )

        axes[row_idx, 1].set_title(

            f"patch similarity | peak={record['peak_score']:.4f}",

            fontsize=10,

        )

        axes[row_idx, 1].set_xlabel("patch column")

        axes[row_idx, 1].set_ylabel("patch row")

        axes[row_idx, 1].plot(

            record["peak_col"],

            record["peak_row"],

            marker="x",

            color="white",

            markersize=8,

            markeredgewidth=2,

        )



        axes[row_idx, 2].imshow(overlay)

        axes[row_idx, 2].axis("off")

        axes[row_idx, 2].set_title(

            f"similarity overlay | peak patch=({record['peak_row']}, {record['peak_col']})",

            fontsize=10,

        )

        draw_peak_region(

            axes[row_idx, 2],

            record["peak_row"],

            record["peak_col"],

            args.peak_radius,

        )



        report.extend(

            [

                f"token {token_index}",

                f"  source frame: {frame_idx}",

                f"  timestamp: {timestamp:.6f}s",

                f"  peak patch row/col: [{record['peak_row']}, {record['peak_col']}]",

                f"  peak patch center: "

                f"[{record['peak_col'] * PATCH_SIZE + PATCH_SIZE / 2:.1f}, "

                f"{record['peak_row'] * PATCH_SIZE + PATCH_SIZE / 2:.1f}]",

                f"  peak cosine similarity: {record['peak_score']:.6f}",

                f"  mean cosine similarity: {record['mean_score']:.6f}",

                f"  similarity std: {record['std_score']:.6f}",

                "",

            ]

        )



    fig.suptitle(

        "V-JEPA 2.1 object-level patch similarity tracking",

        fontsize=16,

        y=0.998,

    )

    fig.tight_layout()



    colorbar = fig.colorbar(

        im,

        ax=axes[:, 1],

        fraction=0.025,

        pad=0.02,

    )

    colorbar.set_label("Cosine similarity to reference blue-ball feature")



    fig.savefig(track_out, dpi=180, bbox_inches="tight")

    plt.close(fig)



    with open(report_out, "w", encoding="utf-8") as f:

        f.write("\n".join(report))



    print()

    print("Object-level patch tracking: success")

    print(f"Reference visualization: {reference_out}")

    print(f"Tracking visualization: {track_out}")

    print(f"Text report: {report_out}")

    print()

    print("Reference patch region:")

    print(f"rows {row_start}:{row_end - 1}, cols {col_start}:{col_end - 1}")

    print()

    print("Peak similarity locations:")

    for record in peak_records:

        print(

            f"token {record['token']:2d}: "

            f"peak=({record['peak_row']:2d}, {record['peak_col']:2d}), "

            f"cos={record['peak_score']:.4f}"

        )





if __name__ == "__main__":

    main()

