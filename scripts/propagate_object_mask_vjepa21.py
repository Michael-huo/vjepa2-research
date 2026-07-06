
import argparse

import os

from collections import deque



import cv2

import matplotlib.pyplot as plt

import numpy as np

import torch

from decord import VideoReader, cpu





PATCH_SIZE = 16

GRID_SIZE = 24

IMAGE_SIZE = PATCH_SIZE * GRID_SIZE

FEATURE_DIM = 768





def parse_args():

    parser = argparse.ArgumentParser(

        description="Propagate an object mask using V-JEPA 2.1 dense patch features."

    )

    parser.add_argument("--features", required=True, help="Path to extracted .npz features.")

    parser.add_argument("--video", required=True, help="Path to source video.")

    parser.add_argument("--output-dir", default="outputs", help="Directory for output files.")

    parser.add_argument("--reference-token", type=int, default=12)

    parser.add_argument("--start-token", type=int, default=12)

    parser.add_argument("--end-token", type=int, default=19)

    parser.add_argument(

        "--roi",

        nargs=4,

        type=int,

        default=[48, 272, 64, 64],

        metavar=("X", "Y", "W", "H"),

        help="Reference ROI in model-aligned 384x384 coordinates.",

    )

    parser.add_argument("--spatial-sigma", type=float, default=6.0)

    parser.add_argument("--max-jump", type=float, default=11.0)

    parser.add_argument("--spatial-weight", type=float, default=0.08)

    parser.add_argument("--reference-weight", type=float, default=0.75)

    parser.add_argument("--memory-frames", type=int, default=3)

    parser.add_argument("--top-k-update", type=int, default=12)

    parser.add_argument("--mask-quantile", type=float, default=0.88)

    parser.add_argument("--peak-margin", type=float, default=0.08)

    parser.add_argument("--min-mask-patches", type=int, default=5)

    parser.add_argument("--max-mask-patches", type=int, default=45)

    return parser.parse_args()





def normalize_features(features):

    return features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)





def denormalize_imagenet(video_tensor):

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)

    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)



    frames = video_tensor.detach().cpu().float().permute(1, 2, 3, 0).numpy()

    frames = frames * std[None, None, None, :] + mean[None, None, None, :]

    return np.clip(frames, 0.0, 1.0)





def roi_to_mask(x, y, w, h):

    row_start = max(0, int(np.floor(y / PATCH_SIZE)))

    row_end = min(GRID_SIZE, int(np.ceil((y + h) / PATCH_SIZE)))

    col_start = max(0, int(np.floor(x / PATCH_SIZE)))

    col_end = min(GRID_SIZE, int(np.ceil((x + w) / PATCH_SIZE)))



    if row_end <= row_start or col_end <= col_start:

        raise ValueError("ROI does not cover any valid patch.")



    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    mask[row_start:row_end, col_start:col_end] = True

    return mask, row_start, row_end, col_start, col_end





def connected_component(binary_mask, seed_row, seed_col):

    if not binary_mask[seed_row, seed_col]:

        return np.zeros_like(binary_mask, dtype=bool)



    output = np.zeros_like(binary_mask, dtype=bool)

    queue = [(int(seed_row), int(seed_col))]

    output[seed_row, seed_col] = True



    while queue:

        row, col = queue.pop(0)



        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):

            nr, nc = row + dr, col + dc



            if (

                0 <= nr < GRID_SIZE

                and 0 <= nc < GRID_SIZE

                and binary_mask[nr, nc]

                and not output[nr, nc]

            ):

                output[nr, nc] = True

                queue.append((nr, nc))



    return output





def build_mask_from_scores(

    appearance_scores,

    valid_region,

    peak_row,

    peak_col,

    quantile,

    peak_margin,

    min_patches,

    max_patches,

):

    valid_scores = appearance_scores[valid_region]



    if valid_scores.size == 0:

        raise RuntimeError("No valid patches remain in the spatial search region.")



    peak_score = float(appearance_scores[peak_row, peak_col])



    for relaxation in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10):

        threshold = max(

            float(np.quantile(valid_scores, quantile)) - relaxation,

            peak_score - peak_margin - relaxation,

        )



        candidates = (appearance_scores >= threshold) & valid_region

        component = connected_component(candidates, peak_row, peak_col)



        if component.sum() >= min_patches:

            break

    else:

        component = np.zeros_like(appearance_scores, dtype=bool)

        component[peak_row, peak_col] = True



    if component.sum() > max_patches:

        component_scores = appearance_scores.copy()

        component_scores[~component] = -np.inf



        top_indices = np.argpartition(

            component_scores.ravel(),

            -max_patches,

        )[-max_patches:]



        limited = np.zeros_like(component, dtype=bool)

        limited.flat[top_indices] = True

        component = limited



    return component





def weighted_centroid(mask, scores):

    rows, cols = np.indices(mask.shape)

    weights = np.where(mask, scores, 0.0)



    min_selected = weights[mask].min() if mask.any() else 0.0

    weights = np.where(mask, weights - min_selected + 1e-4, 0.0)



    total = weights.sum()



    if total <= 0:

        selected = np.argwhere(mask)

        return selected.mean(axis=0)



    return np.array(

        [

            float((rows * weights).sum() / total),

            float((cols * weights).sum() / total),

        ]

    )





def patch_mask_to_image(mask, size=IMAGE_SIZE):

    return cv2.resize(

        mask.astype(np.float32),

        (size, size),

        interpolation=cv2.INTER_NEAREST,

    )





def draw_mask_and_centroid(ax, mask, centroid, color="red"):

    mask_up = patch_mask_to_image(mask)

    ax.contour(mask_up, levels=[0.5], colors=color, linewidths=1.6)



    cx = centroid[1] * PATCH_SIZE + PATCH_SIZE / 2

    cy = centroid[0] * PATCH_SIZE + PATCH_SIZE / 2



    ax.plot(

        cx,

        cy,

        marker="x",

        color="lime",

        markersize=8,

        markeredgewidth=2,

    )





def make_similarity_overlay(frame, scores, vmin, vmax):

    scaled = np.clip((scores - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

    heatmap = plt.get_cmap("turbo")(scaled)[..., :3]



    heatmap = cv2.resize(

        heatmap,

        (frame.shape[1], frame.shape[0]),

        interpolation=cv2.INTER_CUBIC,

    )



    return np.clip(0.52 * frame + 0.48 * heatmap, 0.0, 1.0)





def main():

    args = parse_args()



    if not os.path.isfile(args.features):

        raise FileNotFoundError(f"Feature file not found: {args.features}")



    if not os.path.isfile(args.video):

        raise FileNotFoundError(f"Video file not found: {args.video}")



    os.makedirs(args.output_dir, exist_ok=True)



    with np.load(args.features) as data:

        sampled_indices = data["sampled_frame_indices"].astype(np.int64)

        fps = float(data["fps"][0])

        patch_tokens = data["patch_tokens"].astype(np.float32)



    token_count = len(sampled_indices) // 2



    expected_shape = (1, token_count * GRID_SIZE * GRID_SIZE, FEATURE_DIM)

    if patch_tokens.shape != expected_shape:

        raise ValueError(

            f"Unexpected patch_tokens shape {patch_tokens.shape}; "

            f"expected {expected_shape}."

        )



    if not (0 <= args.reference_token < token_count):

        raise ValueError("reference-token out of range.")



    if not (0 <= args.start_token <= args.end_token < token_count):

        raise ValueError("Invalid start-token/end-token range.")



    if not (args.start_token <= args.reference_token <= args.end_token):

        raise ValueError("reference-token must lie inside the selected token range.")



    token_grid = patch_tokens.reshape(

        token_count,

        GRID_SIZE,

        GRID_SIZE,

        FEATURE_DIM,

    )

    token_grid = normalize_features(token_grid)



    x, y, w, h = args.roi

    reference_mask, r0, r1, c0, c1 = roi_to_mask(x, y, w, h)



    reference_features = token_grid[args.reference_token][reference_mask]

    reference_features = normalize_features(reference_features)



    reference_centroid = weighted_centroid(

        reference_mask,

        np.ones_like(reference_mask, dtype=np.float32),

    )



    reader = VideoReader(args.video, ctx=cpu(0))

    raw_frames = reader.get_batch(sampled_indices).asnumpy()



    print("Loading official V-JEPA preprocessor for model-aligned visualization...")

    processor = torch.hub.load(

        ".",

        "vjepa2_preprocessor",

        source="local",

        crop_size=IMAGE_SIZE,

    )



    processed = processor(raw_frames)[0]

    model_frames = denormalize_imagenet(processed)



    if model_frames.shape[0] != token_count * 2:

        raise RuntimeError("Unexpected number of preprocessed frames.")



    row_coords, col_coords = np.indices((GRID_SIZE, GRID_SIZE))

    token_indices = list(range(args.start_token, args.end_token + 1))



    memory_bank = deque(maxlen=args.memory_frames)

    memory_bank.append(reference_features)



    masks = []

    appearance_maps = []

    fused_maps = []

    centroids = []

    records = []



    previous_centroid = reference_centroid.copy()



    for token_index in token_indices:

        target = token_grid[token_index]



        reference_similarity = np.max(

            target.reshape(-1, FEATURE_DIM) @ reference_features.T,

            axis=1,

        ).reshape(GRID_SIZE, GRID_SIZE)



        memory_features = np.concatenate(list(memory_bank), axis=0)

        memory_similarity = np.max(

            target.reshape(-1, FEATURE_DIM) @ memory_features.T,

            axis=1,

        ).reshape(GRID_SIZE, GRID_SIZE)



        appearance = (

            args.reference_weight * reference_similarity

            + (1.0 - args.reference_weight) * memory_similarity

        )



        if token_index == args.reference_token:

            mask = reference_mask.copy()

            centroid = reference_centroid.copy()

            fused = appearance.copy()

            peak_row, peak_col = np.unravel_index(

                np.argmax(np.where(mask, appearance, -np.inf)),

                appearance.shape,

            )

            valid_region = np.ones_like(mask, dtype=bool)

        else:

            distance = np.sqrt(

                (row_coords - previous_centroid[0]) ** 2

                + (col_coords - previous_centroid[1]) ** 2

            )



            valid_region = distance <= args.max_jump

            spatial_prior = np.exp(

                -(distance ** 2) / (2.0 * args.spatial_sigma ** 2)

            )



            fused = appearance + args.spatial_weight * spatial_prior

            gated_fused = np.where(valid_region, fused, -np.inf)



            peak_row, peak_col = np.unravel_index(

                np.argmax(gated_fused),

                gated_fused.shape,

            )



            mask = build_mask_from_scores(

                appearance_scores=appearance,

                valid_region=valid_region,

                peak_row=peak_row,

                peak_col=peak_col,

                quantile=args.mask_quantile,

                peak_margin=args.peak_margin,

                min_patches=args.min_mask_patches,

                max_patches=args.max_mask_patches,

            )



            centroid = weighted_centroid(mask, appearance)



            selected_indices = np.argwhere(mask)

            selected_scores = appearance[mask]



            top_count = min(args.top_k_update, len(selected_indices))

            top_order = np.argsort(selected_scores)[-top_count:]

            update_indices = selected_indices[top_order]



            update_features = target[

                update_indices[:, 0],

                update_indices[:, 1],

            ]



            memory_bank.append(normalize_features(update_features))



        peak_app = float(appearance[peak_row, peak_col])

        peak_ref = float(reference_similarity[peak_row, peak_col])

        mask_mean = float(appearance[mask].mean()) if mask.any() else float("nan")



        masks.append(mask)

        appearance_maps.append(appearance)

        fused_maps.append(fused)

        centroids.append(centroid)



        frame_idx = int(sampled_indices[2 * token_index])

        records.append(

            {

                "token": token_index,

                "frame": frame_idx,

                "time": frame_idx / fps,

                "peak_row": int(peak_row),

                "peak_col": int(peak_col),

                "peak_appearance": peak_app,

                "peak_reference_similarity": peak_ref,

                "mask_size": int(mask.sum()),

                "mask_mean_similarity": mask_mean,

                "centroid_row": float(centroid[0]),

                "centroid_col": float(centroid[1]),

            }

        )



        previous_centroid = centroid.copy()



    all_scores = np.concatenate(

        [score.reshape(-1) for score in appearance_maps]

    )

    vis_min = float(np.percentile(all_scores, 2))

    vis_max = float(np.percentile(all_scores, 98))



    stem = os.path.splitext(os.path.basename(args.features))[0]



    reference_out = os.path.join(

        args.output_dir,

        f"{stem}_propagation_reference_token_{args.reference_token}.png",

    )

    propagation_out = os.path.join(

        args.output_dir,

        f"{stem}_object_mask_propagation_tokens_{args.start_token}_{args.end_token}.png",

    )

    trajectory_out = os.path.join(

        args.output_dir,

        f"{stem}_object_mask_trajectory_tokens_{args.start_token}_{args.end_token}.png",

    )

    report_out = os.path.join(

        args.output_dir,

        f"{stem}_object_mask_propagation_tokens_{args.start_token}_{args.end_token}.txt",

    )

    npz_out = os.path.join(

        args.output_dir,

        f"{stem}_object_mask_propagation_tokens_{args.start_token}_{args.end_token}.npz",

    )



    reference_frame = model_frames[2 * args.reference_token]



    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(reference_frame)

    ax.axis("off")

    ax.set_title(

        f"Reference blue-ball region | token {args.reference_token} | "

        f"rows {r0}:{r1 - 1}, cols {c0}:{c1 - 1}",

        fontsize=12,

    )



    roi_rect = plt.Rectangle(

        (x, y),

        w,

        h,

        fill=False,

        edgecolor="red",

        linewidth=2.5,

    )

    ax.add_patch(roi_rect)



    for pos in range(0, IMAGE_SIZE + 1, PATCH_SIZE):

        ax.axhline(pos, color="white", linewidth=0.35, alpha=0.45)

        ax.axvline(pos, color="white", linewidth=0.35, alpha=0.45)



    fig.tight_layout()

    fig.savefig(reference_out, dpi=180, bbox_inches="tight")

    plt.close(fig)



    rows = len(token_indices)

    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.3 * rows))



    if rows == 1:

        axes = np.expand_dims(axes, axis=0)



    for row, (token_index, mask, appearance, record) in enumerate(

        zip(token_indices, masks, appearance_maps, records)

    ):

        frame = model_frames[2 * token_index]

        overlay = make_similarity_overlay(frame, appearance, vis_min, vis_max)



        axes[row, 0].imshow(frame)

        axes[row, 0].axis("off")

        axes[row, 0].set_title(

            f"token {token_index} | frame {record['frame']} | "

            f"{record['time']:.2f}s",

            fontsize=10,

        )

        draw_mask_and_centroid(

            axes[row, 0],

            mask,

            np.array([record["centroid_row"], record["centroid_col"]]),

        )



        if token_index == args.reference_token:

            axes[row, 0].add_patch(

                plt.Rectangle(

                    (x, y),

                    w,

                    h,

                    fill=False,

                    edgecolor="red",

                    linewidth=2.2,

                )

            )



        im = axes[row, 1].imshow(

            appearance,

            cmap="turbo",

            vmin=vis_min,

            vmax=vis_max,

            interpolation="nearest",

        )

        axes[row, 1].set_title(

            f"appearance similarity | peak={record['peak_appearance']:.4f}",

            fontsize=10,

        )

        axes[row, 1].set_xlabel("patch column")

        axes[row, 1].set_ylabel("patch row")

        axes[row, 1].contour(mask.astype(np.float32), levels=[0.5], colors="white", linewidths=1.2)

        axes[row, 1].plot(

            record["centroid_col"],

            record["centroid_row"],

            marker="x",

            color="white",

            markersize=8,

            markeredgewidth=2,

        )



        axes[row, 2].imshow(overlay)

        axes[row, 2].axis("off")

        axes[row, 2].set_title(

            f"propagated mask | patches={record['mask_size']} | "

            f"mean={record['mask_mean_similarity']:.4f}",

            fontsize=10,

        )

        draw_mask_and_centroid(

            axes[row, 2],

            mask,

            np.array([record["centroid_row"], record["centroid_col"]]),

        )



    fig.suptitle(

        "V-JEPA 2.1 multi-reference patch mask propagation",

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

    colorbar.set_label("Appearance similarity to blue-ball memory")



    fig.savefig(propagation_out, dpi=180, bbox_inches="tight")

    plt.close(fig)



    trajectory = np.array(

        [[r["centroid_col"], r["centroid_row"]] for r in records],

        dtype=np.float32,

    )



    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(reference_frame)

    ax.set_xlim(0, IMAGE_SIZE)

    ax.set_ylim(IMAGE_SIZE, 0)

    ax.set_aspect("equal")

    ax.set_title("Propagated blue-ball patch-centroid trajectory")



    ax.plot(

        trajectory[:, 0] * PATCH_SIZE + PATCH_SIZE / 2,

        trajectory[:, 1] * PATCH_SIZE + PATCH_SIZE / 2,

        marker="o",

        linewidth=2,

        markersize=5,

        color="lime",

    )



    for record in records:

        px = record["centroid_col"] * PATCH_SIZE + PATCH_SIZE / 2

        py = record["centroid_row"] * PATCH_SIZE + PATCH_SIZE / 2



        ax.text(

            px + 4,

            py - 4,

            str(record["token"]),

            color="white",

            fontsize=9,

            weight="bold",

        )



    ax.axis("off")

    fig.tight_layout()

    fig.savefig(trajectory_out, dpi=180, bbox_inches="tight")

    plt.close(fig)



    stacked_masks = np.stack(masks, axis=0).astype(np.uint8)

    stacked_maps = np.stack(appearance_maps, axis=0).astype(np.float32)

    stacked_centroids = np.array(centroids, dtype=np.float32)



    np.savez_compressed(

        npz_out,

        token_indices=np.array(token_indices, dtype=np.int64),

        masks=stacked_masks,

        appearance_similarity=stacked_maps,

        centroids=stacked_centroids,

        reference_roi=np.array(args.roi, dtype=np.int64),

        reference_patch_bounds=np.array([r0, r1, c0, c1], dtype=np.int64),

    )



    report = [

        "V-JEPA 2.1 multi-reference patch mask propagation",

        f"features: {args.features}",

        f"video: {args.video}",

        f"fps: {fps:.6f}",

        f"token range: {args.start_token} to {args.end_token}",

        f"reference token: {args.reference_token}",

        f"reference ROI xywh: {args.roi}",

        f"reference patch rows: {r0} to {r1 - 1}",

        f"reference patch cols: {c0} to {c1 - 1}",

        f"reference patch count: {len(reference_features)}",

        f"spatial sigma: {args.spatial_sigma}",

        f"max jump: {args.max_jump}",

        f"spatial weight: {args.spatial_weight}",

        f"memory frames: {args.memory_frames}",

        "",

    ]



    for record in records:

        report.extend(

            [

                f"token {record['token']}",

                f"  source frame: {record['frame']}",

                f"  timestamp: {record['time']:.6f}s",

                f"  peak patch row/col: [{record['peak_row']}, {record['peak_col']}]",

                f"  peak appearance similarity: {record['peak_appearance']:.6f}",

                f"  peak reference similarity: {record['peak_reference_similarity']:.6f}",

                f"  propagated mask patches: {record['mask_size']}",

                f"  mask mean similarity: {record['mask_mean_similarity']:.6f}",

                f"  centroid patch row/col: "

                f"[{record['centroid_row']:.3f}, {record['centroid_col']:.3f}]",

                "",

            ]

        )



    with open(report_out, "w", encoding="utf-8") as f:

        f.write("\n".join(report))



    print()

    print("Object mask propagation: success")

    print(f"Reference visualization: {reference_out}")

    print(f"Propagation figure: {propagation_out}")

    print(f"Trajectory figure: {trajectory_out}")

    print(f"Report: {report_out}")

    print(f"Raw propagation arrays: {npz_out}")

    print()



    for record in records:

        print(

            f"token {record['token']:2d}: "

            f"center=({record['centroid_row']:.2f}, {record['centroid_col']:.2f}), "

            f"peak={record['peak_appearance']:.4f}, "

            f"mask={record['mask_size']:2d}"

        )





if __name__ == "__main__":

    main()

