
import argparse

import os



import cv2

import matplotlib.pyplot as plt

import numpy as np

from decord import VideoReader, cpu





def parse_args():

    parser = argparse.ArgumentParser(

        description="Visualize dense V-JEPA 2.1 patch features with PCA and local change maps."

    )

    parser.add_argument("--features", required=True, help="Path to .npz features file.")

    parser.add_argument("--video", required=True, help="Path to original video.")

    parser.add_argument("--output-dir", default="outputs", help="Output directory.")

    parser.add_argument("--start-token", type=int, default=12, help="Start temporal token index.")

    parser.add_argument("--end-token", type=int, default=19, help="End temporal token index (inclusive).")

    return parser.parse_args()





def robust_normalize(x):

    lo = np.percentile(x, 1, axis=0, keepdims=True)

    hi = np.percentile(x, 99, axis=0, keepdims=True)

    x = (x - lo) / (hi - lo + 1e-8)

    return np.clip(x, 0.0, 1.0)





def pca_project(features_2d, n_components=3):

    mean = features_2d.mean(axis=0, keepdims=True)

    centered = features_2d - mean

    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    basis = vt[:n_components].T

    projected = centered @ basis

    return projected, mean, basis





def pca_apply(features_2d, mean, basis):

    centered = features_2d - mean

    projected = centered @ basis

    return projected





def cosine_change_map(a, b):

    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)

    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)

    cos = np.sum(a_norm * b_norm, axis=-1)

    change = 1.0 - cos

    return np.clip(change, 0.0, 2.0)





def frame_time_str(frame_idx, fps):

    return f"{frame_idx / fps:.2f}s"





def main():

    args = parse_args()



    os.makedirs(args.output_dir, exist_ok=True)



    with np.load(args.features) as data:

        sampled_frame_indices = data["sampled_frame_indices"].astype(np.int64)

        fps = float(data["fps"][0])

        patch_tokens = data["patch_tokens"].astype(np.float32)

        temporal_similarity = data["temporal_similarity"].astype(np.float32)



    if patch_tokens.shape[0] != 1:

        raise ValueError(f"Expected batch size 1, got {patch_tokens.shape[0]}")



    token_count = temporal_similarity.shape[0]

    if token_count * 24 * 24 != patch_tokens.shape[1]:

        raise ValueError("patch_tokens shape is inconsistent with 24x24 token grid")



    token_grid = patch_tokens.reshape(1, token_count, 24, 24, 768)[0]



    start_token = args.start_token

    end_token = args.end_token



    if start_token < 0 or end_token >= token_count or start_token > end_token:

        raise ValueError(f"Invalid token range [{start_token}, {end_token}] for token_count={token_count}")



    token_indices = list(range(start_token, end_token + 1))



    # ------------------------------------------------------------

    # Load representative original frames (use first sampled frame of each token)

    # ------------------------------------------------------------

    vr = VideoReader(args.video, ctx=cpu(0))

    representative_frame_indices = [int(sampled_frame_indices[2 * t]) for t in token_indices]

    representative_frames = vr.get_batch(representative_frame_indices).asnumpy()



    # ------------------------------------------------------------

    # PCA over all patch tokens in the selected token range

    # ------------------------------------------------------------

    selected_features = token_grid[start_token:end_token + 1].reshape(-1, 768)

    projected_all, mean, basis = pca_project(selected_features, n_components=3)

    projected_all = robust_normalize(projected_all)



    pca_maps = []

    offset = 0

    for _ in token_indices:

        chunk = projected_all[offset:offset + 24 * 24]

        offset += 24 * 24

        rgb = chunk.reshape(24, 24, 3)

        rgb_up = cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_NEAREST)

        pca_maps.append(rgb_up)



    # ------------------------------------------------------------

    # Figure 1: Original frame + PCA feature map

    # ------------------------------------------------------------

    rows = len(token_indices)

    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))

    if rows == 1:

        axes = np.expand_dims(axes, axis=0)



    for i, token_idx in enumerate(token_indices):

        frame_idx = representative_frame_indices[i]

        frame = representative_frames[i]

        pca_map = pca_maps[i]



        axes[i, 0].imshow(frame)

        axes[i, 0].axis("off")

        axes[i, 0].set_title(f"token {token_idx} | frame {frame_idx} | {frame_time_str(frame_idx, fps)}", fontsize=11)



        axes[i, 1].imshow(pca_map)

        axes[i, 1].axis("off")

        axes[i, 1].set_title(f"token {token_idx} | V-JEPA PCA features", fontsize=11)



    fig.suptitle("V-JEPA 2.1 dense patch features (PCA visualization)", fontsize=16, y=0.995)

    fig.tight_layout()

    pca_out = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.features))[0]}_pca_tokens_{start_token}_{end_token}.png")

    fig.savefig(pca_out, dpi=180, bbox_inches="tight")

    plt.close(fig)



    # ------------------------------------------------------------

    # Figure 2: Local change maps for consecutive token pairs

    # ------------------------------------------------------------

    pair_indices = list(range(start_token, end_token))

    rows = len(pair_indices)

    fig, axes = plt.subplots(rows, 3, figsize=(14, 4.2 * rows))

    if rows == 1:

        axes = np.expand_dims(axes, axis=0)



    report_lines = []

    report_lines.append("V-JEPA 2.1 local dense-feature change analysis")

    report_lines.append(f"feature file: {args.features}")

    report_lines.append(f"video file: {args.video}")

    report_lines.append(f"token range: {start_token} to {end_token}")

    report_lines.append(f"fps: {fps:.6f}")

    report_lines.append("")



    for row, t in enumerate(pair_indices):

        frame_idx_a = int(sampled_frame_indices[2 * t])

        frame_idx_b = int(sampled_frame_indices[2 * (t + 1)])

        frame_a = vr[frame_idx_a].asnumpy()

        frame_b = vr[frame_idx_b].asnumpy()



        feat_a = token_grid[t]

        feat_b = token_grid[t + 1]



        change_map = cosine_change_map(feat_a, feat_b)

        mean_change = float(change_map.mean())

        max_change = float(change_map.max())

        sim_to_next = float(temporal_similarity[t, t + 1])



        change_map_up = cv2.resize(change_map, (frame_b.shape[1], frame_b.shape[0]), interpolation=cv2.INTER_CUBIC)



        axes[row, 0].imshow(frame_a)

        axes[row, 0].axis("off")

        axes[row, 0].set_title(f"token {t} | frame {frame_idx_a} | {frame_time_str(frame_idx_a, fps)}", fontsize=10)



        axes[row, 1].imshow(frame_b)

        axes[row, 1].axis("off")

        axes[row, 1].set_title(f"token {t+1} | frame {frame_idx_b} | {frame_time_str(frame_idx_b, fps)}", fontsize=10)



        axes[row, 2].imshow(frame_b)

        axes[row, 2].imshow(change_map_up, cmap="jet", alpha=0.48)

        axes[row, 2].axis("off")

        axes[row, 2].set_title(

            f"local change {t}->{t+1}\ncos={sim_to_next:.4f} | meanΔ={mean_change:.4f} | maxΔ={max_change:.4f}",

            fontsize=10,

        )



        report_lines.append(f"pair {t}->{t+1}")

        report_lines.append(f"  frame indices: [{frame_idx_a}, {frame_idx_b}]")

        report_lines.append(f"  timestamps: [{frame_idx_a / fps:.3f}s, {frame_idx_b / fps:.3f}s]")

        report_lines.append(f"  temporal cosine similarity: {sim_to_next:.6f}")

        report_lines.append(f"  mean local change: {mean_change:.6f}")

        report_lines.append(f"  max local change: {max_change:.6f}")

        report_lines.append("")



    fig.suptitle("V-JEPA 2.1 local patch-wise temporal changes", fontsize=16, y=0.995)

    fig.tight_layout()

    change_out = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.features))[0]}_local_changes_{start_token}_{end_token}.png")

    fig.savefig(change_out, dpi=180, bbox_inches="tight")

    plt.close(fig)



    report_out = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.features))[0]}_local_changes_{start_token}_{end_token}.txt")

    with open(report_out, "w", encoding="utf-8") as f:

        f.write("\n".join(report_lines))



    print("Dense feature visualization: success")

    print(f"PCA figure: {pca_out}")

    print(f"Local-change figure: {change_out}")

    print(f"Report: {report_out}")





if __name__ == "__main__":

    main()

