# Phase 2 Dense DAVIS Pipeline

`phase2-dense` runs a DAVIS 2017 TrainVal 480p-only dense baseline under `research/`.
It uses V-JEPA 2.1 ViT-B frame by frame through the image-tokenizer path to produce dense PCA feature visualizations and a simple first-frame binary foreground VOS propagation diagnostic.

This is not an exact reproduction of the paper's full video-tokenizer Figure 15 protocol and it is not official DAVIS J&F evaluation.
It is also not a resolution, runtime, memory, throughput, or edge-device feasibility study.
Phase 3 will evaluate input resolution choices, computational cost, memory usage, inference time, and whether this style of method is practical under edge-computing constraints.

## Manual Dataset Step

Run the single authoritative DAVIS preparation command in [research/README.md](README.md#数据准备) from the repository root.

Expected layout:

```text
research/assets/datasets/davis2017/DAVIS/
  JPEGImages/480p/
  Annotations/480p/
  ImageSets/
```

Default sequences:

- `dog`
- `car-shadow`
- `parkour`

The pipeline reads DAVIS frame sequences and masks directly from this layout. It does not require mp4 files, copied images, or copied masks.

## Commands

Lightweight DAVIS validation only:

```bash
python research/scripts/run_phase2_dense.py --prepare-only
```

Full Phase 2 run:

```bash
python research/scripts/run_phase2_dense.py
```

Optional sequence override:

```bash
python research/scripts/run_phase2_dense.py --sequences dog car-shadow parkour
```

## Outputs

Final output directory:

```text
research/outputs/phase2-dense/
  image_dense_pca/
    image_pca_summary.png
    metrics.json
  video_dense_pca/
    video_pca_summary.png
    metrics.json
  vos_label_propagation/
    vos_summary.png
    metrics.json
  phase2_dense_summary.png
  summary.csv
  metrics.json
  report.md
  manifest.json
```

The prepared dataset manifest is written to:

```text
research/assets/prepared/davis2017/manifest.json
```

The prepared manifest records paths and a lightweight DAVIS structure fingerprint only. It does not copy frames or masks.

## Interpretation

- PCA maps are dense feature visualizations, not semantic segmentation and not RGB reconstruction.
- Video PCA fits one PCA transform per sequence, so colors are comparable across sampled frames within a sequence.
- VOS uses `00000.png` as a first-frame annotation and converts `mask > 0` to foreground, merging all DAVIS instances into one binary mask.
- VOS metrics are sampled aligned-crop IoU/J on the deterministic 384x384 preprocessing crop, not official DAVIS J&F.
- `summary.csv` is a compact sequence-level table for quick comparison across adjacent-frame cosine, PCA color drift, sampled-crop J, and low-IoU frames.
- Phase 2 intentionally does not test 512/768/1024-style resolutions, runtime, memory, throughput, or edge feasibility; those are Phase 3 topics.

## Lightweight Checks

```bash
python -m py_compile research/scripts/run_phase2_dense.py research/scripts/common/phase2_data.py research/scripts/common/dense_pca.py research/scripts/common/vos.py research/scripts/common/visualization.py
python research/scripts/run_phase2_dense.py --prepare-only
```

Do not commit DAVIS data, downloaded archives, generated outputs, checkpoints, or feature tensors.
