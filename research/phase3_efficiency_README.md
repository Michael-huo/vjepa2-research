# Phase 3 Efficiency

Phase 3 measures whether V-JEPA 2.1 dense features are practical under real latency, throughput, GPU-memory, and sampled quality-efficiency constraints. Phase 2 established the basic dense-feature and DAVIS VOS propagation behavior; Phase 3 measures its compute demand.

## Models and cross-platform paths

The logical model IDs remain `vit_b`, `vit_l`, `vit_g`, and `vit_G`. Human-readable figures and reports use `ViT-B`, `ViT-L`, `ViT-g`, and `ViT-G`.

Case-insensitive filesystems such as common Windows installations cannot safely distinguish directories named only by `g` versus `G`. Filesystem artifacts therefore use separate path encodings:

| Model ID | Display name | Artifact slug |
| --- | --- | --- |
| `vit_b` | ViT-B | `vit_b` |
| `vit_l` | ViT-L | `vit_l` |
| `vit_g` | ViT-g | `vit_g_lc` |
| `vit_G` | ViT-G | `vit_G_uc` |

The `lc` and `uc` suffixes are filesystem escaping markers only. They are not model names and are not used in scientific conclusions.

## Data

Phase 3 reuses the validated Phase 2 DAVIS dataset at:

```text
research/assets/phase2-dense/datasets/davis2017/DAVIS
```

It deterministically selects distinct frames round-robin from `dog`, `car-shadow`, and `parkour`, decodes them once, and reuses the same in-memory RGB pool for every model and benchmark. Disk access and JPEG decoding are outside benchmark timing.

DAVIS 2017 TrainVal is 480p. Resizing its frames to 1024 does not add physical image information. That setting measures denser patch-grid compute and may alter mask discretization or alignment.

## Commands

Quick mode evaluates ViT-B at 384 and 512 with batch sizes 1 and 4:

```bash
python research/scripts/run_phase3_efficiency.py
```

Full mode evaluates ViT-B, ViT-L, ViT-g, and ViT-G across the configured resolution and batch matrix:

```bash
python research/scripts/run_phase3_efficiency.py --full
```

Dataset validation only, without CUDA or model loading:

```bash
python research/scripts/run_phase3_efficiency.py --prepare-only
```

Efficiency without quality evaluation:

```bash
python research/scripts/run_phase3_efficiency.py --no-quality
python research/scripts/run_phase3_efficiency.py --full --no-quality
```

No-quality mode still writes feasibility and batch results, a quality CSV with headers, placeholder quality figures, and explanatory report sections. It creates no selected quality-output directories.

Full mode may download or load very large checkpoints, can OOM on 24 GB GPUs, and now takes longer because it uses five warmups and ten measured repeats per configuration. Individual unavailable, error, and OOM results are recorded without terminating the remaining matrix.

## Canonical benchmark

Each model is loaded once per run. For every measured `model x resolution x batch_size`, Phase 3 creates exactly one canonical result. The batch-size-1 result is projected into feasibility and referenced by quality evaluation; it is never independently remeasured.

Timing starts from RGB arrays already decoded in RAM:

- Preprocessing includes resize, center crop, tensor conversion, normalization, and batch stacking.
- Encoder core includes CPU-to-GPU transfer, encoder forward, and CUDA synchronization.
- Online pipeline directly times preprocessing, transfer, forward, and synchronization as one contiguous operation.
- Model loading is a separate cold-start cost and is excluded from steady-state latency and FPS.

Batch-size-1 online pipeline latency and FPS are authoritative for real-time classification, edge interpretation, Pareto analysis, and recommendations. Large-batch pipeline throughput describes cloud or batch-processing capacity, not low-latency online inference.

The harness preserves the existing inference behavior and detects the active dtype from actual model output at runtime instead of assuming it in reports or manifests.

## Recommendation method

Configuration recommendations operate on the sampled-quality versus canonical batch-size-1 pipeline-latency Pareto set. The low-latency candidate has the lowest online pipeline latency, while the high-quality candidate has the highest sampled-crop J among non-dominated configurations.

The balanced candidate maximizes the marginal sampled-crop J gain per additional pipeline millisecond relative to the low-latency baseline. A candidate must improve absolute J by at least `0.01`; otherwise the balanced role explicitly falls back to the low-latency candidate. This replaces the former unnormalized absolute `J / latency` ratio, which was biased toward selecting the lowest-latency configuration for both roles.

## GPU memory

Phase 3 distinguishes:

- pre-load CUDA allocated and reserved memory
- model-resident memory after model loading and before inputs
- peak allocated and reserved memory during a configuration
- incremental inference memory above the model-resident baseline

Peak allocated memory is the deployment-relevant total. Incremental memory isolates input and inference workspace demand.

## Outputs and integrity

Outputs are staged and published to:

```text
research/outputs/phase3-efficiency/
```

The output includes the three CSV files, six figures, selected quality summaries, `metrics.json`, the Chinese `report.md`, `artifact_index.json`, and `manifest.json`.

Before publication, Phase 3 verifies successful quality contracts, rejects case-insensitive file or directory collisions, hashes actual artifacts with SHA-256, constructs the manifest from files that really exist, and validates every listed path. Failed integrity checks prevent publication and suppress the final success message.

Quality metrics remain Phase 2-style sampled aligned-crop J/IoU diagnostics, not official DAVIS J&F. Phase 3 does not implement segmentation or depth probes, VSLAM, tokenizer comparisons, training, compression sweeps, or edge-device deployment.
