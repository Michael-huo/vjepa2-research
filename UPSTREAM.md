
# Upstream and research baseline



This repository is a research derivative of `facebookresearch/vjepa2`.



## Upstream



- Upstream repository: `facebookresearch/vjepa2`

- Local upstream baseline commit: `20e498b Fix figure (#143)`

- Original upstream license files and copyright notices are retained.



## Local research additions



This baseline records an initial V-JEPA 2.1 inference study:



- CUDA-enabled V-JEPA 2.1 deployment and pretrained-weight loading.

- Dense video-token extraction and temporal-similarity analysis.

- Dense PCA feature visualization and patch-wise local-change analysis.

- Patch-similarity object correspondence and mask-propagation experiments.

- Masked latent-feature prediction and context-ablation diagnostics.



## Storage policy



The following are intentionally excluded from Git:



- Downloaded model checkpoints and caches.

- Generated experiment outputs under `research/outputs/`.

- Raw `.npz` and `.npy` feature arrays.

- Third-party sample videos under `research/assets/`.

- Crash dumps and local backup files.



The `main` branch preserves the initial working baseline. Later code reorganization is performed in dedicated research branches.
