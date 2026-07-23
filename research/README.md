# Research Reproduction Guide

`research/` is the local workspace for the Phase 1–4 research scripts, datasets, prepared manifests, and generated outputs. This document is an operational guide: following the commands below on a new machine prepares the project and runs every Phase. Experimental objectives, metrics, and result interpretation belong in each generated `research/outputs/<phase>/report.md`.

Run every command below from the repository root unless the command states otherwise.

## 1. Get the code

```bash
git clone https://github.com/Michael-huo/vjepa2-research.git
cd vjepa2-research
```

## 2. Create the environment

The research runs require Python 3.11+ and CUDA for Phase 1, the full Phase 2 run, the non-prepare-only Phase 3 runs, and the full Phase 4 run. The commands below use the CUDA 12.4 PyTorch build used for this project.

```bash
conda create -n vjepa python=3.12 -y
conda activate vjepa
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python -m pip install matplotlib pytest
```

Check the Python, CUDA, and GPU setup before running a GPU Phase:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Phase 1 and Phase 2 use the first CUDA device visible to the shell. The scripts do not fall back to CPU. Model weights are loaded through the existing local Torch Hub integration and may be fetched into the normal Torch cache on their first use.

## 3. Prepare local data

All data is local and ignored by Git. Do not commit files under `research/assets/datasets/` or `research/assets/prepared/`.

### Bowling video for Phase 1

```bash
mkdir -p research/assets/datasets/bowling && wget -c -O research/assets/datasets/bowling/sample_bowling.mp4 https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4 && test -s research/assets/datasets/bowling/sample_bowling.mp4 && echo "Bowling sample ready: research/assets/datasets/bowling/sample_bowling.mp4"
```

The fixed Phase 1 input is:

```text
research/assets/datasets/bowling/sample_bowling.mp4
```

### DAVIS 2017 TrainVal 480p for Phases 2 and 3

This command creates the target directory, resumes the archive download, only extracts when the required DAVIS layout is incomplete, validates all required directories, and prints the final location:

```bash
mkdir -p research/assets/datasets/davis2017 && cd research/assets/datasets/davis2017 && wget -c -O DAVIS-2017-trainval-480p.zip https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip && ( { [ -d DAVIS/JPEGImages/480p ] && [ -d DAVIS/Annotations/480p ] && [ -d DAVIS/ImageSets/2017 ]; } || unzip -q -o DAVIS-2017-trainval-480p.zip ) && test -d DAVIS/JPEGImages/480p && test -d DAVIS/Annotations/480p && test -d DAVIS/ImageSets/2017 && echo "DAVIS 2017 ready: $(pwd)/DAVIS"
```

The required layout is:

```text
research/assets/datasets/davis2017/DAVIS/
├── JPEGImages/480p/
├── Annotations/480p/
└── ImageSets/2017/
```

Phase 2 and Phase 3 share this root and the generated manifest at:

```text
research/assets/prepared/davis2017/manifest.json
```

Prepare or validate the shared manifest before GPU runs. The command validates the fixed `dog`, `car-shadow`, and `parkour` sequences, creates the manifest when required, and reuses it when its structure fingerprint is unchanged:

```bash
python research/scripts/run_phase2_dense.py --prepare-only
```

### TUM RGB-D for Phase 4

Phase 4 uses the official `rgbd_dataset_freiburg2_pioneer_slam` sequence. This command creates the fixed target directory, resumes the official archive download, extracts only when the required RGB/trajectory layout is incomplete, preserves the official directory name, validates all three required inputs, and prints the final location:

```bash
mkdir -p research/assets/datasets/tum_rgbd && cd research/assets/datasets/tum_rgbd && wget -c -O rgbd_dataset_freiburg2_pioneer_slam.tgz https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam.tgz && ( { [ -d rgbd_dataset_freiburg2_pioneer_slam/rgb ] && [ -f rgbd_dataset_freiburg2_pioneer_slam/rgb.txt ] && [ -f rgbd_dataset_freiburg2_pioneer_slam/groundtruth.txt ]; } || tar -xzf rgbd_dataset_freiburg2_pioneer_slam.tgz ) && test -d rgbd_dataset_freiburg2_pioneer_slam/rgb && test -f rgbd_dataset_freiburg2_pioneer_slam/rgb.txt && test -f rgbd_dataset_freiburg2_pioneer_slam/groundtruth.txt && echo "TUM RGB-D ready: $(pwd)/rgbd_dataset_freiburg2_pioneer_slam"
```

The fixed Phase 4 root and prepared manifest are:

```text
research/assets/datasets/tum_rgbd/rgbd_dataset_freiburg2_pioneer_slam/
research/assets/prepared/tum_rgbd/rgbd_dataset_freiburg2_pioneer_slam/manifest.json
```

Phase 4 currently reads only `rgb/`, `rgb.txt`, and `groundtruth.txt`; it does not use depth, accelerometer data, ROS, or rosbag tooling. Prepare or validate the timestamp-associated manifest without CUDA:

```bash
python research/scripts/run_phase4_feasibility.py --prepare-only
```

Phase 4 uses the associated RGB timestamps directly; the sequence's average effective frequency is descriptive only, and the experiments do not assume a strictly fixed 30 Hz, 18.5 Hz, or other frame rate. The fixed scientific protocol is:

- Time-based scout/reference analysis at 0.1, 0.25, 0.5, 1, 2, 4, and 8 seconds, comparing ViT-B@384 scout state changes with ViT-B@512 reference state changes.
- Offline adjacent-change score evaluation at exact 5%, 10%, 20%, 30%, and 40% high-quality observation budgets. Counts use half-up rounding: `max(1, floor(budget * N + 0.5))`.
- Causal anchor-based adaptive evaluation calibrated on the first 20% of sequence time and frozen on the remaining 80%. Stable and Normal states use 2.0-second and 0.5-second refresh timeouts.
- The primary Uniform comparison for the adaptive JEPA policy uses exactly the JEPA policy's evaluation observation count; the fixed Uniform 10% target remains a descriptive reference.

The offline experiment scores adjacent frames over the complete sequence and is not an online policy. The causal experiment instead scores each current frame against its latest refresh anchor without using future evaluation data.

The sequence is provided by the TUM RGB-D benchmark. Cite J. Sturm et al., “A Benchmark for the Evaluation of RGB-D SLAM Systems,” IROS 2012, and follow the benchmark's data access and usage terms.

The dataset archive is supplied by the DAVIS project. Follow its access, licensing, and usage terms. The Bowling command uses the existing Kinetics-mini Hugging Face source; follow that source's terms as well.

## 4. Run the Phases

### Phase 1

```bash
python research/scripts/run_phase1_probe.py
```

Requires the Bowling video and CUDA. Outputs are published to `research/outputs/phase1-probe/sample_bowling/` after the run completes.

### Phase 2

Validate or build the shared DAVIS manifest without CUDA:

```bash
python research/scripts/run_phase2_dense.py --prepare-only
```

Run the default DAVIS experiment on `dog`, `car-shadow`, and `parkour`:

```bash
python research/scripts/run_phase2_dense.py
```

Existing command options:

```bash
python research/scripts/run_phase2_dense.py --sequences dog car-shadow parkour
python research/scripts/run_phase2_dense.py --force-prepare
```

Outputs are written to `research/outputs/phase2-dense/`.

### Phase 3

Validate or reuse the same DAVIS manifest without CUDA or model loading:

```bash
python research/scripts/run_phase3_efficiency.py --prepare-only
```

Run the quick configuration:

```bash
python research/scripts/run_phase3_efficiency.py
```

Run the full configuration matrix, or skip quality evaluation when needed:

```bash
python research/scripts/run_phase3_efficiency.py --full
python research/scripts/run_phase3_efficiency.py --no-quality
python research/scripts/run_phase3_efficiency.py --full --no-quality
python research/scripts/run_phase3_efficiency.py --force-prepare
```

Outputs are written to `research/outputs/phase3-efficiency/`.

### Phase 4

Validate or reuse the fixed TUM RGB-D RGB/pose association manifest without CUDA or model loading:

```bash
python research/scripts/run_phase4_feasibility.py --prepare-only
```

Run the fixed ViT-B@384 scout and ViT-B@512 reference feasibility pipeline:

```bash
python research/scripts/run_phase4_feasibility.py
```

Outputs are atomically published to `research/outputs/phase4-feasibility/`.

## 5. Verify the installation

Use the built-in help for the current command interface:

```bash
python research/scripts/run_phase1_probe.py --help
python research/scripts/run_phase2_dense.py --help
python research/scripts/run_phase3_efficiency.py --help
python research/scripts/run_phase4_feasibility.py --help
```

Run the repository test suite after installation:

```bash
python -m pytest tests -v
```

Generated reports, metrics, manifests, figures, feature tensors, and local datasets remain under `research/outputs/` and `research/assets/`; they are intentionally ignored by Git.
