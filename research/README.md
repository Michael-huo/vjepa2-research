# Research Reproduction Guide

`research/` is the local workspace for the Phase 1–3 research scripts, datasets, prepared manifests, and generated outputs. This document is an operational guide: following the commands below on a new machine prepares the project and runs every Phase. Experimental objectives, metrics, and result interpretation belong in each generated `research/outputs/<phase>/report.md`.

Run every command below from the repository root unless the command states otherwise.

## 1. Get the code

```bash
git clone https://github.com/Michael-huo/vjepa2-research.git
cd vjepa2-research
```

## 2. Create the environment

The research runs require Python 3.11+ and CUDA for Phase 1, the full Phase 2 run, and the non-prepare-only Phase 3 runs. The commands below use the CUDA 12.4 PyTorch build used for this project.

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

## 5. Verify the installation

Use the built-in help for the current command interface:

```bash
python research/scripts/run_phase1_probe.py --help
python research/scripts/run_phase2_dense.py --help
python research/scripts/run_phase3_efficiency.py --help
```

Run the repository test suite after installation:

```bash
python -m pytest tests -v
```

Generated reports, metrics, manifests, figures, feature tensors, and local datasets remain under `research/outputs/` and `research/assets/`; they are intentionally ignored by Git.
