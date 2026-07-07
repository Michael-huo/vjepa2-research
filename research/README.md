# Phase 1 — V-JEPA 2.1 Capability Probe

`research/` 是本仓库新增研究代码、本地测试视频和生成结果的唯一工作区。仓库根目录中的 `app/`、`configs/`、`evals/`、`src/`、`tests/`、官方 `assets/` 等仍是上游 V-JEPA 2.1 内容；上游来源与许可证信息见根目录 `UPSTREAM.md`、`LICENSE` 和 `APACHE-LICENSE`。

## 阶段命名

后续研究阶段使用简洁命名：

```text
phase<N>-<single-lowercase-english-word>
```

当前阶段是：

```text
phase1-probe
```

含义是第一阶段 V-JEPA 2.1 核心能力探测。它不是通用视频分析工具，而是固定样例上的首次能力检查。

## 目录结构

```text
research/
├── README.md
├── __init__.py
├── assets/
│   ├── .gitkeep
│   └── sample_bowling.mp4
├── outputs/
│   ├── .gitkeep
│   └── phase1-probe/
│       └── sample_bowling/
└── scripts/
    ├── __init__.py
    ├── run_phase1_probe.py
    └── common/
        ├── __init__.py
        ├── runtime.py
        ├── video_models.py
        ├── analysis.py
        └── visualization.py
```

`research/assets/` 保存本地测试视频，例如 `research/assets/sample_bowling.mp4`。`research/outputs/` 保存阶段运行结果。视频、图片、NPZ、JSON、Markdown 报告等运行产物都被 Git 忽略；只保留 `.gitkeep` 作为空目录占位。模型权重仍使用系统缓存，例如 `~/.cache/torch/hub/checkpoints/`，不进入仓库。

## 环境

推荐使用已准备好的环境：

```bash
conda activate vjepa
```

已验证的 CUDA 12.4 / PyTorch 安装命令：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python -m pip install matplotlib
```

GPU 验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Phase 1 强制使用当前 shell 可见的第一张 CUDA GPU；脚本不会回退到 CPU，也不要求在命令前手动写 `CUDA_VISIBLE_DEVICES=0`。

## 测试视频

下载固定样例视频：

```bash
mkdir -p research/assets
wget -c -O research/assets/sample_bowling.mp4 https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4
```

脚本不会自动下载第三方视频或数据集。使用其他完整数据集时，需要自行遵守对应许可证、隐私和访问要求。

## 唯一运行命令

```bash
python research/scripts/run_phase1_probe.py
```

脚本也兼容模块形式：

```bash
python -m research.scripts.run_phase1_probe
```

README 只推荐直接运行命令。每次运行会先验证固定视频与 CUDA 环境，然后写入 staged 目录；只有七个最终文件全部生成并校验通过后，才替换当前视频的旧结果目录。若中途失败，上一次成功结果会保留。

## 七个核心输出

最终目录：

```text
research/outputs/phase1-probe/sample_bowling/
```

只包含七个文件：

- `features.npz`：保存 sampled frame indices、FPS、ViT-B patch tokens、temporal features、video feature、temporal similarity 和 schema metadata，便于未来复用 dense features。
- `representation.png`：综合展示 temporal similarity、代表性 temporal changes、token 12/14/16/19 的原始帧与 PCA latent pseudo-color，以及同位置 local latent change。
- `correspondence.png`：综合展示 reference ROI、对象候选区域、appearance response、patch centroid、简洁 trajectory 与 uncertainty flags。
- `completion.png`：综合展示 target masked block、full context latent cosine、四种 context 指标表和 full retrieval matrix。
- `metrics.json`：保存 experiment measurements and analysis values，也就是 representation、correspondence、completion 的核心数值。
- `report.md`：自动生成中文运行报告，说明关键结果与能力边界。
- `manifest.json`：保存复现信息，包括环境、固定配置、输入视频 hash、Git revision、dirty-worktree state、tracked diff hash、精确 Phase 1 source-file hashes、combined source fingerprint、模型、设备、输出文件和 end-to-end timing。

## 内部分析环节

### representation

固定抽取 64 帧，使用官方 `vjepa2_preprocessor` 与 V-JEPA 2.1 ViT-B/16 384 encoder，生成 `(1, 18432, 768)` patch tokens。32 个 temporal tokens 对应 24x24 patch grid。脚本计算 temporal features、video feature、temporal similarity 和 mean consecutive temporal similarity。

token 12-19 的 PCA 只对这一连续区间的全部 patch token 统一拟合一次，用于 dense latent pseudo-color 可视化，不是 RGB 重建。local latent change 定义为：

```text
D_t(h,w) = 1 - cosine(F_t(h,w), F_(t+1)(h,w))
```

它表示同一空间位置的潜在状态变化，不是 optical flow、不是 RGB pixel difference，也不能直接等价于跨空间对象跟踪。

### correspondence

固定 reference token 为 12，reference ROI 为 `(48, 272, 64, 64)`，坐标系是模型对齐后的 384x384 图像。内部使用多 reference patch similarity、memory bank、spatial prior、local search gate 和 peak-connected 连通候选区域。

图中的红色轮廓是粗粒度对象候选区域，不是像素级对象分割；patch centroid 是 patch-level 重心，不是精确物理轨迹。`area_expanded`、`response_diffuse`、`trajectory_jump`、`low_confidence` 等 flags 都应理解为不确定性信号。

### completion

固定 target block 为 token 16、row 16、col 2、height 6、width 8，共 48 个 target tokens。上下文模式固定为：

- `full`：除 target block 外全部 token 可见，是时空 latent completion，不是纯未来预测。
- `spatial_only`：只可见同一时间平面中 target block 外的 patch，用于测量同帧空间上下文贡献。
- `temporal_bi`：隐藏当前 target token 的整张 24x24 平面，只使用前后时间 token。
- `past_only`：只使用历史 token，是因果式诊断，不证明严格因果 world model。

teacher 使用 V-JEPA 2.1 ViT-G/16 target latent；student 使用 ViT-B/16 encoder + predictor，并强制 `training=False`。`mean non-matching cosine` 会对每个 predicted target token 排除 diagonal 正确位置后，对其他 teacher target token 求均值，不使用 random permutation 或 roll baseline。retrieval 图中的白色对角线标注为 `ideal correspondence reference`，它只是正确位置参考线，不是模型输出轨迹。

## 指标解释

- `mean target cosine`：同位置 predicted latent 与 teacher latent 的平均 cosine。
- `mean non-matching cosine`：错误空间位置 latent similarity 的均值基线。
- `Top-1` / `Top-5`：正确 target patch 是否在 retrieval 排名靠前。
- `mean rank` / `MRR`：正确位置的排序质量。
- `Top-1 within 1-patch neighborhood`：最佳匹配是否位于真实 patch 的 8 邻域。
- `mean best-match spatial error`：最佳匹配与真实位置的平均 patch 距离。

默认 48 个 target tokens 时，Top-1 random chance = `1/48`，Top-5 random chance = `5/48`。这些数值只用于理解诊断难度，不是通用 benchmark 分数。

## 局限

- ViT-B dense token grid 为 24x24，空间分辨率只有 patch-level。
- 对应与传播是粗粒度能力诊断，不是精确实例分割。
- 遮挡、模糊、快速运动和手-物体接触会提高不确定性。
- `full` context 不等于未来预测；`past_only` 不等于动作条件或严格因果世界模型。
- Phase 1 是能力探测，不是论文 benchmark 复现，也不证明控制、规划或机器人执行能力。

## Git 与复现

不提交 `research/assets/` 中的视频、`research/outputs/` 中的生成结果、NPZ 或模型权重。正式实验应保留 `manifest.json`、`metrics.json` 和对应 Git hash，以便追踪结果来源。

`metrics.json` 记录实验测量值和分析指标；`manifest.json` 记录环境、固定配置、输入视频哈希、Git revision、dirty-worktree state、tracked diff hash、Phase 1 源码文件哈希、combined source fingerprint 和端到端运行时间。当 `git_worktree_dirty=true` 时，结果来自未提交的 Phase 1 代码，应结合 `phase_source_fingerprint` 与 `phase_source_files` 精确识别该次运行使用的源码状态。
