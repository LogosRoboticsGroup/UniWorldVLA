<div align="center">

# **Uni-World VLA**: Interleaved World Modeling and Planning for Autonomous Driving

**Qiqi Liu<sup>1,2,3</sup>\*, Huan Xu<sup>3</sup>\*, Jingyu Li<sup>1,2,3</sup>, Bin Sun<sup>3</sup>†, Zhihui Hao<sup>3</sup>†, Dangen She<sup>3</sup>, Xiatian Zhu<sup>4</sup>, Li Zhang<sup>1,2</sup>‡**

<sup>1</sup>Fudan University; <sup>2</sup>Shanghai Innovation Institute; <sup>3</sup>Li Auto Inc.; <sup>4</sup>University of Surrey

<span style="color: #aaaaaa;">\* equal contribution; † project leader; ‡ corresponding author</span>

[![arXiv](https://img.shields.io/badge/arXiv-2603.27287-b31b1b.svg)](https://arxiv.org/abs/2603.27287)

</div>

---

## 📰 News

- **[2026-04]** Paper released on arXiv.

---

## 🔭 Project Overview

![Overview](assets/Overview.png)

Uni-World VLA is a unified Vision-Language-Action model for autonomous driving that performs **interleaved world modeling and planning**. It jointly predicts future visual observations and ego trajectories in a single autoregressive sequence, tightly coupling world understanding with planning under temporal causality.

---

## 💡 Key Features

- **Interleaved world modeling and planning:** alternates future frame prediction and ego action/trajectory generation step-by-step, forming a closed-loop interaction that keeps planning conditioned on imagined observations.
- **Unified autoregressive VLA formulation:** generates visual tokens and action queries in a single sequence, tightly coupling prediction and control under temporal causality.
- **Depth integration for geometric cues:** augments historical frames with monocular depth maps and fuses geometry features via cross-attention to improve long-horizon scene prediction.

---

## 📊 Results

**Table 1.** Closed-loop planning results on NAVSIM.

![Table1](assets/Table1.png)

**Table 2.** World modeling / prediction results on NAVSIM.

![Table2](assets/Table2.png)

**Visualization.**

![Visualization](assets/Visualization.png)

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/LogosRoboticsGroup/UniWorldVLA.git
cd UniWorldVLA
```

> **Note:** The `DA3` (Depth Anything 3) component is included as a git submodule. If you cloned without `--recurse-submodules`, run:
> ```bash
> git submodule update --init --recursive
> ```

### 2. Create environment

```bash
conda create -n uniworld python=3.10 -y
conda activate uniworld
```

### 3. Install dependencies

```bash
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> Full pinned environment (including all transitive deps) is available in `environment.yaml` for exact reproducibility:
> ```bash
> conda env create -f environment.yaml
> ```

---

## 📦 Pretrained Weights & Data

Model weights are released at [SII-Rigby/UniWorldVLA](https://huggingface.co/SII-Rigby/UniWorldVLA) on HuggingFace.

See **[SETUP.md](SETUP.md)** for detailed instructions on:
- Downloading backbone model weights (Show-O, Phi-1.5, MagViT-v2, I3D)
- Downloading our released checkpoints (VQ tokenizer, pre-trained model, SFT checkpoint)
- Preparing the NAVSIM dataset
- Setting up the DA3 depth model

---

## ⚙️ Configuration

All paths are configured in `configs/sft_navsim/navsim.yaml`. Before running, set the `base_root` field to the absolute path of this repository on your machine:

```yaml
experiment:
    base_root: '/path/to/UniWorldVLA'   # <-- change this
```

All other paths in the config are derived from `base_root` via OmegaConf interpolation. See [SETUP.md](SETUP.md) for the expected directory layout.

---

## 🚀 Training

### Single node, 8 GPUs

```bash
cd UniWorldVLA
bash scripts/finetune/navsim/run_sft_navsim_baseline8.sh
```

The script launches `training/fine-tune_navsim.py` via `accelerate` with the DeepSpeed ZeRO-2 config in `accelerate_configs/8_gpus_deepspeed_zero2.yaml`.

Key training options are controlled via environment variables (see `training/fine-tune_navsim.py` → `configure_experiment_from_env`):

| Variable | Default | Description |
|---|---|---|
| `MAX_TRAIN_STEPS` | 160000 | Total training steps |
| `LEARNING_RATE` | 1e-5 | Base learning rate |
| `BATCH_SIZE_TRAIN_NUS` | 7 | Per-GPU batch size |
| `VIDEO_COEFF` | 0.3 | Weight for video prediction loss |
| `TJ_COEFF` | 1.0 | Weight for trajectory loss |
| `EVAL_ONLY` | 0 | Set to 1 to run evaluation only |
| `LOCAL_RUN_PWM` | 0 | Set to 1 for local debug mode (small subset) |

Example with custom settings:

```bash
MAX_TRAIN_STEPS=50000 LEARNING_RATE=3e-5 \
bash scripts/finetune/navsim/run_sft_navsim_baseline8.sh
```

### Local / single-GPU debug

```bash
LOCAL_RUN_PWM=1 bash scripts/finetune/navsim/run_sft_navsim_baseline8_local.sh
```

---

## 📐 Evaluation

```bash
EVAL_ONLY=1 EVAL_FROM_CHECKPOINT=1 \
EVAL_DIR=/path/to/checkpoint \
bash scripts/finetune/navsim/run_sft_navsim_baseline8.sh
```

Evaluation computes:
- **PDMS** (PDM Score) for closed-loop planning
- **FVD** for video prediction quality

---

## 🧾 TODO

- [x] Release arXiv paper
- [x] Release code
- [x] Release model weights

---

## 📖 Citation

```bibtex
@article{liu2026uniworld,
  title   = {Uni-World VLA: Interleaved World Modeling and Planning for Autonomous Driving},
  author  = {Liu, Qiqi and Xu, Huan and Li, Jingyu and Sun, Bin and Hao, Zhihui and She, Dangen and Zhu, Xiatian and Zhang, Li},
  journal = {arXiv preprint arXiv:2603.27287},
  year    = {2026},
}
```
