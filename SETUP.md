# 🧭 Setup Guide

This document covers all steps needed to reproduce training and evaluation results for Uni-World VLA.

---

## Directory Layout

After completing all setup steps, your repository should look like:

```
UniWorldVLA/
├── DA3/                        # Depth Anything 3 (git submodule)
├── checkpoints/
│   ├── show-o-w-clip-vit/      # Show-O backbone (HuggingFace)
│   ├── phi-1_5/                # Phi-1.5 LLM (HuggingFace)
│   ├── magvitv2/               # MagViT-v2 tokenizer config
│   └── i3d/
│       └── i3d_torchscript.pt  # I3D model for FVD evaluation
├── pretrained_models/
│   ├── tokenizer/
│   │   └── diffusion_pytorch_model.safetensors   # fine-tuned VQ tokenizer (released)
│   ├── pretrain_ckpt/
│   │   └── unwrapped_model/
│   │       └── pytorch_model.bin                 # pre-trained PWM checkpoint (released)
│   ├── ckpt_sft_navsim/
│   │   └── unwrapped_model/
│   │       └── pytorch_model.bin                 # SFT checkpoint (released)
│   └── DA3-GIANT-LARGE/        # DA3 model weights
├── dataset/
│   └── navsim/
│       ├── nuplan_scene_blobs/
│       ├── navsim/
│       │   └── nuplan_img_logs/
│       ├── navsim_logs/
│       ├── maps/
│       └── ...
├── depth_cache_8_futrue_frame_flash/   # DA3 depth cache (generated or downloaded)
│   └── {scene_token}.pt
├── configs/
├── models/
├── training/
└── ...
```

---

## 1. DA3 Submodule (Depth Anything 3)

`DA3` is included as a git submodule pointing to the official [Depth Anything 3 repository](https://github.com/ByteDance-Seed/depth-anything-3). Initialize it after cloning:

```bash
git submodule update --init --recursive
```

Then install the DA3 package:

```bash
pip install -e DA3/
```

After installing, download the DA3-GIANT-LARGE model weights from the [Depth Anything 3 releases](https://github.com/ByteDance-Seed/depth-anything-3/releases) and place them at:

```
pretrained_models/DA3-GIANT-LARGE/
```

Update the path in `configs/sft_navsim/navsim.yaml`:

```yaml
model:
    da3:
        pretrained_model_path: "${experiment.base_root}/pretrained_models/DA3-GIANT-LARGE"
```

---

## 2. Backbone Model Weights

### Show-O

Download from HuggingFace [`showlab/show-o`](https://huggingface.co/showlab/show-o) and place at `checkpoints/show-o-w-clip-vit/`:

```bash
huggingface-cli download showlab/show-o --local-dir checkpoints/show-o-w-clip-vit
```

### Phi-1.5

Download from HuggingFace [`microsoft/phi-1_5`](https://huggingface.co/microsoft/phi-1_5) and place at `checkpoints/phi-1_5/`:

```bash
huggingface-cli download microsoft/phi-1_5 --local-dir checkpoints/phi-1_5
```

### MagViT-v2 config

The MagViT-v2 tokenizer architecture config is already included in `checkpoints/magvitv2/` within the repository. The fine-tuned tokenizer weights are released separately — see Section 3 below.

### I3D (for FVD evaluation)

Download `i3d_torchscript.pt` from [`flateon/FVD-I3D-torchscript`](https://huggingface.co/flateon/FVD-I3D-torchscript) and place it at:

```bash
mkdir -p checkpoints/i3d
huggingface-cli download flateon/FVD-I3D-torchscript i3d_torchscript.pt \
  --local-dir checkpoints/i3d
```

---

## 3. Released Checkpoints

All released weights are hosted at [`SII-Rigby/UniWorldVLA`](https://huggingface.co/SII-Rigby/UniWorldVLA) on HuggingFace.

| Checkpoint | HF filename | Local path |
|---|---|---|
| VQ Tokenizer | `tokenizer/diffusion_pytorch_model.safetensors` | `pretrained_models/tokenizer/` |
| Pre-trained PWM | `pretrain_ckpt/unwrapped_model/pytorch_model.bin` | `pretrained_models/pretrain_ckpt/unwrapped_model/` |
| SFT NavSim | `ckpt_sft_navsim/unwrapped_model/pytorch_model.bin` | `pretrained_models/ckpt_sft_navsim/unwrapped_model/` |

Download with:

```bash
huggingface-cli download SII-Rigby/UniWorldVLA --local-dir pretrained_models
```

Once downloaded, the layout under `pretrained_models/` should match the directory structure shown at the top of this document.

---

## 4. NAVSIM Dataset

Follow the [official NAVSIM instructions](https://github.com/autonomousvision/navsim) to download the dataset.

The expected structure under `dataset/navsim/` is:

```
dataset/navsim/
├── nuplan_scene_blobs/    # raw scene blob files
├── navsim/
│   └── nuplan_img_logs/   # image logs
├── navsim_logs/           # log files
└── maps/                  # HD map data
```

---

## 5. Depth Cache (DA3 Pre-computed Features)

During training and evaluation, `DepthEncoder` runs DA3 inference on every NavSim scene's camera frames and caches the results to disk, avoiding redundant computation across epochs.

Cache directory (relative to `base_root`):

```
depth_cache_8_futrue_frame_flash/    # note: "futrue" is a typo in the original name — keep it as-is
└── {scene_token}.pt                 # one file per NavSim scene token
```

### Option A: Download pre-computed cache (recommended)

Running DA3-GIANT over the full NavSim training set is time-consuming. Download the pre-computed cache directly from HuggingFace:

```bash
huggingface-cli download SII-Rigby/UniWorldVLA \
  --include "depth_cache_8_futrue_frame_flash/*" \
  --local-dir .
```

### Option B: Generate from scratch

If the cache has not yet been released, or you need to regenerate it on your own NavSim data, use the following command to do a **dry run over the training set** — no model training or inference is performed:

```bash
EVAL_ONLY=1 RUN_FLASH_DATA_LOADER=1 \
bash scripts/finetune/navsim/run_sft_navsim_baseline8.sh
```

How it works:
- `EVAL_ONLY=1`: the val dataloader automatically loads the **train split** instead of the test split
- `RUN_FLASH_DATA_LOADER=1`: tokens that are already cached are skipped; for uncached tokens, DA3 is called and the result is saved to disk — **no model forward pass is executed**
- The run is safely resumable: already-generated files are never recomputed

> To generate cache for the **test split only** (e.g. for evaluation), omit `EVAL_ONLY=1`. The val dataloader defaults to the test split.

---

## 6. Final Config Check

Open `configs/sft_navsim/navsim.yaml` and set:

```yaml
experiment:
    base_root: '/absolute/path/to/UniWorldVLA'
```

Verify by running a quick import check:

```bash
LOCAL_RUN_PWM=1 python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/sft_navsim/navsim.yaml')
print('base_root:', cfg.experiment.base_root)
print('showo path:', cfg.model.showo.pretrained_model_path)
print('Config OK')
"
```
