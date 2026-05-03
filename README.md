# L-SR1: Learned Symmetric-Rank-One Preconditioning

Official PyTorch code for **L-SR1: Learned Symmetric-Rank-One Preconditioning** (ICML 2026). **Links:** [Project page](https://gallif.github.io/lsr1) · [Paper (arXiv)](https://arxiv.org/abs/2508.12270) · [PDF](https://arxiv.org/pdf/2508.12270.pdf) · ICML proceedings *(currently unavailable)*.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{lifshitz2025lsr1,
  title         = {{L-SR1}: Learned Symmetric-Rank-One Preconditioning},
  author        = {Lifshitz, Gal and Zuler, Shahar and Fouks, Ori and Raviv, Dan},
  year          = {2025},
  eprint        = {2508.12270},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2508.12270},
}
```

Official ICML proceedings `@inproceedings` citation will be added here when available.

## Requirements

- Linux recommended
- NVIDIA GPU with CUDA (environment below uses CUDA 11.7 via conda; see `environment.yaml` for pins)
- [Conda](https://docs.conda.io/) or Mamba

## Setup

### 1. Environment

```bash
conda env create -f environment.yaml
conda activate hmr
```

### 2. Data

1. Follow dataset preparation from the [Learned Gradient Descent](https://github.com/InpatientJam/Learned-Gradient-Descent) repository where applicable.
2. Use the preprocessing scripts in `scripts/` if you need to build AMASS / 3DPW caches in this codebase’s format:
   - `scripts/preprocess_AMASS.py`
   - `scripts/preprocess_3DPW.py`

Set `dataset_root` to a directory that contains (after preprocessing):

| File | Used by |
|------|---------|
| `AMASS.npz` | Training |
| `3DPW_valid.npz` | Validation during training |
| `3DPW_test.npz` | Evaluation |

Paths are configured in `configs/train.yaml` and `configs/eval.yaml` via `dataset_root`.

### 3. Pretrained weights (evaluation)

The released checkpoint is included in this repository:

`checkpoints/lsr1__l4__best-model.ckpt` (~59&nbsp;MB)

Use it with `load_checkpoint` (see below). For other experiments, training still writes checkpoints under each Hydra run directory (`ckpts/`).

## Usage

### Training (AMASS)

Runs with Hydra; checkpoints are written under the Hydra run directory (see `configs/train.yaml` for `hydra.run.dir`).

```bash
python train.py dataset_root=/path/to/data/dir
```

Optional overrides, e.g. GPU index:

```bash
python train.py dataset_root=/path/to/data/dir cuda_devices=0
```

### Evaluation (3DPW)

```bash
python eval.py \
  load_checkpoint=checkpoints/lsr1__l4__best-model.ckpt \
  dataset_root=/path/to/data/dir
```

**Weights and biases:** Training and evaluation use `WandbLogger` if configured in code; set up [Weights & Biases](https://wandb.ai/) or adjust loggers in `train.py` / `eval.py` for offline/desired behavior.

### Standalone `LSR1Optimizer`

The learned symmetric-rank-one update lives in `lsr1/optimizer/`. It expects the same hyperparameter namespace as `model.lsr1` in the Hydra configs (`inner_buffer_size`, `inner_dim`, `alpha1`, `alpha2`, …). Usage pattern: **`reset()`** before a new solve; each step **`backward()`** on your loss, then **`forward(x.detach(), grad.detach())`** and add the returned update to `x` (see `LOPTModel.forward` in `lsr1/models/lsr1_baseline.py`).

A tiny driver (quadratic toy) is in `examples/minimal_lsr1.py`. From the repo root:

```bash
python -m examples.minimal_lsr1
```

## Repository layout

```
configs/          # Hydra configs (train / eval)
checkpoints/      # Released eval weights (lsr1__l4__best-model.ckpt)
examples/         # minimal_lsr1.py — toy use of LSR1Optimizer
lsr1/
  optimizer/      # L-SR1 algorithm (LSR1Optimizer)
  data/           # Dataset loaders
  models/         # LOPTModel (HMR + inner L-SR1 loops)
  utils/          # SMPL, losses, camera, …
scripts/          # AMASS / 3DPW preprocessing
train.py
eval.py
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
