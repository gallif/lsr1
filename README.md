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

`eval.py` instantiates the model from `configs/eval.yaml`’s `model` block (not from hyperparameters inside the `.ckpt`). The defaults match the released checkpoint; if you use another checkpoint, update `model.*` in the config or via CLI overrides to match that run.

**Weights and biases:** Training and evaluation use `WandbLogger` if configured in code; set up [Weights & Biases](https://wandb.ai/) or adjust loggers in `train.py` / `eval.py` for offline/desired behavior.

## Repository layout

```
configs/          # Hydra configs (train / eval)
checkpoints/      # Released eval weights (lsr1__l4__best-model.ckpt)
lsr1/
  data/           # Dataset loaders
  models/         # Model definition
  utils/          # L-SR1 optimizer, SMPL, losses, etc.
scripts/          # AMASS / 3DPW preprocessing
train.py
eval.py
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
