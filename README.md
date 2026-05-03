# L-SR1: Learned Symmetric-Rank-One Preconditioning

Official implementation accompanying the ICML paper **L-SR1: Learned Symmetric-Rank-One Preconditioning**.

**Authors:** Gal Lifshitz, Shahar Zuler, Ori Fouks, Dan Raviv · **Code:** [github.com/gallif/lsr1](https://github.com/gallif/lsr1) · **Project page:** [gallif.github.io/lsr1](https://gallif.github.io/lsr1) · **Contact:** [gallif](https://github.com/gallif)

**Paper:** TBD (under construction).

<!--
  TODO (maintainer — not rendered on GitHub): add paper URL above when public; sync docs/index.html (PAPER_HREF) and teaser at docs/assets/teaser.png; refresh abstract / BibTeX in index.html; README citation block when proceedings are final.
-->

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{lsr12026,
  title     = {L-SR1: Learned Symmetric-Rank-One Preconditioning},
  author    = {Lifshitz, Gal and Zuler, Shahar and Fouks, Ori and Raviv, Dan},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```

Full proceedings metadata (`volume`, `pages`, `url` / `doi`) — **TBD**.

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
  model.inner_num_steps=12 \
  dataset_root=/path/to/data/dir
```

Adjust `model.inner_num_steps` and other `model.*` options to match the checkpoint.

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
