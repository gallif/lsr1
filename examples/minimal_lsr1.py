#!/usr/bin/env python3
"""
Minimal L-SR1 inner step (same API as in `lsr1.models.lsr1_baseline`).

From the repository root (directory that contains `lsr1/` and `examples/`):

  python -m examples.minimal_lsr1

  python examples/minimal_lsr1.py

For each outer iteration: set `x` to require grads, compute your scalar loss,
`backward()`, then pass `(x.detach(), x.grad.detach())` into the optimizer.
Call `reset()` before a new sequence (new batch or new solve).
"""
from __future__ import annotations

import sys
from pathlib import Path

# So `python examples/minimal_lsr1.py` finds the `lsr1` package when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import types

import torch

from lsr1.optimizer import LSR1Optimizer


def lsr1_args_like_config() -> types.SimpleNamespace:
    """Hyperparameters matching `configs/*.yaml` under `model.lsr1`."""
    return types.SimpleNamespace(
        inner_ang_t=0.01,
        inner_dir_vec_eps=1e-10,
        inner_buffer_size=4,
        inner_l_arch="mlp",
        inner_dim=128,
        alpha1=0.1,
        alpha2=0.001,
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, dim = 4, 72
    inner = LSR1Optimizer(lsr1_args_like_config()).to(device)
    inner.train()
    inner.reset()

    x = torch.randn(batch, dim, device=device, requires_grad=True)
    loss = (x**2).sum()
    loss.backward()

    update, _sec, lr = inner(x.detach(), x.grad.detach())
    print("update shape:", tuple(update.shape))
    print("done")


if __name__ == "__main__":
    main()
