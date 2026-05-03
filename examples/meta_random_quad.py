#!/usr/bin/env python3
"""
Standalone example: meta-training of `LSR1Optimizer` on random quadratics.

Each outer step samples a batch of convex quadratics; an inner loop unrolls the learned
optimizer with a weighted trajectory loss plus optional secant consistency. A fixed
validation batch is kept in memory for periodic inner-loop diagnostics (no checkpoints).

From the repository root::

    python -m examples.meta_random_quad --help

See the README for a diagram of inner vs. outer backward passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim

# Repo root (directory containing `lsr1/`) on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lsr1.optimizer import LSR1Optimizer

MAX = 1e30


# --- Random quadratic batch objectives ---


class RandomQuadraticFunction(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        A = torch.randn(shape[0], shape[0])
        H = A.T @ A
        H = H / (H**2).sum().sqrt()
        b_ = torch.randn(shape)
        b = b_ / (b_**2).sum().sqrt()
        self.register_buffer("H", H)
        self.register_buffer("b", b)
        opt = -torch.linalg.inv(self.H) @ self.b
        self.register_buffer("optimizer_point", opt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Hx = self.H @ x
        return 0.5 * (x * Hx).sum() + (self.b * x).sum()


class AggrQuadObjectives(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        # shape = (batch, dim)
        modules = [RandomQuadraticFunction(shape[1:]) for _ in range(shape[0])]
        self.objs = nn.ModuleList(modules)
        opts = torch.stack([o.optimizer_point for o in self.objs])
        self.register_buffer("optimizer", opts)

    def forward(self, x: torch.Tensor):
        vals = [obj_(x_) for obj_, x_ in zip(self.objs, x)]
        return torch.stack(vals).mean(), vals


# --- Meta losses: weighted trajectory loss + secant consistency ---


def unsup_meta_loss(losses: list[torch.Tensor], gamma: float) -> torch.Tensor:
    n = len(losses)
    ws = torch.tensor(
        [gamma ** (n - 1 - i) for i in range(n)],
        device=losses[0].device,
        dtype=losses[0].dtype,
    )
    return (ws * torch.stack(losses)).sum() / n


def secant_meta_loss(dirs: list[tuple[torch.Tensor, torch.Tensor]], gamma: float) -> torch.Tensor:
    n = len(dirs)
    terms = torch.stack([((bkdg - dx) ** 2).mean() for bkdg, dx in dirs])
    ws = torch.tensor(
        [gamma ** (n - 1 - i) for i in range(n)],
        device=terms.device,
        dtype=terms.dtype,
    )
    return (ws * terms).sum() / n


def lsr1_config_ns(
    buffer_size: int,
    inner_dim: int,
    alpha1: float,
    alpha2: float,
    inner_ang_t: float = 0.01,
    inner_dir_vec_eps: float = 1e-10,
) -> SimpleNamespace:
    return SimpleNamespace(
        inner_ang_t=inner_ang_t,
        inner_dir_vec_eps=inner_dir_vec_eps,
        inner_buffer_size=buffer_size,
        inner_l_arch="mlp",
        inner_dim=inner_dim,
        alpha1=alpha1,
        alpha2=alpha2,
    )


def inner_loop_learned(
    inner: LSR1Optimizer,
    inner_obj: AggrQuadObjectives,
    x0: torch.Tensor,
    *,
    iters: int,
    train_mode: bool,
) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Unroll inner steps. Gradients are scaled by batch size so each term matches sum-reduction."""
    inner.train()
    inner.reset()
    grad_scale = x0.shape[0]
    x = x0.detach().clone().requires_grad_(True)
    x.retain_grad()
    meta_losses: list[torch.Tensor] = []
    dirs_out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(iters):
        loss, _ = inner_obj(x)
        loss = torch.clamp(loss, max=MAX)
        loss.backward(retain_graph=True)
        update, (bkdg, dx), _lr = inner(x.detach(), x.grad.detach() * grad_scale)
        if train_mode:
            x = x + update
            x.retain_grad()
        else:
            x = (x + update).detach().requires_grad_(True)
            x.retain_grad()
        meta_losses.append(loss)
        dirs_out.append((bkdg, dx))
    return meta_losses, dirs_out, x


def train_main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    dim = args.dim
    box = args.boxlim
    inner = LSR1Optimizer(
        lsr1_config_ns(
            args.buffer_size,
            args.inner_dim,
            args.alpha1,
            args.alpha2,
        )
    ).to(device)
    meta_opt = optim.AdamW(inner.parameters(), lr=args.outer_lr, betas=(0.9, 0.999))

    x0_valid = (box * (2 * torch.rand(args.valid_batch, dim, device=device) - 1)).requires_grad_(True)
    val_obj = AggrQuadObjectives((args.valid_batch, dim)).to(device)

    for it in range(args.iters):
        x0 = (box * (2 * torch.rand(args.train_batch, dim, device=device) - 1)).requires_grad_(True)
        train_obj = AggrQuadObjectives(x0.shape).to(device)

        losses, dirs, _ = inner_loop_learned(
            inner,
            train_obj,
            x0,
            iters=args.unroll,
            train_mode=True,
        )
        ml = unsup_meta_loss(losses, args.gamma)
        mc = secant_meta_loss(dirs, args.gamma) if args.w_sec > 0 else torch.zeros(1, device=device)
        total = args.w_unsup * ml + args.w_sec * mc

        meta_opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), max_norm=args.grad_clip)
        meta_opt.step()

        if (it + 1) % args.log_every == 0 or it == 0:
            # inner_loop_learned calls loss.backward — must run outside torch.no_grad().
            _, _, x_end = inner_loop_learned(
                inner,
                val_obj,
                x0_valid,
                iters=args.eval_iters,
                train_mode=False,
            )
            with torch.no_grad():
                dist = torch.stack(
                    [((xi - oi) ** 2).sum().sqrt() for xi, oi in zip(x_end.detach(), val_obj.optimizer)]
                ).mean()
            print(
                f"[{it+1}/{args.iters}] meta_loss={total.item():.4f} "
                f"unsup={ml.item():.4f} sec={mc.item():.4f} "
                f"val_dist_mean={dist.item():.4f}"
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Meta-train LSR1Optimizer on random quadratics (in-memory val; no checkpoint I/O)."
    )
    p.add_argument("--buffer-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--boxlim", type=float, default=5.12)
    p.add_argument("--iters", type=int, default=400, help="outer / meta iterations")
    p.add_argument("--unroll", type=int, default=8, help="inner steps per meta step")
    p.add_argument("--train-batch", type=int, default=32)
    p.add_argument("--valid-batch", type=int, default=16)
    p.add_argument("--eval-iters", type=int, default=20, help="inner steps during periodic val")
    p.add_argument("--w-unsup", type=float, default=1.0)
    p.add_argument("--w-sec", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=1.0, help="time weighting for trajectory + secant terms")
    p.add_argument("--outer-lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=0.01)
    p.add_argument("--inner-dim", type=int, default=128)
    p.add_argument("--alpha1", type=float, default=0.4)
    p.add_argument("--alpha2", type=float, default=0.001)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    train_main(args)


if __name__ == "__main__":
    main()
