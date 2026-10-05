"""Shared fixtures for the gpu-lab test suite.

Design rules (the project's honest-benchmarks formula):
- CPU-only, fast, seeded — every test deterministic, no network, no GPU
  requirement (CUDA paths get exercised by the notebook/scripts, not here).
- Tests assert against EXACT/analytic references where one exists, never
  hand-waved constants — the hard-won lesson from build_dataset.py.
"""
from __future__ import annotations

import pytest
import torch

import wganlib as wl

# tiny deterministic dataset fixture values (synthetic, exact by construction)
N_PATHS = 512
N_STEPS = 8


@pytest.fixture(scope="session")
def small_data():
    """Seeded synthetic returns with KNOWN per-step mean and std.

    mean_i = -0.001*(i+1), std_i = 0.02*(1 + i/8) — inhomogeneous on
    purpose so per-step operations can't hide behind homogeneity.
    """
    g = torch.Generator().manual_seed(2026)
    e = torch.randn(N_PATHS, N_STEPS, generator=g)
    mean = -0.001 * (torch.arange(1, N_STEPS + 1).float())
    std = 0.02 * (1 + torch.arange(N_STEPS).float() / N_STEPS)
    return mean + std * e


@pytest.fixture(scope="session")
def tiny_trainer(small_data):
    """A tiny CPU trainer, trained a few iters — shared, read-only-ish."""
    tr = wl.Trainer(
        small_data, hidden_g=16, hidden_d=32, depth=2, noise_dim=4,
        n_critic=1, batch=32, seed=7,
    )
    tr.train(3, log_every=1, ckpt_every=10**9)  # no mid-run checkpoints
    return tr