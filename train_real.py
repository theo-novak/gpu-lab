"""train_real.py — the endgame lever: WGAN-GP on REAL SPY daily returns.

The BofA idea, actually run: train on real market data where NO closed-form
engine exists to arbitrate. Gate is pre-registered in evaluate_real.py
(R1 moving-block-bootstrap CIs, R2 beat iid-Gaussian on ACF/leverage,
R3 no memorization) — written to disk BEFORE this run, single look at 16k
iterations, matching the synthetic protocol.

Design notes, honest:
  - critic="conv" (the architecture winner: closest structure at 1/10 params)
  - center=True: subtract the EMPIRICAL per-step mean (an estimate from data
    in hand — no future information) and re-add it at generation.
    (First draft used center=False reasoning "estimate ≠ known constant" —
    wrong application: center=True is the same convention as every prior
    run, and generation re-adds mu automatically. center=False would leave
    generation mean-less while the real panel carries its sample mean — a
    spurious per-step-mean gap on the report card. Killed 2 min in; this
    docstring is the corrected record.)
  - data: SPY 2y daily -> make_panel(64, stride=1) = 437 overlapping panels.
    Effective sample ~7 independent 64-day stretches — the GAN trains on
    almost nothing, which is the POINT of the experiment (the BofA regime).
"""
from __future__ import annotations

import sys

import torch

import wganlib as wl
from realdata import fetch_returns, make_panel

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 16_000
SYMBOL = sys.argv[2] if len(sys.argv) > 2 else "SPY"


def main() -> int:
    r = fetch_returns(SYMBOL, years=2)
    panel = make_panel(r["returns"].float(), 64, 1).contiguous()
    print(f"{SYMBOL} 2y: {len(r['returns'])} returns -> panel {tuple(panel.shape)} | "
          f"center=True (empirical mean, re-added at gen) | conv critic | {ITERS} iters")

    tr = wl.Trainer(panel, critic="conv", center=True)
    tr.ckpt_path = wl.MODEL_DIR / "wgan_real_spy.pt"
    assert tr.iter == 0, "real-panel run must start fresh (or resume explicitly)"

    n_chunks = 4
    for _ in range(n_chunks):
        tr.train(ITERS // n_chunks)

    print("\n==== honest report card vs the real panel (same estimators) ====")
    tr.evaluate(real=panel, n_gen=panel.shape[0])  # same panel count, real vs gen
    return 0


if __name__ == "__main__":
    raise SystemExit(main())