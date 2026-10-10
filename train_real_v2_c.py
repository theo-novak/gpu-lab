"""train_real_v2_c.py — run C, the last rung of the pre-declared ladder:
tail-weighted batch sampling (gamma=1.0: prob ∝ 1 + (winmax/|r|99)^2).

The extreme window (COVID, |r|=11.6% ≈ 4× the 99th pct) draws ~17× more
often than a calm one. Purpose: feed the critic real tail mass directly —
the one lever neither scale (run A) nor variance shaping (run B) was.

Run: uv run python train_real_v2_c.py [iters]
"""
from __future__ import annotations

import sys

import torch

import wganlib as wl
from realdata import fetch_returns, make_panel

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 16_000
GAMMA = 1.0  # declared


def main() -> int:
    r = fetch_returns("SPY", years=10)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = make_panel(r["returns"].float(), 64, 1).contiguous().to(dev)
    print(f"SPY 10y panel {tuple(panel.shape)} on {dev} | conv critic | "
          f"center=True | tail_gamma={GAMMA} | {ITERS} iters")

    tr = wl.Trainer(panel, critic="conv", center=True, tail_gamma=GAMMA)
    tr.ckpt_path = wl.MODEL_DIR / "wgan_real_spy_10y_tail.pt"
    assert tr.iter == 0, "v2 run C must start fresh"

    for _ in range(4):
        tr.train(ITERS // 4)

    print("\n==== honest report card vs the real 10y panel ====")
    tr.evaluate(real=panel, n_gen=panel.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())