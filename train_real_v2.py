"""train_real_v2.py — run A of the pre-declared v2 ladder: plain conv retrain
at 10y. One look at 16k, judged by evaluate_real_v2.py's committed contract
(ba090e2, written before this run). If scale alone fixes dispersion, runs
B/C never happen.

Run: uv run python train_real_v2.py [iters]
"""
from __future__ import annotations

import sys

import torch

import wganlib as wl
from realdata import fetch_returns, make_panel

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 16_000


def main() -> int:
    r = fetch_returns("SPY", years=10)
    # GPU explicitly (the Oct-6 lesson: Trainer follows its data's device)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = make_panel(r["returns"].float(), 64, 1).contiguous().to(dev)
    print(f"SPY 10y panel {tuple(panel.shape)} on {dev} | conv critic | "
          f"center=True | {ITERS} iters")

    tr = wl.Trainer(panel, critic="conv", center=True)
    tr.ckpt_path = wl.MODEL_DIR / "wgan_real_spy_10y.pt"
    assert tr.iter == 0, "v2 run A must start fresh"

    for _ in range(4):
        tr.train(ITERS // 4)

    print("\n==== honest report card vs the real 10y panel ====")
    tr.evaluate(real=panel, n_gen=panel.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())