"""train_real_v2_b.py — run B of the pre-declared ladder: run A + variance
auxiliary loss (lambda_aux=1.0, scale-free log-ratio form).

Purpose under the contract: test whether DIRECT variance shaping propagates
to the marginal tail shape (q99.9, beyond-max mass) — moments the aux does
NOT target. If it does not propagate, run C (tail-weighted sampling) is
the last rung; if it does and R5/R6 clear, the latch is found.

Run: uv run python train_real_v2_b.py [iters]
"""
from __future__ import annotations

import sys

import torch

import wganlib as wl
from realdata import fetch_returns, make_panel

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 16_000
AUX = 1.0  # declared: unit weight — the log-ratio form is already O(1)


def main() -> int:
    r = fetch_returns("SPY", years=10)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = make_panel(r["returns"].float(), 64, 1).contiguous().to(dev)
    print(f"SPY 10y panel {tuple(panel.shape)} on {dev} | conv critic | "
          f"center=True | aux_var={AUX} | {ITERS} iters")

    tr = wl.Trainer(panel, critic="conv", center=True, aux_var=AUX)
    tr.ckpt_path = wl.MODEL_DIR / "wgan_real_spy_10y_aux.pt"
    assert tr.iter == 0, "v2 run B must start fresh"

    for _ in range(4):
        tr.train(ITERS // 4)

    print("\n==== honest report card vs the real 10y panel ====")
    tr.evaluate(real=panel, n_gen=panel.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())