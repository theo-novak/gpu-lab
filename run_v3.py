"""run_v3.py — v3 ladder runner: ONE lever x ONE seed -> a named checkpoint.

Pre-registered levers (contract header in evaluate_v3.py; amend there, never
post-verdict):
  c        run C's recipe verbatim (tail-weighted sampling, gamma=1) at a
           fresh training seed  -> the NOISE FLOOR when run at new seeds
  tailcrit V3-1: tail-weighted CRITIC loss (W1 mass emphasis on tail regions;
           the spirit-preserving BofA lever — needs wganlib support, lands
           with Bucket 2)
  termq    V3-2: terminal-law aux (quantile/variance matching on log S_T;
           the sledgehammer — caveat: trains the terminal law directly, so
           terminal-law gate rows become "trained-on"; needs wganlib support,
           lands with Bucket 2)

Run: uv run python run_v3.py <lever> <seed> [iters]
Out: models/v3_<lever>_s<seed>.pt
"""
from __future__ import annotations

import sys

import torch

import wganlib as wl
from realdata import fetch_returns, make_panel

ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 16_000
LEVER = sys.argv[1] if len(sys.argv) > 1 else "c"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 2026


def main() -> int:
    r = fetch_returns("SPY", years=10)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = make_panel(r["returns"].float(), 64, 1).contiguous().to(dev)
    print(f"v3 lever={LEVER} seed={SEED} iters={ITERS} panel {tuple(panel.shape)} on {dev}")

    if LEVER == "c":
        kwargs: dict = dict(critic="conv", center=True, tail_gamma=1.0)
    elif LEVER in ("tailcrit", "termq"):
        raise SystemExit(f"lever '{LEVER}' lands with Bucket 2 (wganlib support pending)")
    else:
        raise SystemExit(f"unknown lever {LEVER}")

    tr = wl.Trainer(panel, seed=SEED, **kwargs)
    tr.ckpt_path = wl.MODEL_DIR / f"v3_{LEVER}_s{SEED}.pt"
    assert tr.iter == 0, f"v3_{LEVER}_s{SEED} must start fresh"

    for _ in range(4):
        tr.train(ITERS // 4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())