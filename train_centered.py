"""Gate 1 retrain: fresh WGAN-GP with drift-centered data, then the verdict.

Trains the centered proven or big config, then prints the honest report card
and the economic test. Success criterion (|z| < 2 on the call price) ships
gate 1; anything else prints honestly.

Run: uv run python train_centered.py [iters] [proven|big|conv]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

import wganlib as wl

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 16_000
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "proven"
SPECS = {
    "proven": dict(hidden_g=256, hidden_d=512),
    "big": dict(hidden_g=1024, hidden_d=2048),
    "conv": dict(critic="conv"),   # G stays 3x256; conv critic ~60k params
}


def main() -> int:
    data, meta = wl.load_data()
    spec = SPECS[CONFIG]
    print(f"data {tuple(data.shape)} | centering ON | {ITERS} iters | {CONFIG} {spec}")

    # archive any existing notebook checkpoint so the run starts clean and
    # nothing is clobbered (the label records config + iters)
    if wl.MODEL_DIR.joinpath("wgan_notebook.pt").exists():
        arch = wl.MODEL_DIR / f"wgan_notebook_arch_{Path(wl.MODEL_DIR / 'wgan_notebook.pt').stat().st_size // 2**20}MB.bin"
        shutil.copy(wl.MODEL_DIR / "wgan_notebook.pt", arch)
        print(f"archived previous checkpoint -> {arch.name}")

    tr = wl.Trainer(data, depth=3, **spec)
    assert tr.iter == 0, "expected a fresh start; stale checkpoint matched?!"

    n_chunks = 4
    for _ in range(n_chunks):
        tr.train(ITERS // n_chunks)

    print("\n==== honest report card (centered run) ====")
    tr.evaluate(n_gen=100_000)

    print("\n==== economic test (the gate) ====")
    v = tr.price_call(n_paths=200_000, K=1.0)
    for k, val in v.items():
        print(f"  {k}: {val}")
    z = v["z (GAN vs engine)"]
    print(f"\ngate 1 verdict: {'PASS' if abs(z) < 2 else 'FAIL'} (|z| = {abs(z):.2f}, need < 2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())