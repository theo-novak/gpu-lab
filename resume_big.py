"""Resume the interrupted big-config centered run to 16k, then the verdict.

The 16k big run (G 1024 / D 2048, drift-centered) was killed at iter 5000
by a session close — before its verdict. This finishes the declared
experiment: resume from models/wgan_notebook_big_5k_interrupted.pt, train
to 16k, then print the honest report card and the economic test (mean
enforcement ON, as shipped in wganlib.generate_paths).

Saves land on models/wgan_notebook.pt (the ACTIVE slot) — the
proven-centered 16k checkpoint stays safe at models/wgan_notebook_arch_8MB.bin.

Run: uv run python resume_big.py > resume_big.log 2>&1
"""
from __future__ import annotations

import torch

import wganlib as wl

CKPT = wl.MODEL_DIR / "wgan_notebook_big_5k_interrupted.pt"
TARGET = 16_000


def main() -> int:
    data, _ = wl.load_data()
    tr = wl.Trainer.from_checkpoint(CKPT, data=data)
    assert tr.config["hidden_g"] == 1024 and tr.config["hidden_d"] == 2048, "not the big config"
    assert tr.config["center"], "checkpoint is not the centered run"
    assert tr.iter == 5000, f"expected iter 5000, got {tr.iter}"
    print(f"resumed: iter {tr.iter} | G/D {tr.config['hidden_g']}/{tr.config['hidden_d']} "
          f"| center={tr.config['center']} | target {TARGET}")

    while tr.iter < TARGET:
        tr.train(min(4000, TARGET - tr.iter))

    print("\n==== honest report card (big, centered, 16k iters) ====")
    tr.evaluate(n_gen=100_000)

    print("\n==== economic test (the gate) ====")
    v = tr.price_call(n_paths=200_000, K=1.0)
    for k, val in v.items():
        print(f"  {k}: {val}")
    z = v["z (GAN vs engine)"]
    print(f"\ngate 1 verdict: {'PASS' if abs(z) < 2 else 'FAIL'} "
          f"(|z| = {abs(z):.2f}, need < 2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())