"""evaluate_real_v2.py — v2's pre-registered gate, written BEFORE training.

Data regime (Bucket 1, verified): SPY 10y daily = 2,513 returns
(2016-10 .. 2026-10) -> 2,450 overlapping / ~2,386 independent 64-day
windows, vs 2y's ~7. COVID in-sample by design (-11.59% 2020-03-16).

The v2 question (declared 2026-10-05, before any v2 training):
  does the path-GAN beat historical block-resampling precisely BEYOND the
  observed tail — the one place resampling is structurally blind?
Historical resampling CANNOT produce a daily return beyond the sample
extreme, ever. The GAN can. If it cannot either (or only by exploding),
the free baseline wins again and "resample, don't train" becomes the
measured claim for daily single-name scenarios at this horizon.

GATES (unchanged from v1's contract where marked =):
  R1 =  generator estimators inside 90% moving-block CIs (block=64,
        circular, B=2000), rows: roughness/acf1/acf5/leverage.
  R2 =  beat iid-Gaussian on acf1/acf5/leverage (Gaussian ~ 0).
  R3 =  zero verbatim 64-blocks AND panel-var ratio in [0.5, 2.0].
  R4 =  descriptive only (no arbiter on real data).
  R5 (NEW, headline): equal-draw tail comparison (3.2M daily draws each):
        |GAN q99.9 - real q99.9| < |hist q99.9 - real q99.9|
        AND GAN P(daily |r| > real sample max) > 0 with the generated
        beyond-max draws not absurd (R6).
     win condition is explicit: the GAN must be CLOSER to the real tail
     quantile than the resampler, and put real mass beyond the observed
     extreme — resampling puts exactly 0.
  R6 (NEW, defensive): generated daily kurtosis <= 1.5x real (17.9 -> <=
     26.9) — a GAN "winning" R5 by exploding tails is a fail here.

TRAINING LADDER (pre-declared order, one look per run, 16k single-look):
  run A: plain conv retrain at 10y  (does scale alone fix dispersion?)
  run B: run A + variance aux loss  (legal training aid; the gate then
         judges kurtosis + tail quantiles, which the aux does NOT target
         directly — else the test is self-fulfilling)
  run C: tail-weighted batch sampling (extreme days oversampled)
  stop at the first run that passes R5+R6+R1..R3; if none pass, the
  negative result ships with the four-way table.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

import wganlib as wl
from evaluate_real import (
    BOOT_SEED, HERE_REAL, STRIDE, WINDOW, bootstrap_ci, estimator_vector,
    verbatim_check,
)
from realdata import fetch_returns, make_panel

DRAWS = 3_200_000
MODEL_ARGV = sys.argv[1] if len(sys.argv) > 1 else str(
    HERE_REAL / "models" / "wgan_real_spy_10y.pt"
)


def daily_tail_stats(draws: torch.Tensor) -> dict:
    """Tail stats of a pool of single-day returns: the v2 frontier."""
    q = torch.quantile(draws.to("cpu" if not draws.is_cuda else draws.device),
                       torch.tensor([0.001, 0.005, 0.995, 0.999],
                                    device=draws.device))
    return dict(q99_9_low=q[0].item(), q99_5_low=q[1].item(),
                q99_5_high=q[2].item(), q99_9_high=q[3].item())


def build_hist_draws(series: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Historical resampling, unlimited draws (the free baseline's tail is
    EXACTLY the empirical sample — beyond-max mass = 0 by construction)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(series), (n,), generator=g)
    return series[idx]


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = fetch_returns("SPY", years=10)
    rets = r["returns"].float().to(dev)
    panel = make_panel(rets, WINDOW, STRIDE).contiguous()
    real_v = estimator_vector(panel)
    # persistence-honest bootstrap: block=252 (one year) — the pre-flight
    # showed block=64 chopped vol dependence into seams and placed the REAL
    # acf1/acf5 outside their own CIs (see evaluate_real.bootstrap_ci docstring)
    ci = bootstrap_ci(rets, block=252)
    flat = rets.flatten()
    real_tails = daily_tail_stats(flat)
    g0 = torch.Generator().manual_seed(42)
    gauss_draws = (torch.randn(DRAWS, generator=g0) * flat.std().item()
                   + flat.mean().item()).to(dev)
    hist_draws = build_hist_draws(flat, DRAWS, 43).to(dev)
    hist_tails = daily_tail_stats(hist_draws)
    gauss_tails = daily_tail_stats(gauss_draws)

    print(f"SPY 10y: {len(rets)} returns -> panel {tuple(panel.shape)}")
    print("\n==== PRE-REGISTERED GATES v2 (unchanged contract + R5/R6) ====")
    print(f"{'gated':10s} {'real':>8s} {'90% CI':>24s} (R1/R2 as v1)")
    for k in ("roughness", "acf1", "acf5", "leverage"):
        lo, hi = ci[k]
        print(f"{k:10s} {real_v[k]:8.3f}   [{lo:8.3f}, {hi:8.3f}]")
    print(f"\ntail facts (real, 2513 obs): min {flat.min()*100:.2f}%  "
          f"max {flat.max()*100:.2f}%  q99.9(high) {real_tails['q99_9_high']*100:.2f}%")

    gate_model = None
    mp = Path(MODEL_ARGV)
    if mp.exists():
        tr = wl.Trainer.from_checkpoint(mp, data=panel)
        gen = tr.generate_paths(50_000, seed=2099).float()
        gate_model = estimator_vector(gen)
        copies = verbatim_check(gen, panel)
        var_ratio = gen.var(unbiased=False).item() / panel.var(unbiased=False).item()
        gen_draws = gen.flatten()
        gen_tails = daily_tail_stats(gen_draws)
        gen_kurt = ((((gen_draws - gen_draws.mean()) / gen_draws.std()) ** 4)
                    .mean().item())
        d_real = abs(gen_tails["q99_9_high"] - real_tails["q99_9_high"])
        d_hist = abs(hist_tails["q99_9_high"] - real_tails["q99_9_high"])
        beyond = (gen_draws.abs() > flat.abs().max()).float().mean().item()
        r5 = "PASS" if (d_real < d_hist and beyond > 0) else "FAIL"
        r6 = "PASS" if gen_kurt <= 1.5 * real_v["kurtosis"] else "FAIL"
        print(f"\nmodel: {mp.name} | verbatim {copies} | var-ratio {var_ratio:.2f}")
        print(f"{'tails':14s} {'real':>8s} {'hist-sim':>9s} {'iid-Gauss':>10s} {'GAN':>8s}")
        for row in ("q99_9_high", "q99_5_high", "q99_5_low", "q99_9_low"):
            print(f"{row:14s} {real_tails[row]*100:8.2f} {hist_tails[row]*100:9.2f} "
                  f"{gauss_tails[row]*100:10.2f} {gen_tails[row]*100:8.2f}")
        print(f"\nP(|daily|>real max): hist-sim 0 BY CONSTRUCTION | GAN {beyond:.6f}")
        print(f"R5 verdict: {r5}  (GAN |Δq99.9|={d_real*100:.3f}pp vs hist {d_hist*100:.3f}pp; "
              f"beyond-max mass {'>0' if beyond > 0 else '=0'})")
        print(f"R6 verdict: {r6}  (gen kurt {gen_kurt:.1f} vs cap {1.5 * real_v['kurtosis']:.1f})")
        print(f"R1: " + " ".join(
            f"{k} {'PASS' if ci[k][0] <= gate_model[k] <= ci[k][1] else 'FAIL'}"
            for k in ("roughness", "acf1", "acf5", "leverage")))
        print(f"R3: var-ratio {'PASS' if 0.5 <= var_ratio <= 2.0 else 'FAIL'} "
              f"| verbatim {'PASS' if copies == 0 else 'FAIL'}")
    else:
        print(f"\n(no model at {mp.name} — contract-only run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())