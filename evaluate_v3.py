"""evaluate_v3.py — v3's gate harness: per-checkpoint eval + multi-seed claims.

REUSES the frozen v2 contract pieces (imports from evaluate_real/evaluate_real_v2,
which are committed history and may not change post-verdict). v3 adds:

  1. MULTI-SEED CLAIMS RULE: a lever's R5 margin is claimed at the MEDIAN of
     its seeds (>=3 when available); spread (max-min) reported alongside.
     Rationale: the rebuild nondeterminism incident (|z| -15.8 -> -34.1) —
     single-draw magnitudes are not claims.
  2. NOISE-FLOOR MODE (--floor): evaluate two FRESH training seeds of run C's
     recipe (v3_c_s3 / v3_c_s7) and report the R5-margin spread ACROSS SEEDS
     — the bar any new lever must clear to be called signal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import wganlib as wl
from evaluate_real import (
    BOOT_SEED, HERE_REAL, STRIDE, WINDOW, bootstrap_ci, estimator_vector,
    verbatim_check,
)
from evaluate_real_v2 import DRAWS, build_hist_draws, daily_tail_stats
from realdata import fetch_returns, make_panel

HERE = Path(__file__).resolve().parent
V3_MODELS = HERE / "models"


def gate_ckpt(path: Path, panel: torch.Tensor, flat: torch.Tensor,
              real_tails: dict, hist_tails: dict, ci: dict,
              real_kurt: float) -> dict:
    """One checkpoint through the frozen R1/R3/R5/R6 gates. Returns a dict row."""
    tr = wl.Trainer.from_checkpoint(path, data=panel)
    gen = tr.generate_paths(50_000, seed=2099).float()
    gm = estimator_vector(gen)
    copies = verbatim_check(gen, panel)
    var_ratio = gen.var(unbiased=False).item() / panel.var(unbiased=False).item()
    gd = gen.flatten()
    gt = daily_tail_stats(gd)
    kurt = ((((gd - gd.mean()) / gd.std()) ** 4).mean().item())
    d_real_hi = abs(gt["q99_9_high"] - real_tails["q99_9_high"])
    d_hist_hi = abs(hist_tails["q99_9_high"] - real_tails["q99_9_high"])
    d_real_lo = abs(gt["q99_9_low"] - real_tails["q99_9_low"])
    d_hist_lo = abs(hist_tails["q99_9_low"] - real_tails["q99_9_low"])
    beyond = (gd.abs() > flat.abs().max()).float().mean().item()
    r1 = {k: bool(ci[k][0] <= gm[k] <= ci[k][1]) for k in ("roughness", "acf1", "acf5", "leverage")}
    row = dict(
        model=path.name,
        r1_pass=all(r1.values()), r1=r1,
        var_ratio=var_ratio, verbatim=copies, r3=bool(copies == 0 and 0.5 <= var_ratio <= 2.0),
        d_q999_high=d_real_hi, d_hist_high=d_hist_hi,
        d_q999_low=d_real_lo, d_hist_low=d_hist_lo,
        r5=bool(d_real_hi < d_hist_hi and d_real_lo < d_hist_lo and beyond > 0),
        beyond_max=beyond, kurt=kurt, r6=bool(kurt <= 1.5 * real_kurt),
        margins_pp=dict(high=round((d_real_hi - d_hist_hi) * 100, 3),
                        low=round((d_real_lo - d_hist_lo) * 100, 3)),
    )
    del tr
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return row


def summarize(levers: dict[str, list[dict]]) -> None:
    """The claims rule: median margin across seeds, spread reported."""
    print("\n==== V3 CLAIMS (median across seeds; spread = max-min) ====")
    print(f"{'lever':10s} {'n':>2s} {'med_hi':>7s} {'spr_hi':>7s} "
          f"{'med_lo':>7s} {'spr_lo':>7s} {'pass':>5s}")
    for lev, rows in levers.items():
        hi = sorted(rr["margins_pp"]["high"] for rr in rows)
        lo = sorted(rr["margins_pp"]["low"] for rr in rows)
        med_hi = hi[len(hi) // 2]
        med_lo = lo[len(lo) // 2]
        spr_hi = hi[-1] - hi[0]
        spr_lo = lo[-1] - lo[0]
        passed = sum(r["r5"] and r["r6"] and r["r3"] and r["r1_pass"] for r in rows)
        print(f"{lev:10s} {len(rows):2d} {med_hi:7.3f} {spr_hi:7.3f} "
              f"{med_lo:7.3f} {spr_lo:7.3f} {passed}/{len(rows)}")
    print("(margin = GAN-vs-real minus hist-vs-real, in pp; NEGATIVE = GAN closer = the lever worked)")


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = fetch_returns("SPY", years=10)
    rets = r["returns"].float().to(dev)
    panel = make_panel(rets, WINDOW, STRIDE).contiguous()
    flat = rets.flatten()
    real_v = estimator_vector(panel)
    ci = bootstrap_ci(rets, block=252)
    real_tails = daily_tail_stats(flat)
    hist_tails = daily_tail_stats(build_hist_draws(flat, DRAWS, 43).to(dev))
    real_kurt = real_v["kurtosis"]

    levers: dict[str, list[dict]] = {}
    for lev in ("c", "tailcrit", "termq"):
        rows = [gate_ckpt(p, panel, flat, real_tails, hist_tails, ci, real_kurt)
                for p in sorted(V3_MODELS.glob(f"v3_{lev}_s*.pt"))]
        if rows:
            levers[lev] = rows

    for rows in levers.values():
        for rr in rows:
            tag = " ".join(f"{k}:{'P' if v else 'F'}" for k, v in rr["r1"].items())
            print(f"\n[{rr['model']}] R5 {'PASS' if rr['r5'] else 'FAIL'} "
                  f"(margins {rr['margins_pp']}) | beyond {rr['beyond_max']:.6f} | "
                  f"var {rr['var_ratio']:.2f} | kurt {rr['kurt']:.1f} | R1 {tag}")
    if levers:
        summarize(levers)
    else:
        print("no v3 checkpoints yet (models/v3_*.pt) — contract-only mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())