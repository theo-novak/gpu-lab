"""evaluate_real.py — the real-panel gate, defined BEFORE training (pre-registration).

Teacher is gone: on real SPY returns there is no closed-form engine to
arbitrate the call price, so gate 1's |z| test is impossible here. The
honest replacement (pre-registered 2026-10-05, before any real-panel
training):

  R1  structure inside the data's own sampling CI:
      64-day block bootstrap of SPY (B=2000) -> 90% CI per estimator;
      the generator must land inside for roughness slope, ACF1(|r|), ACF5, leverage.
  R2  beat the Gaussian baseline, which FAILS it by construction:
      iid N(mu, sigma^2) panels have |r|-ACF ~ 0 (no clustering) and
      leverage ~ 0. The generator must clear those margins with margin.
  R3  no memorization / no collapse: no generated panel rounds to a real
      64-block, and per-panel variance must sit in [0.5x, 2x] the real one.
  R4  descriptive only (no gate): retargeted-vol ATM call prices vs BS,
      reported with MC errors. No |z| — no truth to z against.

Honest expectation stated ahead of results: ~441 overlapping 64-day windows
from ONE series are far fewer effective samples than the 100k synthetic
panels (~7 independent 64-day stretches). A wide CI is a LOW bar — passing
means "consistent with what this data can support", not book-ready.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

import wganlib as wl
from realdata import fetch_returns, make_panel

HERE_REAL = Path(__file__).resolve().parent

WINDOW = 64
STRIDE = 1
BLOCK_BOOT_B = 2000
BOOT_SEED = 2026
MODEL_PATH = HERE_REAL / "models" / "wgan_real_spy.pt"  # the real-panel checkpoint


def verbatim_check(gen: torch.Tensor, real_panel: torch.Tensor) -> int:
    """R3 memorization check: how many generated panels equal a real 64-day
    block under 4-decimal rounding. Expected 0 for a real generator."""
    gs = {tuple(torch.round(p, decimals=4).tolist()) for p in gen.cpu()}
    rs = {tuple(torch.round(p, decimals=4).tolist()) for p in real_panel.cpu()}
    return len(gs & rs)


def estimator_vector(panel: torch.Tensor) -> dict:
    """The estimators, computed the SAME way for real/Gaussian/GAN.

    kurtosis and logst_var are DESCRIPTIVE-only: the percentile bootstrap is
    unreliable for them (kurtosis 22.6 sits far in the tail of its own
    sampling distribution — v1's panel CI didn't even contain the real
    value). GATED = the four rows with trustworthy CIs.
    """
    m = wl.panel_metrics(panel.float())
    return dict(
        roughness=m["roughness_slope"],
        acf1=m["acf1"],
        acf5=m["acf5"],
        leverage=m["leverage"],
        kurtosis=m["kurtosis"],
        logst_var=(panel.sum(1) - panel.mean()).var(unbiased=False).item(),
    )


GATED = ("roughness", "acf1", "acf5", "leverage")


def bootstrap_ci(return_series: torch.Tensor, b: int = BLOCK_BOOT_B, seed: int = BOOT_SEED):
    """Moving-block bootstrap CI on the ORIGINAL 1D return series (block = 64,
    circular, B=2000, seed 2026): resample contiguous blocks of the series,
    rebuild the 64-day panel per replica, recompute estimators.

    (v1 of this harness bootstrapped panel ROWS of the overlapping-window
    panel instead — invalid: duplicating already-overlapping windows distorts
    the cross-window averaging, and the real point estimates landed OUTSIDE
    their own 90% CIs — roughness -0.511 vs [-0.507, -0.440]. Caught by the
    pre-registration run itself, 2026-10-05, before any GAN was judged.)
    """
    g = torch.Generator().manual_seed(seed)
    T = len(return_series)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    series = return_series.to(dev)
    n_blocks = math.ceil(T / WINDOW)
    stats = {k: [] for k in GATED}
    for start in range(0, b, 100):
        bt = min(100, b - start)
        for _ in range(bt):
            starts = torch.randint(0, T, (n_blocks,), generator=g).tolist()
            blocks = [(series[(s + i) % T] for i in range(WINDOW)) for s in starts]
            flat = torch.tensor([v for blk in blocks for v in blk][:T],
                                device=dev)
            v = estimator_vector(make_panel(flat, WINDOW, STRIDE))
            for k in GATED:
                stats[k].append(v[k])
    return {k: (lambda t: (t.quantile(0.05).item(), t.quantile(0.95).item()))(
        torch.tensor(stats[k])) for k in GATED}


def historical_block_baseline(series: torch.Tensor, n: int = 437,
                              seed: int = 43) -> dict:
    """Report-only comparator (added BEFORE the verdict, so no gate tuning):
    build 'synthetic' panels by stitching random contiguous 64-day blocks of
    the real series — historical simulation adapted to panel form. If the
    GAN cannot match the real estimators better than literal resampling of
    itself does, it has no reason to exist on this data (the BofA page's
    three-way logic, panel edition). R2's Gaussian is the parametric stand-in;
    this is the historical one."""
    g = torch.Generator().manual_seed(seed)
    T = len(series)
    n_blocks = math.ceil(T / WINDOW)
    out = torch.empty(n, WINDOW)
    for i in range(n):
        starts = torch.randint(0, T, (n_blocks,), generator=g).tolist()
        vals = [series[(s + j) % T].item() for s in starts for j in range(WINDOW)]
        out[i] = torch.tensor(vals[:WINDOW * 1][-WINDOW:])  # one panel = one window
    # stitched blocks give panels of ONE block each (a light historical sim);
    # estimators on (n, 64):
    return estimator_vector(out)


def gaussian_baseline(panel: torch.Tensor, n: int = 100_000, seed: int = 42) -> dict:
    """iid Gaussian panels — fails R2 by construction on the ACF rows.
    roughness is EXCLUDED for the baseline (iid -> ACF ~ 0 scattered around
    the 1e-6 log floor -> slope is floor-artifact garbage, the exact
    pathology documented in tests/test_wganlib.py's white-noise docstring)."""
    g = torch.Generator().manual_seed(seed)
    flat = panel.flatten().double()
    mu, sd = flat.mean().item(), flat.std().item()
    fake = torch.randn(n, panel.shape[1], generator=g) * sd + mu
    v = estimator_vector(fake)
    v["roughness"] = float("nan")
    return v


def main() -> int:
    r = fetch_returns("SPY", years=2)
    panel = make_panel(r["returns"].float(), WINDOW, STRIDE)
    print(f"SPY 2y: {len(r['returns'])} returns -> panel {tuple(panel.shape)} "
          f"(~{len(r['returns']) - WINDOW + 1 - (WINDOW - 1)} independent windows)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real = estimator_vector(panel)
    ci = bootstrap_ci(r["returns"].to(dev).float())
    gauss = gaussian_baseline(panel)
    hist = historical_block_baseline(r["returns"].to(dev).float())

    gate_model = None
    if MODEL_PATH.exists():
        tr = wl.Trainer.from_checkpoint(MODEL_PATH)
        gen = tr.generate_paths(panel.shape[0], seed=2099).float()
        gate_model = estimator_vector(gen)
        copies = verbatim_check(gen, panel)
        var_ratio = gen.var(unbiased=False).item() / panel.var(unbiased=False).item()
        print(f"\nmodel: {MODEL_PATH.name} | verbatim copies: {copies} "
              f"(R3 needs 0) | panel-var ratio {var_ratio:.2f} (R3 needs [0.5, 2.0])")

    print("\n==== PRE-REGISTERED GATES (written before training) ====")
    hdr = (f"{'gated':10s} {'real':>8s} {'moving-block 90% CI':>24s} "
           f"{'iid-Gauss':>10s} {'hist-sim':>9s}")
    if gate_model:
        hdr += f" {'GAN':>8s}  R1"
    print(hdr)
    n_pass = 0
    for k in GATED:
        lo, hi = ci[k]
        line = (f"{k:10s} {real[k]:8.3f}   [{lo:8.3f}, {hi:8.3f}]  "
                f"{gauss[k]:10.3f} {hist[k]:9.3f}")
        if gate_model:
            inside = lo <= gate_model[k] <= hi
            n_pass += inside
            line += f" {gate_model[k]:8.3f}  {'PASS' if inside else 'FAIL'}"
        print(line)
    print("\ndescriptive (no gate — bootstrap CI unreliable for these):")
    for k in ("kurtosis", "logst_var"):
        extra = f"  | GAN {gate_model[k]:8.3f}" if gate_model else ""
        print(f"  {k}: real {real[k]:9.3f}{extra}")
    print("\ngate R1: generator metric inside CI per gated row")
    print("gate R2: generator must beat iid-Gaussian toward the real value (acf/lev rows)")
    print("gate R3: no verbatim 64-block copies; per-panel var in [0.5x, 2.0x]")
    print("gate R4 (descriptive, no gate): ATM call vs BS at retargeted vol")
    if gate_model:
        copies_note = "OK" if copies == 0 else "FAIL"
        r3 = "PASS" if (copies == 0 and 0.5 <= var_ratio <= 2.0) else "FAIL"
        print(f"\nR1 verdict: {n_pass}/4 inside CI | R3: {r3} ({copies_note}, "
              f"var ratio {var_ratio:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())