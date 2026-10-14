"""mc_oracle.py — rough-vol Monte Carlo odds for arbitrary scenarios (Use #3).

The validated engine as an oracle: "what are the odds of X over a horizon,
under rough volatility (H=0.1) rather than GBM?" Calibrated to a ticker's
realized vol, or run at any vol level you name.

Questions it answers (any combination):
  --dd 10     P(max drawdown >= 10% within the window)
  --lose 5    P(total return <= -5%)
  --gain 20   P(total return >= +20%)
Horizon: --days N (any trading-day count; no grid cap — the engine is exact).

Honesty notes (printed with every run):
  * answers are under the rBergomi MODEL (H=0.1, eta=1.9, rho=-0.9,
    flat forward variance calibrated to realized vol) — a structured
    view of the world, not a forecast;
  * flat forward variance = no term structure; day 1 vol = day N vol;
  * single-asset: no cross-asset correlation.

Run:
  uv run python mc_oracle.py SPY --dd 10 --lose 5
  uv run python mc_oracle.py NVDA --days 126 --dd 20 --gain 50 --vol 45
  uv run python mc_oracle.py --vol 20 --dd 10            # pure vol-level mode
Flags: --days 64  --paths 65536  --vol-years 5  --seed 2026  --plot
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import rbergomi as rb
import realdata as rd

HERE = Path(__file__).resolve().parent
OUT = HERE / "report" / "oracle"


def argval(flag: str, default: float) -> float:
    if flag in sys.argv:
        try:
            return float(sys.argv[sys.argv.index(flag) + 1])
        except (ValueError, IndexError):
            return default
    return default


def has(flag: str) -> bool:
    return flag in sys.argv


def simulate_paths(S0: float, sigma: float, days: int, paths: int, seed: int):
    """Engine paths -> (n, days+1) PRICE paths incl. day 0."""
    T = days / 252.0
    n_steps = max(8, min(days, 252))
    dt = T / n_steps
    v, dW, _ = rb.simulate_variance(paths, n_steps, T, H=0.1, eta=1.9,
                                    xi0=lambda tt: torch.full_like(tt, sigma ** 2),
                                    seed=seed)
    r = rb.log_return_increments_dt(v, dW, rho=-0.9, dt=dt, seed=seed + 1)
    cum = r.cumsum(1)                                        # per-path cumulative log-return
    logp = torch.nn.functional.pad(cum, (1, 0), value=0.0)   # prepend day-0 -> (n, days+1)
    return S0 * torch.exp(logp)


def max_drawdown(p: torch.Tensor) -> torch.Tensor:
    return ((p - p.cummax(1).values) / p.cummax(1).values).min(1).values  # <=0


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want = {"dd": has("--dd"), "lose": has("--lose"), "gain": has("--gain")}
    if not any(want.values()):
        print("nothing to compute — pass --dd / --lose / --gain (see --help)")
        return 1

    days = int(argval("--days", 64))
    paths = int(argval("--paths", 65536))
    seed = int(argval("--seed", 2026))

    if args and not has("--vol"):
        sym = args[0].upper()
        vol_years = int(argval("--vol-years", 5))
        hist = rd.fetch_returns(sym, years=vol_years)
        sigma = float(hist["returns"].float().std(unbiased=True) * math.sqrt(252))
        S0 = 1.0
        label = f"{sym} (realized vol {sigma*100:.1f}% from {vol_years}y)"
    else:
        sym = f"VOL{argval('--vol', 20):.0f}"
        sigma = argval("--vol", 20) / 100.0
        S0 = 1.0
        label = f"a generic name at {sigma*100:.0f}% vol"

    print(f"rough-vol oracle — {label}")
    print(f"  horizon {days} trading days | {paths:,} engine paths (H=0.1, eta=1.9, rho=-0.9, seed {seed})")

    P = simulate_paths(S0, sigma, days, paths, seed)
    tot = (P[:, -1] / S0 - 1.0) * 100            # total return %
    mdd = max_drawdown(P) * 100                   # <= 0

    out = []
    if has("--dd"):
        x = argval("--dd", 10)
        p = float((mdd <= -x).float().mean()) * 100
        e = 100 * 1.96 * math.sqrt(max(p / 100 * (1 - p / 100), 1e-12) / paths)
        out.append((f"P(maxDD >= {x:g}% in {days}d)", p, e, (mdd <= -x)))
        print(f"  P(max drawdown >= {x:g}%) = {p:.2f}%  (±{e:.2f} MC)")
    if has("--lose"):
        x = argval("--lose", 5)
        p = float((tot <= -x).float().mean()) * 100
        e = 100 * 1.96 * math.sqrt(max(p / 100 * (1 - p / 100), 1e-12) / paths)
        out.append((f"P(total <= -{x:g}%)", p, e, (tot <= -x)))
        print(f"  P(total return <= -{x:g}%) = {p:.2f}%  (±{e:.2f} MC)")
    if has("--gain"):
        x = argval("--gain", 20)
        p = float((tot >= x).float().mean()) * 100
        e = 100 * 1.96 * math.sqrt(max(p / 100 * (1 - p / 100), 1e-12) / paths)
        out.append((f"P(total >= +{x:g}%)", p, e, (tot >= x)))
        print(f"  P(total return >= +{x:g}%) = {p:.2f}%  (±{e:.2f} MC)")

    if has("--plot"):
        OUT.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        pn = P[:200].cpu().T.numpy()
        ax.plot(pn, color="#8a93b5", lw=0.4, alpha=0.15)
        for q, c, lab in ((0.05, "#c0455a", "p5"), (0.5, "#3ddad7", "median"), (0.95, "#5fae6a", "p95")):
            band = torch.quantile(P.cpu(), q, dim=0).numpy()
            ax.plot(band, color=c, lw=1.6, label=lab)
        ax.set_title(f"rBergomi paths — {label}, {days}d, {paths:,} paths", fontsize=11)
        ax.set_xlabel("trading day"); ax.set_ylabel("price (S0 = 1)")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
        note = ("rBergomi model view (H=0.1, eta=1.9, rho=-0.9, flat xi0) — "
                "a structured assumption, not a forecast")
        ax.annotate(note, (0.01, -0.13), xycoords="axes fraction", fontsize=7.5, color="#888")
        from datetime import date as _d
        fp = OUT / f"oracle_{sym}_{days}d_{_d.today():%Y%m%d}.png"
        fig.savefig(fp, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {fp}")

    print("\n  honesty: model-view probabilities under rough vol, flat forward variance;"
          "\n  not a forecast; single-asset (no correlations); horizon = trading days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())