"""scenario_cli.py — resampled scenario bundles from real returns.

The tool that beat the GAN (v2/v3 verdict): historical block-resampling.
Feed it tickers; it stamps out thousands of plausible 64-day futures per
ticker with vol clustering intact, draws the path bundles, and prints a
mini risk table (drawdowns, tail percentiles). No model, no keys, no GPU.

Honesty note (measured 2026-10-10): resampling cannot emit a day beyond
its own sample — that is its known blindness (the GAN experiment showed
the trained model CAN emit such days). For calibrated drawdown planning,
this is the tool.

Run:
  uv run python scenario_cli.py              # prompts for tickers
  uv run python scenario_cli.py SPY AAPL     # or pass them right away
Flags: --years 5   --paths 2000   --days 64   --seed 2026   --quiet
"""
from __future__ import annotations

import sys
import subprocess
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from realdata import fetch_returns

HERE = Path(__file__).resolve().parent
OUT = HERE / "report" / "scenarios"


def argval(flag: str, default: int) -> int:
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    return default


QUIET = "--quiet" in sys.argv
YEARS = argval("--years", 5)
PATHS = argval("--paths", 2000)
DAYS = argval("--days", 64)
SEED = argval("--seed", 2026)


def prompt_tickers() -> list[str]:
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        raw = sys.argv[1]
    else:
        try:
            raw = input("Ticker(s), comma-separated [SPY]: ").strip()
        except EOFError:
            raw = ""
        raw = raw or "SPY"
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def resample_paths(rets: torch.Tensor, n: int, days: int, seed: int) -> torch.Tensor:
    """Bootstrap n contiguous `days` windows (with replacement)."""
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(rets) - days + 1, (n,), generator=g)
    idx = starts.view(n, 1) + torch.arange(days)
    return rets[idx].cumsum(1)          # (n, days) cumulative log-return


def max_drawdown(paths: torch.Tensor) -> torch.Tensor:
    runmax = torch.maximum(
        paths.cummax(1).values,
        torch.zeros(1),                  # paths start at 0
    )
    dd = paths - runmax
    return dd.min(1).values              # <= 0


def scenario_report(ticker: str) -> dict:
    r = fetch_returns(ticker, years=YEARS)
    rets = r["returns"].float()
    span = f"{r['history'][0][:10]} .. {r['history'][-1][:10]}" if r.get("history") else "cached series"

    paths = resample_paths(rets, PATHS, DAYS, SEED)
    totals = paths[:, -1] * 100          # 64-day total log-return, %
    mdd = max_drawdown(paths) * 100
    worst_day = rets.abs().max() * 100

    q = torch.quantile(totals, torch.tensor([0.05, 0.5, 0.95]))
    dq = torch.quantile(mdd, torch.tensor([0.5, 0.95]))
    p_bigdd = (mdd < -10).float().mean() * 100
    p_bigloss = (totals < -5).float().mean() * 100

    print(f"\n{ticker}  ({YEARS}y of data: {span})  —  {PATHS:,} resampled {DAYS}-day scenarios")
    print(f"  {DAYS}-day return:   p5 {q[0]:+7.1f}%   median {q[1]:+7.1f}%   p95 {q[2]:+7.1f}%")
    print(f"  max drawdown:     median {dq[0]:7.1f}%   p95 {dq[1]:7.1f}% (p95 = milder 5-in-20 case)")
    print(f"  P(maxDD > 10%) {p_bigdd:5.1f}%        P(64d ret < -5%) {p_bigloss:5.1f}%")
    print(f"  observed worst single day: ±{worst_day:.1f}%  (resampling cannot exceed this)")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(paths[:250].T.numpy(), color="#8a93b5", lw=0.5, alpha=0.16)
    for qlo, qhi, lab in ((0.05, 0.95, "p5-p95"), (0.5, 0.5, "median")):
        band = torch.quantile(paths, torch.tensor([qlo, qhi]), dim=0).numpy() * 100
        if qlo == qhi:
            ax.plot(band[0], color="#3ddad7", lw=1.8, label="median")
        else:
            ax.fill_between(range(DAYS), band[0], band[1],
                            color="#5f7cff", alpha=0.13, label=lab)
    ax.axhline(0, color="gray", lw=0.7, ls=":")
    ax.set_title(f"{ticker} — {PATHS:,} resampled {DAYS}-day scenarios "
                 f"({YEARS}y history, seed {SEED})", fontsize=11)
    ax.set_xlabel("trading day"); ax.set_ylabel("cumulative log-return (%)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    stats = (f"p5 {q[0]:+.1f}% | median {q[1]:+.1f}% | p95 {q[2]:+.1f}%   "
             f"maxDD median {dq[0]:.1f}% / mild-5-in-20 {dq[1]:.1f}%   "
             f"P(DD>10%) {p_bigdd:.1f}%")
    ax.annotate(stats, (0.01, 0.02), xycoords="axes fraction",
                fontsize=8, color="#555")
    ax.annotate("honesty: your tail IS the sample — no day beyond the observed extreme is possible",
                (0.01, -0.14), xycoords="axes fraction", fontsize=7.5, color="#888")

    OUT.mkdir(exist_ok=True)
    fp = OUT / f"{ticker}_scenario_{date.today():%Y%m%d}.png"
    fig.savefig(fp, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {fp}")
    return dict(fp=fp)


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0
    tickers = prompt_tickers()
    print(f"scenario generator — {len(tickers)} ticker(s) | {YEARS}y | "
          f"{PATHS:,} paths x {DAYS}d | seed {SEED}")
    files = [scenario_report(t) for t in tickers]
    print("\nopening the folder...")
    try:
        subprocess.Popen(["explorer", str(files[0]["fp"].parent)])
    except Exception:
        pass
    if not QUIET:
        print("\nagain?  uv run python scenario_cli.py SPY AAPL --years 10 --paths 5000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())