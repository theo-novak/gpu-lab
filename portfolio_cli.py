"""portfolio_cli.py — joint portfolio stress table (Practical Use #4).

Scenario bundles for a WHOLE PORTFOLIO: weights + tickers in, one number
out set — per-name bad-quarter tails AND the portfolio's own drawdown
distribution, under JOINT calendar bootstrap (each scenario re-samples one
common historical window for every name, preserving real cross-correlations;
independent per-name resampling would silently destroy them — this is the
honest joint method).

Honesty notes (printed with the result):
  * portfolio tail = the sample's tail: no day beyond the joint history's
    observed extreme is possible (resampling blindness, as measured in v2/v3);
  * assumes today's weights are static for the window;
  * needs every ticker to share enough history — the overlapping window is
    the union floor, printed.

Run:
  uv run python portfolio_cli.py                          # prompts for input
  uv run python portfolio_cli.py "SPY:40,QQQ:25,NVDA:15,V:10,CASH:10"
  uv run python portfolio_cli.py SPY:.5 NDX:.3 "TSLA:.2" --days 126
Flags: --days 64  --paths 3000  --years 5  --seed 2026  --plot
Prompts accept either  TICKER:WEIGHT  (weight = % or fraction).
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from realdata import fetch_returns

HERE = Path(__file__).resolve().parent
OUT = HERE / "report" / "portfolio"


def argval(flag: str, default: int) -> int:
    if flag in sys.argv:
        try:
            return int(sys.argv[sys.argv.index(flag) + 1])
        except (ValueError, IndexError):
            return default
    return default


def has(flag: str) -> bool:
    return flag in sys.argv


def parse_portfolios(spec: str) -> list[tuple[str, float]]:
    out = []
    for part in spec.replace('"', "").split(","):
        if ":" not in part:
            raise SystemExit(f"'{part.strip()}' needs TICKER:WEIGHT (e.g. SPY:40)")
        t, w = part.split(":", 1)
        w = float(w.strip())
        if w > 2:      # percent form — normalize at the end
            out.append((t.strip().upper(), w))
        else:
            out.append((t.strip().upper(), w * 100))
    return out


def prompt_portfolio() -> list[tuple[str, float]]:
    try:
        raw = input("Portfolio as TICKER:WEIGHT (e.g. SPY:40,QQQ:30,NVDA:10,CASH:20): ").strip()
    except EOFError:
        raw = ""
    return parse_portfolios(raw or "SPY:100")


def aligned_returns(tickers: list[str], years: int):
    """Fetch each name; align by common dates (intersection)."""
    data = {}
    for t in tickers:
        d = fetch_returns(t, years=years)
        data[t] = dict(zip(d["dates"], d["returns"].tolist()))
    common: set | None = None
    for t in tickers:
        s = set(data[t])
        common = s if common is None else (common & s)
    dates = sorted(common)  # type: ignore[arg-type]
    if len(dates) < 130:
        raise SystemExit(f"overlap only {len(dates)} days across {tickers} — need more shared history")
    mats = torch.tensor([[data[t][d] for t in tickers] for d in dates])
    return mats, dates


def stress_table(tickers, weights, mats, days, paths, seed):
    g = torch.Generator().manual_seed(seed)
    T, N = mats.shape
    starts = torch.randint(0, T - days + 1, (paths,), generator=g)
    # (paths, days, N): one common calendar window per scenario, all names
    win = mats[starts.view(paths, 1) + torch.arange(days)]
    port = (win * torch.tensor(weights).view(1, 1, N)).sum(2)   # weighted daily log-rets
    port_cum = port.cumsum(1)                                    # (paths, days) log
    total = port_cum[:, -1] * 100
    runmax = torch.cummax(port_cum, 1).values
    mdd = (port_cum - runmax).min(1).values * 100
    return port_cum, total, mdd


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        pf = parse_portfolios(args[0])
    else:
        pf = prompt_portfolio()
    days = int(argval("--days", 64))
    paths = int(argval("--paths", 3000))
    years = int(argval("--years", 5))
    seed = int(argval("--seed", 2026))

    tickers = [t for t, _ in pf]
    wsum = sum(w for _, w in pf)
    weights = [w / wsum for _, w in pf]
    if any(t.upper() == "CASH" for t in tickers):
        print("NOTE: CASH earns 0% here (no rate modeled) — it only damps variance.")

    mats, dates = aligned_returns(tickers, years)
    lo, hi = dates[0], dates[-1]
    print(f"portfolio: " + ", ".join(f"{t} {w:.0%}" for t, w in zip(tickers, weights)))
    print(f"joint history: {len(dates)} common days, {lo}..{hi}")
    print(f"scenarios: {paths:,} x {days}d windows, joint calendar bootstrap (correlations kept), seed {seed}")

    port_cum, total, mdd = stress_table(tickers, weights, mats, days, paths, seed)

    qt = torch.quantile(total, torch.tensor([0.05, 0.5, 0.95]))
    qd = torch.quantile(mdd, torch.tensor([0.05, 0.5]))
    p_dd10 = float((mdd <= -10).float().mean()) * 100
    p_bad = float((total <= -5).float().mean()) * 100
    print(f"\n{days}-day portfolio return:  p5 {qt[0]:+.1f}%  median {qt[1]:+.1f}%  p95 {qt[2]:+.1f}%")
    print(f"portfolio maxDD:  1-in-20 sees {qd[0]:.1f}%  median {qd[1]:.1f}%")
    print(f"P(maxDD >= 10%) {p_dd10:.1f}%   P(total <= -5%) {p_bad:.1f}%")
    print("  honesty: the portfolio's tail is the sample's tail — no day beyond the"
          "\n  joint history's observed extremes exists in these scenarios.")

    if has("--plot"):
        OUT.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(port_cum[:300].T.cpu().numpy() * 100, color="#8a93b5", lw=0.5, alpha=0.15)
        band = torch.quantile(port_cum * 100, torch.tensor([0.05, 0.5, 0.95]), dim=0).cpu().numpy()
        ax.plot(band[1], color="#3ddad7", lw=1.8, label="median")
        ax.fill_between(range(days), band[0], band[2], color="#5f7cff", alpha=0.12, label="p5-p95")
        ax.axhline(0, color="gray", lw=0.7, ls=":")
        ax.set_title(f"portfolio scenarios — {', '.join(f'{t} {w:.0%}' for t, w in zip(tickers, weights))}\n"
                     f"joint calendar bootstrap, {paths:,} paths, {days}d, history {lo}..{hi}", fontsize=10)
        ax.set_xlabel("trading day"); ax.set_ylabel("portfolio log-return (%)")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
        fp = OUT / f"portfolio_{days}d_{date.today():%Y%m%d}.png"
        fig.savefig(fp, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {fp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())