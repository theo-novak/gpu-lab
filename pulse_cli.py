"""pulse_cli.py — the market-pulse daily log (Practical Use #2).

One row per run: date, spot, realized vol (N-day), ATM-21d BS price,
engine price, and their spread — appended to report/pulse/pulse.csv.
Run it daily (cron/task scheduler); the CSV becomes a growing personal
record of how the BS-vs-rough-vol disagreement moves through regimes.

Honesty note: the engine prices under ROUGH volatility (H=0.1, calibrated
per-name to realized vol over --vol-years). Its spread vs BS is a model-
disagreement signal, not an edge over the market — no IV surface is used
(option chains are auth-gated keyless), so both prices share the same
input vol by design.

Run:
  uv run python pulse_cli.py                # snapshot for SPY
  uv run python pulse_cli.py SPY QQQ        # multiple names, one row each
  uv run python pulse_cli.py --show         # pretty-print the whole log
Flags: --vol-years 2  --days 21  --n 4   (n rows of history for --show)
"""
from __future__ import annotations

import csv
import math
import sys
from datetime import date, datetime
from pathlib import Path

import torch

import rbergomi as rb
import realdata as rd

HERE = Path(__file__).resolve().parent
LOG = HERE / "report" / "pulse" / "pulse.csv"
FIELDS = ["date", "time", "symbol", "spot", "realized_vol", "vol_days",
          "days", "bs_price", "engine_price", "spread_pct", "engine_mc95"]


def argval(flag: str, default: int) -> int:
    if flag in sys.argv:
        raw = sys.argv[sys.argv.index(flag) + 1]
        try:
            return int(raw)
        except ValueError:
            return default
    return default


VOL_YEARS = argval("--vol-years", 2)
DAYS = argval("--days", 21)
SHOW_N = int(argval("--n", 8))


def realized_vol(rets: torch.Tensor, window: int) -> float:
    """Annualized realized vol from the last `window` daily log returns."""
    r = rets[-window:]
    return float(r.std(unbiased=True) * math.sqrt(252))


def snapshot(symbol: str, vol_years: int, days: int) -> dict:
    quote = rd.fetch_spot(symbol)
    hist = rd.fetch_returns(symbol, years=vol_years)
    rets = hist["returns"].float()
    sigma = realized_vol(rets, min(60, len(rets)))
    S0 = float(quote["spot"])
    T = days / 252.0

    # the validated engine, calibrated to THIS name's realized vol:
    # xi0 = sigma^2 / 0.235 * 0.235 ... i.e. scale the canonical flat curve
    # so annualized vol == sigma. rbergomi's flat xi0=0.235 is vol sqrt(.235).
    xi_val = sigma ** 2
    n_paths, n_steps = 2 ** 15, 64
    v, dW, t = rb.simulate_variance(n_paths, n_steps, T, H=0.1, eta=1.9,
                                    xi0=lambda tt: torch.full_like(tt, xi_val),
                                    seed=2026)
    r = rb.log_return_increments_dt(v, dW, rho=-0.9, dt=T / n_steps, seed=2027)
    ST = S0 * torch.exp(r.sum(1))
    K = S0
    payoff = torch.clamp(ST - K, min=0)
    price = float(payoff.mean().item())
    mc95 = 1.96 * float(payoff.std(unbiased=False).item()) / math.sqrt(n_paths)
    bs = rb.bs_call_price(S0, K, T, sigma)

    return dict(
        date=date.today().isoformat(),
        time=datetime.now().strftime("%H:%M"),
        symbol=symbol,
        spot=round(S0, 2),
        realized_vol=round(sigma, 4),
        vol_days=min(60, len(rets)),
        days=days,
        bs_price=round(bs, 3),
        engine_price=round(price, 3),
        spread_pct=round((price / bs - 1.0) * 100, 2) if bs > 0 else 0.0,
        engine_mc95=round(mc95, 3),
    )


def show(n: int) -> None:
    if not LOG.exists():
        print("no log yet — run a snapshot first")
        return
    rows = list(csv.DictReader(LOG.open()))
    print(f"market pulse log — {len(rows)} rows ({rows[0]['date']} .. {rows[-1]['date'] if rows else '-'})")
    print(f"{'date':11s} {'sym':5s} {'spot':>9s} {'rvol':>7s} {'BS':>8s} {'engine':>8s} {'spr%':>7s}")
    for row in rows[-n:]:
        print(f"{row['date']:11s} {row['symbol']:5s} {float(row['spot']):9.2f} "
              f"{float(row['realized_vol'])*100:6.1f}% {float(row['bs_price']):8.2f} "
              f"{float(row['engine_price']):8.2f} {float(row['spread_pct']):+6.2f}%")


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0
    if "--show" in sys.argv:
        show(SHOW_N)
        return 0
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tickers = args or ["SPY"]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG.exists()
    for tick in tickers:
        row = snapshot(tick, int(VOL_YEARS), int(DAYS))
        with LOG.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
                new_file = False
            w.writerow(row)
        print(f"{row['symbol']}: spot {row['spot']} | rvol {row['realized_vol']*100:.1f}% | "
              f"BS {row['bs_price']} vs engine {row['engine_price']} "
              f"({row['spread_pct']:+.2f}%, MC95 ±{row['engine_mc95']}) -> logged")
    print(f"log: {LOG}")


if __name__ == "__main__":
    raise SystemExit(main())