"""price_cli.py — the live option pricer: spot from the web, three prices out.

The ship surface for gpu-lab: fetch live spot + 2y history for a ticker
(Yahoo chart API, no key), calibrate vol from realized daily returns, then
price a European call three ways side-by-side:

  1. BS            — the naive baseline (closed form; the honest benchmark)
  2. rBergomi      — the validated exact engine at the SAME calibrated vol
                     (xi0 = sigma^2, teacher H/eta/rho) — the model price
  3. WGAN-GP       — the generator's price, vol-retargeted (gate 2)

MC errors and every input are stated. The GAN's economic-test status from
its latest training run is printed next to its price — no silent asterisks.

Usage:
  uv run python price_cli.py SPY --days 21
  uv run python price_cli.py AAPL --strike 260 --days 42 --paths 100000
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import typer

import realdata as rd
import rbergomi as rb
import wganlib as wl

app = typer.Typer(add_completion=False, help="Live European call pricer: three ways.")

HERE = Path(__file__).resolve().parent
CKPT = HERE / "models" / "wgan_notebook.pt"
N_STEPS = 64          # GAN paths: 64 trading days max horizon
TRADING_DAYS = 252


@app.command()
def price(
    symbol: str = typer.Argument(..., help="Ticker, e.g. SPY, AAPL"),
    strike: float = typer.Option(None, "--strike", "-k",
                                 help="Strike; default = ATM at live spot"),
    days: int = typer.Option(21, "--days", "-d",
                             help="Calendar days? No — TRADING days to expiry (1..63)"),
    paths: int = typer.Option(200_000, "--paths", "-n", help="MC paths for all three engines"),
    vol_years: int = typer.Option(2, "--vol-years", help="Years of daily returns for realized vol"),
):
    symbol = symbol.upper()
    if not 1 <= days <= N_STEPS:
        typer.echo(f"--days must be 1..{N_STEPS} (GAN paths are {N_STEPS} steps); got {days}")
        raise typer.Exit(1)

    # ---- live inputs ----
    quote = rd.fetch_spot(symbol)
    hist = rd.fetch_returns(symbol, years=vol_years)
    spot = quote["spot"]
    sigma = hist["returns"].std().item() * math.sqrt(TRADING_DAYS)
    K = strike if strike else round(spot, 2)
    S0 = 1.0                      # work in normalized units; moneyness is what matters
    k_norm = K / spot             # strike in units of spot

    T = days / TRADING_DAYS
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. Black-Scholes baseline ----
    bs = rb.bs_call_price(S0, k_norm, T, sigma)

    # ---- 2. exact rBergomi engine at calibrated vol ----
    v, dW, t = rb.simulate_variance(
        paths, days, T, H=rb.DEFAULTS["H"], eta=rb.DEFAULTS["eta"],
        xi0=lambda tt: torch.full_like(tt, sigma * sigma), seed=777, dev=dev,
    )
    eng = rb.price_european_call(v, dW, t, rho=rb.DEFAULTS["rho"], K=k_norm, dev=dev)

    # ---- 3. WGAN-GP generator, vol-retargeted (gate 2) ----
    gan_price = gan_err = float("nan")
    gan_status = "checkpoint missing"
    verdict_path = HERE / "models" / "gan_verdict.json"
    if CKPT.exists():
        try:
            tr = wl.Trainer.from_checkpoint(CKPT)
            gen = tr.generate_paths(paths, seed=777, retarget_vol=sigma)[:, :days]
            # normalized units like engine/BS: S0=1, strike k_norm; display ×spot
            logST = gen.sum(1)
            pay = torch.clamp(torch.exp(logST) - k_norm, min=0.0)
            gan_price = pay.mean().item()
            gan_err = pay.std().item() / math.sqrt(paths)
            gan_status = "economic test not yet measured for this checkpoint"
            if verdict_path.exists():
                verdict = json.loads(verdict_path.read_text())
                same = (
                    verdict.get("hidden_g") == tr.config["hidden_g"]
                    and verdict.get("hidden_d") == tr.config["hidden_d"]
                    and verdict.get("center") == tr.config.get("center")
                    and verdict.get("iter") == tr.iter
                )
                if same:
                    gan_status = (
                        f"economic test {verdict['status']} "
                        f"(|z|={abs(verdict['z']):.1f}, GAN {verdict['gan_price']:.4f} "
                        f"vs engine {verdict['engine_price']:.4f} on synthetic truth, {verdict['measured']})"
                    )
                else:
                    gan_status = "verdict file describes a different checkpoint — re-measure"
        except Exception as e:  # noqa: BLE001 — the CLI degrades honestly, not silently
            gan_status = f"unavailable: {type(e).__name__}: {e}"

    # ---- report ----
    moneyness = k_norm - 1.0
    print(f"\n=== {symbol} European call | {days} trading days (T = {T:.4f}y) ===")
    print(f"spot        : {spot:,.2f} {quote['currency']} (prev close {quote['prev_close']:,.2f})")
    print(f"strike      : {K:,.2f}  ({moneyness:+.2%} moneyness)")
    print(f"realized vol: {sigma:.2%}  ({vol_years}y daily, n={len(hist['returns'])})")
    print(f"MC paths    : {paths:,} (same for engine and GAN)")
    print(f"\n  Black-Scholes (baseline): {bs * spot:,.4f}")
    print(f"  rBergomi engine         : {eng['price'] * spot:,.4f}  (MC err ±{eng['mc_error'] * spot:,.4f})")
    if gan_status.startswith("unavailable") or gan_status == "checkpoint missing":
        print(f"  WGAN-GP generator       : {gan_status}")
    else:
        print(f"  WGAN-GP generator       : {gan_price * spot:,.4f}  (MC err ±{gan_err * spot:,.4f})")
        print(f"    [GAN status: {gan_status}]")
    print(f"\n  engine vs BS  : {(eng['price'] - bs) * spot:+,.4f}  (rough-vol smile vs flat-vol baseline)")
    print(f"  GAN vs engine : {(gan_price - eng['price']) * spot:+,.4f}"
          if not math.isnan(gan_price) else "  GAN vs engine : n/a")
    print("\nPrices are per share; multiply by contract size for premium. Research")
    print("tool — validated on synthetic ground truth, not a trading system.")


if __name__ == "__main__":
    app()