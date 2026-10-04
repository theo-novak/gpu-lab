"""realdata.py — real equity daily returns from Yahoo Finance's chart API.

No API key (the project rule: no secrets anywhere). Range: 2y daily.
Cached under data/real/ so notebook reruns don't re-fetch; delete the cache
file to force a refresh.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "real"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def fetch_returns(symbol: str, years: int = 2, refresh: bool = False) -> dict:
    """Daily log returns for `symbol` (e.g. 'SPY', 'AAPL').

    Returns dict(returns: Tensor[n], close: Tensor[n], dates: list[str],
    symbol). Dates are aligned to the close they belong to; the first close
    has no return, so returns[i] is the log-move INTO dates[i].
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"{symbol}_{years}y.pt"
    if cache_path.exists() and not refresh:
        return torch.load(cache_path, weights_only=False)

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?range={years}y&interval=1d"
    )
    req = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    close = result["indicators"]["quote"][0]["close"]
    pairs = [(t, c) for t, c in zip(ts, close) if c is not None]
    if len(pairs) < 130:
        raise RuntimeError(f"{symbol}: only {len(pairs)} valid closes — aborting")

    dates_all = [time.strftime("%Y-%m-%d", time.gmtime(t)) for t, _ in pairs]
    prices = torch.tensor([c for _, c in pairs], dtype=torch.float64)

    log_ret = torch.diff(torch.log(prices))
    keep = torch.isfinite(log_ret)
    out = dict(
        returns=log_ret[keep].float(),
        close=prices[1:][keep].float(),
        dates=[d for d, k in zip(dates_all[1:], keep.tolist()) if k],
        symbol=symbol,
    )
    torch.save(out, cache_path)
    return out


def make_panel(returns: torch.Tensor, window: int = 64, stride: int = 1) -> torch.Tensor:
    """Slice a 1D return series into (n_windows, window) panels — the SAME
    shape as the GAN's output, so the identical estimators apply.

    Overlapping windows (stride 1): each real day appears in up to `window`
    panels. Fine for ACF-style structure comparison (the estimator is a
    cross-window average of per-lag statistics); noted in the notebook.
    """
    if len(returns) < window:
        raise ValueError(f"series too short: {len(returns)} < window {window}")
    return returns.unfold(0, window, stride)