"""rBergomi (rough Bergomi) simulation on GPU — the exact/reference engine.

Model (Bayer–Friz–Gatheral 2016 convention, normalization derived to be
self-consistent; see validate_rbergomi.py for the checks that pin it):

    v_t = xi0(t) * exp( eta * X_t - 0.5 * eta^2 * t^(2H) )
    dS_t = sqrt(v_t) * S_t dZ_t,   dZ = rho dW + sqrt(1-rho^2) dW_perp

where X is the Riemann–Liouville Volterra process

    X_t = sqrt(2H) * Integral_0^t (t-s)^(H-1/2) dW_s,   Var[X_t] = t^(2H)

so E[v_t] = xi0(t) exactly (log-normal, mean one by construction).

Discretization: grid kernel convolution (hybrid-scheme style, H in (0, 0.5)):
    X_i = sqrt(2H) * sum_{j<i} ((i-j) dt)^(H-1/2) dW_j   (regular part)
          + dt^(H-1/2) * dW_i                            (moment-matched
                                                          singular diagonal:
        Var[ Integral_{t_{i-1}}^{t_i} (t_i-u)^(H-1/2) dW_u ] = dt^(2H)/(2H)
        => coefficient c with c^2 dt = dt^(2H)/(2H) times outer sqrt(2H)
        gives  dt^(H-1/2). )

One dense matmul per chunk — exactly what GPUs are for.
"""
from __future__ import annotations

import math

import torch

# rBergomi base parameters (BFG 2016 ballpark): H=0.1, eta=1.9, rho=-0.9.
DEFAULTS = dict(H=0.1, eta=1.9, rho=-0.9)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_rng(seed: int, dev: torch.device) -> torch.Generator:
    g = torch.Generator(device=dev.type)
    g.manual_seed(seed)
    return g


def kernel_matrix(N: int, dt: float, H: float, dtype=torch.float32, normalize: bool = True) -> torch.Tensor:
    """Lower-triangular convolution kernel K[i, j] for the Volterra process.

    Convention: X_i = sum_j K[i, j] * dW_j  with dW_j ~ N(0, dt), so
    Var[X_i] = dt * sum_j K[i, j]^2. Row i corresponds to time t_{i+1}.

    Weights (hybrid-scheme style, H in (0, 0.5)):
      regular cells j < i: EXACT integral of the kernel over the cell,
        sqrt(2H) * Integral_{t_j}^{t_{j+1}} (t_{i+1} - u)^(H-1/2) du
      singular cell j = i: moment-matched, w_ii = dt^(H-1/2), since
        Var[ sqrt(2H) Integral_{t_i}^{t_{i+1}} (t_{i+1}-u)^(H-1/2) dW_u ] = dt^(2H)

    Diagnosis (probe, Oct 2026): right-edge POINT evaluation (the original
    implementation) overstates Var[X] by ~+3.7..4.7% at these settings
    (kernel is largest at the right edge, H-1/2 < 0), inflating E[v_t] by
    ~+6.9% via the lognormal mean. Exact-interval integration reduces the
    wedge to ~-0.05%; row normalization below removes it entirely:
        scale row i by sqrt( t_{i+1}^{2H} / (dt * sum_j K[i,j]^2) )
    so Var[X_{t}] = t^{2H} holds EXACTLY on the grid while preserving the
    within-row (conditional) correlation structure.
    """
    if not 0.0 < H < 0.5:
        raise ValueError(f"hybrid discretization targets H in (0, 0.5), got {H}")
    beta = H + 0.5
    idx = torch.arange(N, dtype=torch.float64)
    dist_right = (idx.view(-1, 1) - idx.view(1, -1)).clamp(min=0)   # i - j
    dist_left = dist_right + 1.0                                   # i + 1 - j
    reg = torch.where(
        dist_right > 0,
        # BLP b=1 projection weight: E[∫_cell (t-u)^{H-1/2} dW_u | ΔW_j] / dt
        # = sqrt(2H) * ∫_cell (t-u)^{H-1/2} du / dt   (AVERAGE kernel value —
        # the /dt matters: dropping it scales regular cells by dt and,
        # after row normalization, degenerates X to near-independent noise.)
        (dist_left.pow(beta) - dist_right.pow(beta)) / beta
            * dt ** (beta - 1.0) * math.sqrt(2.0 * H),
        torch.zeros((), dtype=torch.float64),
    )
    ker = reg
    ker[torch.eye(N, dtype=torch.bool)] = dt ** (H - 0.5)           # singular diagonal ONLY
    if normalize:
        target = ((idx + 1.0) * dt).pow(2.0 * H)                   # t_{i+1}^{2H}
        current = dt * (ker * ker).sum(dim=1)
        ker = ker * (target / current).sqrt().view(-1, 1)
    return ker.to(dtype)


def simulate_variance(
    n_paths: int,
    n_steps: int,
    T: float,
    H: float,
    eta: float,
    xi0=None,
    seed: int = 42,
    dev: torch.device | None = None,
    chunk: int = 2**15,
    dtype=torch.float32,
):
    """Simulate (v, dW) on the time grid. Returns (v, dW, t).

    v: (n_paths, n_steps+1) variance path incl. v_0 = xi0(0)
    dW: (n_paths, n_steps) the Brownian increments driving X
    t: (n_steps+1,) time grid
    """
    dev = dev or device()
    # flat forward-variance curve at 0.235 (vol ~48%); the canonical BFG test
    # case uses 0.235^2 (vol 23.5%) — swap in `0.235**2` to match papers.
    xi0 = xi0 or (lambda t: torch.full_like(t, 0.235))
    dt = T / n_steps
    t = torch.linspace(0.0, T, n_steps + 1, dtype=dtype, device=dev)
    ker = kernel_matrix(n_steps, dt, H, dtype).to(dev)

    v = torch.empty(n_paths, n_steps + 1, dtype=dtype, device=dev)
    dW_all = torch.empty(n_paths, n_steps, dtype=dtype, device=dev)
    for start in range(0, n_paths, chunk):
        end = min(start + chunk, n_paths)
        g = make_rng(seed + start // chunk, dev)
        dW = torch.randn(end - start, n_steps, generator=g, device=dev, dtype=dtype) * math.sqrt(dt)
        # Volterra process: one matmul per chunk
        X = dW @ ker.T
        # v on interior grid points 1..N (v_0 handled below)
        ti = t[1:]
        logv = eta * X - 0.5 * eta * eta * ti.pow(2.0 * H)
        v[start:end, 1:] = xi0(ti) * torch.exp(logv)
        dW_all[start:end] = dW
    v[:, 0] = xi0(t[:1]).expand(n_paths)
    return v, dW_all, t


def price_european_call(
    v,
    dW,
    t,
    rho: float,
    S0: float = 1.0,
    K: float | None = None,
    chunk: int = 2**15,
    dev: torch.device | None = None,
):
    """Log-Euler price of a European call. Returns dict with price and S_T.

    Uses the SAME dW that drove the variance (correlation rho) plus an
    independent dW_perp. v[:, i] is the variance on [t_i, t_{i+1}).
    """
    dev = dev or device()
    n_paths, n_steps = dW.shape
    dt = (t[-1] / n_steps).item()
    K = S0 if K is None else K  # ATM by default
    orthogonal_scale = math.sqrt(1.0 - rho * rho)

    payoffs = torch.empty(n_paths, dtype=v.dtype, device=dev)
    put_payoffs = torch.empty(n_paths, dtype=v.dtype, device=dev)
    for start in range(0, n_paths, chunk):
        end = min(start + chunk, n_paths)
        g = make_rng(10_000 + start // chunk, dev)
        dWp = torch.randn(end - start, n_steps, generator=g, device=dev, dtype=v.dtype) * math.sqrt(dt)
        vv = v[start:end, :-1]  # variance on each interval
        logS = torch.zeros(end - start, dtype=v.dtype, device=dev)
        for i in range(n_steps):
            noise = rho * dW[start:end, i] + orthogonal_scale * dWp[:, i]
            logS = logS - 0.5 * vv[:, i] * dt + torch.sqrt(vv[:, i]) * noise
        ST = S0 * torch.exp(logS)
        payoffs[start:end] = torch.clamp(ST - K, min=0.0)
        put_payoffs[start:end] = torch.clamp(K - ST, min=0.0)

    price = payoffs.mean().item()
    std = payoffs.std().item()
    mc_error = std / math.sqrt(n_paths)
    return {
        "price": price,
        "put_price": put_payoffs.mean().item(),
        "K": K,
        "mc_error": mc_error,
        "ci95": (price - 1.96 * mc_error, price + 1.96 * mc_error),
    }


def log_return_increments_dt(
    v: torch.Tensor,
    dW: torch.Tensor,
    rho: float,
    dt: float,
    T: float | None = None,
    seed: int = 12_345,
    chunk: int = 2**15,
) -> torch.Tensor:
    """See log_return_increments — with dt supplied explicitly (T optional)."""
    dev = v.device
    n_paths, n_steps = dW.shape
    orthogonal_scale = math.sqrt(1.0 - rho * rho)
    r = torch.empty(n_paths, n_steps, dtype=v.dtype, device=dev)
    for start in range(0, n_paths, chunk):
        end = min(start + chunk, n_paths)
        g = make_rng(seed + start // chunk, dev)
        dWp = torch.randn(end - start, n_steps, generator=g, device=dev, dtype=v.dtype) * math.sqrt(dt)
        vv = v[start:end, :-1]                       # left-point variance per interval
        noise = rho * dW[start:end] + orthogonal_scale * dWp
        r[start:end] = -0.5 * vv * dt + torch.sqrt(vv) * noise
    return r

def bs_call_price(S0: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if sigma <= 0 or T <= 0:
        return max(S0 - K, 0.0)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    Phi = torch.distributions.Normal(0.0, 1.0).cdf
    return float(S0 * Phi(torch.tensor(d1)).item() - K * math.exp(-r * T) * Phi(torch.tensor(d2)).item())


def implied_vol(price: float, S0: float, K: float, T: float, r: float = 0.0) -> float:
    """Bisection — robust, boring, right."""
    lo, hi = 1e-6, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_call_price(S0, K, T, mid, r) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)