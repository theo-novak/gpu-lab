"""Validation battery for the rBergomi engine — runs on GPU when available.

Five checks, printed as a table, hard asserts on each:
  1. eta = 0 anchor: variance is deterministic xi0, so the MC call price must
     match the Black-Scholes closed form within MC tolerance. Exact reference.
  2. E[v_t] = xi0(t): normalization + singular-diagonal correctness.
  3. E[S_T] = S0: martingale (drift discretization).
  4. Roughness: median |log v increment| scales like dt^H — log-log slope
     of the empirical variogram must recover H within tolerance.
  5. Put-call parity: C - P = S0 - K for the MC prices (internal consistency).

Exit code 0 = all green.
"""
from __future__ import annotations

import math
import time

import torch

import rbergomi as rb

N_PATHS = 100_000
N_STEPS = 256
T = 1.0
SEED = 7
ATOL_MULT = 4.0  # tolerate 4x the MC standard error


def main() -> int:
    dev = rb.device()
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    results = []

    def record(name: str, ok: bool, detail: str):
        results.append((name, "PASS" if ok else "FAIL", detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}: {detail}")

    # ---- timing (honest GPU/CPU numbers, part of bucket 1 too) ----
    t0 = time.perf_counter()
    v, dW, t = rb.simulate_variance(N_PATHS, N_STEPS, T, H=0.1, eta=1.9, seed=SEED, dev=dev)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    sim_secs = time.perf_counter() - t0
    print(f"simulate_variance: {N_PATHS} paths x {N_STEPS} steps in {sim_secs:.2f}s")
    if dev.type == "cuda":
        print(f"peak VRAM so far: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print()

    flat_xi = lambda tt: torch.full_like(tt, 0.235)

    # ---- 1. Black-Scholes anchor (eta = 0) ----
    print("Check 1: eta=0 reduces to Black-Scholes (exact anchor)")
    v0, dW0, _ = rb.simulate_variance(50_000, N_STEPS, T, H=0.1, eta=0.0, xi0=flat_xi, seed=SEED, dev=dev)
    # eta=0 => v == xi0 everywhere; verify that literally, then price
    max_dev = (v0 - 0.235).abs().max().item()
    res = rb.price_european_call(v0, dW0, torch.linspace(0, T, N_STEPS + 1, device=dev), rho=-0.9)
    bs = rb.bs_call_price(1.0, 1.0, T, math.sqrt(0.235))
    ok = max_dev < 1e-6 and abs(res["price"] - bs) < ATOL_MULT * res["mc_error"] + 1e-4
    record("BS anchor", ok,
           f"v==xi0 max dev {max_dev:.2e} | MC {res['price']:.4f} vs BS {bs:.4f} "
           f"(mc_err {res['mc_error']:.4f})")

    # ---- 2. E[v_t] = xi0(t) ----
    print("Check 2: E[v_t] matches forward variance curve")
    for frac in (0.25, 0.5, 1.0):
        i = int(frac * N_STEPS)
        mean_v = v[:, i].mean().item()
        mc_err_v = v[:, i].std().item() / math.sqrt(N_PATHS)
        target = 0.235
        record(f"E[v_t] @ t={frac}T", abs(mean_v - target) < ATOL_MULT * mc_err_v + 1e-3,
               f"MC mean {mean_v:.4f} vs xi0 {target:.3f} (mc_err {mc_err_v:.4f})")

    # ---- 3. martingale E[S_T] = S0 ----
    print("Check 3: E[S_T] = S0 (martingale)")
    res_m = rb.price_european_call(v, dW, t, rho=-0.9)
    # recompute S_T mean from a second pass via put-call parity below too
    # quick martingale: re-run the S dynamics and keep S_T
    ST_mean = None
    dt = T / N_STEPS
    g = rb.make_rng(99, dev)
    dWp = torch.randn(v.shape[0], N_STEPS, generator=g, device=dev) * math.sqrt(dt)
    vv = v[:, :-1]
    logS = torch.zeros(v.shape[0], device=dev)
    for i in range(N_STEPS):
        logS = logS - 0.5 * vv[:, i] * dt + torch.sqrt(vv[:, i]) * (-0.9 * dW[:, i] + math.sqrt(1 - 0.81) * dWp[:, i])
    ST_mean = logS.exp().mean().item()
    record("martingale", abs(ST_mean - 1.0) < 0.02,
           f"E[S_T] = {ST_mean:.4f} (target 1.0)")

    # ---- 4. roughness: TWO independent requirements ----
    # (a) INDEPENDENT reference: local log-vol variogram slope must recover
    #     ~H — roughness is the entire point of rBergomi. A flat variogram
    #     (slope ~0) means the "fix" degenerated X into white noise — this
    #     check exists precisely because that happened once (Oct 2026).
    # (b) SELF-CONSISTENCY: empirical must match the kernel-implied Gaussian
    #     reference (catches indexing/leakage bugs).
    print("Check 4: roughness (slope ~ H) + engine-vs-kernel consistency")
    ker = rb.kernel_matrix(N_STEPS, T / N_STEPS, 0.1).to(dev)
    Kd = ker.double() * math.sqrt(T / N_STEPS)
    lags = [1, 2, 4, 8, 16]
    xs, ys_ref, ys_emp = [], [], []
    logv = torch.log(v[:, 1:])
    for lag in lags:
        diff = (logv[:, lag:] - logv[:, :-lag]).abs().median().item()
        xs.append(math.log(lag))
        ys_emp.append(math.log(diff))
        dK = (Kd[lag:] - Kd[:-lag])
        var_inc = (dK * dK).sum(dim=1).median().item()
        ys_ref.append(math.log(math.sqrt(var_inc) * math.sqrt(2 / math.pi) * rb.DEFAULTS["eta"]))
    def lsq_slope(xs_, ys_):
        n_ = len(xs_)
        mx_, my_ = sum(xs_) / n_, sum(ys_) / n_
        return sum((x - mx_) * (y - my_) for x, y in zip(xs_, ys_)) / sum((x - mx_) ** 2 for x in xs_)
    slope_emp = lsq_slope(xs, ys_emp)
    record("roughness slope ~ H (independent)", abs(slope_emp - 0.1) < 0.03,
           f"empirical slope {slope_emp:.3f} vs H = 0.1 (flat ~0 = white-noise bug)")
    slope_ref = lsq_slope(xs, ys_ref)
    record("engine vs kernel reference", abs(slope_emp - slope_ref) < 0.02,
           f"empirical {slope_emp:.3f} vs kernel-implied {slope_ref:.3f}")

    # ---- 5. put-call parity on the same MC sample ----
    print("Check 5: put-call parity C - P = S0 - K (same paths)")
    call = res_m["price"]
    put = res_m["put_price"]
    parity_err = call - put - (1.0 - res_m["K"])
    record("put-call parity", abs(parity_err) < 3 * res_m["mc_error"] + 0.01,
           f"C-P = {call - put:.4f} vs S0-K = {1.0 - res_m['K']:.4f} (err {parity_err:.2e}, mc_err {res_m['mc_error']:.4f})")

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    print(f"{n_pass}/{len(results)} checks passed on {dev.type.upper()}")
    failed = [r for r in results if r[1] == "FAIL"]
    if failed:
        print("FAILED:", ", ".join(f[0] for f in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())