"""Bucket 3: build the WGAN training dataset from the validated rBergomi engine.

Output: data/rbergomi_returns.pt — 100k log-return paths (n_paths, 64)
  - 64 daily steps (T = 0.25y ≈ 64 trading days), float32, GPU-generated
  - stored with full metadata for reproducibility

Fingerprint checks on the saved dataset (exact references, computed from
the same saved tensors):
  1. E[r_i]  = -0.5 * E[v_{t_i}] * dt          (drift, per-step)
  2. Var[r_i] = E[v_{t_i}] * dt                 (variance, per-step)
  3. Var[sum_i r_i] = Var[log S_T] matches path-level total from the SAME
     saved paths (internal consistency)
  4. cross-correlation Corr(r_i, log v increments) < 0 (leverage effect
     present, sign correct: rho < 0)
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch

import rbergomi as rb

N_PATHS = 100_000
N_STEPS = 64          # daily returns over a quarter
T = 64.0 / 252.0      # 64 trading days
SEED = 2026
DT = T / N_STEPS

H, ETA, RHO = 0.1, 1.9, -0.9
XI0 = 0.235


def main() -> int:
    dev = rb.device()
    print(f"device: {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}")

    t0 = time.perf_counter()
    v, dW, t = rb.simulate_variance(
        N_PATHS, N_STEPS, T, H=H, eta=ETA,
        xi0=lambda tt: torch.full_like(tt, XI0), seed=SEED, dev=dev,
    )
    r = rb.log_return_increments_dt(v, dW, rho=RHO, dt=DT, seed=SEED + 1)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    gen_secs = time.perf_counter() - t0

    # ---- exact per-step references from the SAME variance paths ----
    v_left = v[:, :-1]                      # v_{t_i} driving interval i
    exp_r_ref = (-0.5 * v_left * DT).mean(dim=0)
    var_r_ref = (v_left * DT).mean(dim=0)

    exp_r = r.mean(dim=0)
    var_r = r.var(dim=0, unbiased=True)
    mc_err_mean = r.std(dim=0) / math.sqrt(N_PATHS)

    # tolerances: 4 sigma for the mean, 5% relative for the variance
    ok_mean = (exp_r - exp_r_ref).abs() < 4.0 * mc_err_mean + 1e-5
    ok_var = ((var_r - var_r_ref).abs() / var_r_ref) < 0.05
    print(f"\ngenerated {N_PATHS} x {N_STEPS} return paths in {gen_secs:.2f}s")
    print(f"per-step drift  E[r]:  {ok_mean.float().mean()*100:.0f}% of steps within 4σ")
    print(f"per-step var  Var[r]:   {ok_var.float().mean()*100:.0f}% of steps within 5%")

    # ---- leverage effect: correlation between r_i and log-v increment ----
    dlogv = torch.log(v[:, 1:]) - torch.log(v[:, :-1])
    r_c = r - r.mean(dim=0, keepdim=True)
    d_c = dlogv - dlogv.mean(dim=0, keepdim=True)
    corr = (r_c * d_c).sum(dim=0) / (
        r_c.pow(2).sum(dim=0).sqrt() * d_c.pow(2).sum(dim=0).sqrt() + 1e-12
    )
    mean_corr = corr.mean().item()
    print(f"leverage Corr(r_i, dlog v_i): mean {mean_corr:+.3f}  (expect negative)")

    # ---- terminal consistency: E[log S_T] vs EXACT reference ----
    # E[log S_T] = -0.5 * Integral E[v_t] dt = -0.5 * xi0 * T   (flat xi0)
    # (Oct 2026 lesson: the first version of this check hand-waved "E[log S_T]
    # ~ 0" with a 0.02 tolerance — tighter than the true value -0.0298 —
    # and failed a CORRECT engine. Reference must be exact, not vibes.)
    logST = r.sum(dim=1)
    exact_ref = -0.5 * XI0 * T
    z = (logST.mean().item() - exact_ref) / (logST.std().item() / math.sqrt(N_PATHS))
    print(f"E[log S_T]: {logST.mean().item():+.6f} vs exact {exact_ref:+.6f}  (z = {z:+.2f})")
    print(f"Var[log S_T] from saved paths: {logST.var().item():.4f}")

    checks = [
        ok_mean.float().mean().item() > 0.95,
        ok_var.float().mean().item() > 0.95,
        mean_corr < -0.15,                    # strongly negative at rho=-0.9
        abs(z) < 4.0,                         # within 4 sigma of the EXACT ref
    ]
    if not all(checks):
        print("FINGERPRINT FAILED:", checks)
        return 1

    out = Path(__file__).parent / "data" / "rbergomi_returns.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "returns": r.cpu(),
            "H": H, "eta": ETA, "rho": RHO, "xi0": XI0,
            "n_paths": N_PATHS, "n_steps": N_STEPS, "T": T, "dt": DT,
            "seed": SEED, "seed_dWp": SEED + 1,
        },
        out,
    )
    size_mb = out.stat().st_size / 2**20
    print(f"\nsaved {out}  ({size_mb:.1f} MiB)")
    print("all fingerprint checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())