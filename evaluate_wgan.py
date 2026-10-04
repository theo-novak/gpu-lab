"""Bucket 5: honest report — generated vs real rBergomi return paths.

Loads models/wgan.pt + the dataset, generates 100k fake paths, and compares
on structure the generator was NEVER shown directly:
  1. per-step mean/std profiles            (marginal first two moments)
  2. terminal log-return E / Var           (aggregation across the path)
  3. ACF of |returns|, lags 1..15           (volatility clustering)
  4. roughness: log-log slope of ACF(|r|)  (rBergomi power-law, H=0.1)
  5. leverage proxy Corr(r_i, |r_{i+1}|)   (negative under rho < 0)

Outputs: report/*.png + printed honest table with pass/warn/fail.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from train_wgan import Generator, N_STEPS, NOISE_DIM

HERE = Path(__file__).parent
DATA = HERE / "data" / "rbergomi_returns.pt"
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "models" / "wgan.pt"
REPORT = HERE / "report"
N_GEN = 100_000
SEED = 4242


def acf(x: torch.Tensor, k: int) -> torch.Tensor:
    """Autocorrelation of a (n_paths, T) tensor at lag k, per column pair."""
    T = x.shape[1]
    if k == 0:
        return torch.ones(T - k, device=x.device)
    a, b = x[:, :-k] - x[:, :-k].mean(0, keepdim=True), x[:, k:] - x[:, k:].mean(0, keepdim=True)
    num = (a * b).mean(0)
    den = a.pow(2).mean(0).sqrt() * b.pow(2).mean(0).sqrt()
    return num / (den + 1e-12)


def corr_next_abs(r: torch.Tensor) -> float:
    """Leverage proxy: Corr(r_i, |r_{i+1}|), averaged over i."""
    a = r[:, :-1]
    b = (r[:, 1:]).abs()
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    corr = (a * b).sum(0) / (a.pow(2).sum(0).sqrt() * b.pow(2).sum(0).sqrt() + 1e-12)
    return corr.mean().item()


def lsq_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REPORT.mkdir(exist_ok=True)

    payload = torch.load(DATA)
    real = payload["returns"].to(dev)               # (100k, 64)
    ckpt = torch.load(MODEL, map_location=dev)
    scale = ckpt["scale"]

    G = Generator().to(dev)
    G.load_state_dict(ckpt["G"])
    G.eval()
    rng = torch.Generator(device=dev.type); rng.manual_seed(SEED)

    fakes = []
    with torch.no_grad():
        for start in range(0, N_GEN, 2**15):
            n = min(2**15, N_GEN - start)
            z = torch.randn(n, NOISE_DIM, generator=rng, device=dev)
            fakes.append(G(z))
    gen = torch.cat(fakes) * scale                   # back to return units

    rows = []

    def verdict(err, good, warn):
        return "PASS" if err <= good else ("WARN" if err <= warn else "FAIL")

    # ---- 1. per-step mean / std ----
    mr, mg = real.mean(0), gen.mean(0)
    sr, sg = real.std(0), gen.std(0)
    se_mean = real.std(0) / math.sqrt(real.shape[0])
    mean_z = ((mg - mr).abs() / (4 * se_mean)).max().item()        # 4-sigma units (harsh)
    mean_rel = ((mg - mr).abs() / sr).max().item()                  # relative to step vol (fair)
    std_err = ((sg - sr).abs() / sr).max().item()
    rows.append(("per-step mean (max |z|/4)", f"{mean_z:.2f}",
                 verdict(mean_z, 0.5, 1.0) + f"  [rel-to-vol {mean_rel:.1%} → {verdict(mean_rel, 0.15, 0.30)}]"))
    rows.append(("per-step std (max rel err)", f"{std_err:.1%}", verdict(std_err, 0.05, 0.15)))

    # ---- 2. terminal log-return ----
    tr, tg = real.sum(1), gen.sum(1)
    ev_err = abs(tg.var().item() - tr.var().item()) / tr.var().item()
    mean_T_err = abs(tg.mean().item() - tr.mean().item()) / (tr.std().item() / 10)
    rows.append(("Var[log S_T] (rel err)", f"{ev_err:.1%}", verdict(ev_err, 0.05, 0.15)))
    rows.append(("E[log S_T] (err/0.1σm)", f"{mean_T_err:.2f}", verdict(mean_T_err, 0.5, 1.0)))

    # ---- 3+4. ACF of |r| and roughness slope ----
    lags = list(range(1, 16))
    def acf_curve(x):
        """Mean ACF over columns, per lag — pad each lag's shrinking vector
        (T-k columns) to fixed T-1 before stacking (pad with its own mean)."""
        T = x.shape[1]
        curves = []
        for k in lags:
            v = acf(x, k)                     # (T-k,)
            m = v.mean()
            pad = torch.full((T - 1 - v.shape[0],), m, device=x.device)
            curves.append(torch.cat([v, pad]))
        return torch.stack(curves).mean(1)    # (len(lags),)
    acf_r = acf_curve(real.abs())
    acf_g = acf_curve(gen.abs())
    acf_err = (acf_g - acf_r).abs().mean().item()
    xs = [math.log(k) for k in lags]
    slope_r = lsq_slope(xs, [math.log(max(v, 1e-6)) for v in acf_r.tolist()])
    slope_g = lsq_slope(xs, [math.log(max(v, 1e-6)) for v in acf_g.tolist()])
    slope_err = abs(slope_g - slope_r)
    rows.append(("ACF|r| mean abs dev", f"{acf_err:.3f}", verdict(acf_err, 0.02, 0.05)))
    rows.append(("roughness slope (|Δ|)", f"{slope_err:.3f}", verdict(slope_err, 0.08, 0.15)))
    lev_r, lev_g = corr_next_abs(real), corr_next_abs(gen)
    lev_err = abs(lev_g - lev_r)
    rows.append(("leverage proxy (|Δ|)", f"{lev_err:.3f}", verdict(lev_err, 0.05, 0.12)))

    # ---- report ----
    print(f"\n{'metric':<28}{'value':>10}   verdict   (real vs generated)")
    print("-" * 78)
    for name, val, v in rows:
        print(f"{name:<28}{val:>10}   {v}")
    print("-" * 78)
    print(f"real roughness slope {slope_r:+.3f} | gen {slope_g:+.3f} "
          f"(theory ~ 2H-1 = {-0.8:+.1f} for ACF of |r|, H=0.1)")
    print(f"real leverage {lev_r:+.3f} | gen {lev_g:+.3f} (rho=-0.9)")
    print(f"Var[log S_T]: real {tr.var().item():.4f} | gen {tg.var().item():.4f}")

    # ---- plots ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("WGAN-GP vs rBergomi — honest report", fontsize=13)

    ax = axes[0, 0]
    for i in range(3):
        ax.plot(real[i].cumsum(0).cpu(), color="steelblue", alpha=0.7, lw=1)
    for i in range(3):
        ax.plot(gen[i].cumsum(0).cpu(), color="crimson", alpha=0.7, lw=1, ls="--")
    ax.set_title("cumulative log-price paths (blue=real, red=gen)")

    ax = axes[0, 1]
    ax.plot(sr.cpu(), color="steelblue", label="real")
    ax.plot(sg.cpu(), color="crimson", ls="--", label="generated")
    ax.set_title("per-step std of returns"); ax.legend()

    ax = axes[0, 2]
    bins = 40
    ax.hist(tr.cpu(), bins=bins, density=True, alpha=0.6, color="steelblue", label="real")
    ax.hist(tg.cpu(), bins=bins, density=True, alpha=0.6, color="crimson", label="gen")
    ax.set_title("terminal log-return dist"); ax.legend()

    ax = axes[1, 0]
    ax.plot(lags, acf_r.cpu(), "o-", color="steelblue", label="real")
    ax.plot(lags, acf_g.cpu(), "s--", color="crimson", label="generated")
    ax.set_title("ACF of |returns|"); ax.set_xlabel("lag (days)"); ax.legend()

    ax = axes[1, 1]
    ax.plot(xs, [math.log(max(v, 1e-6)) for v in acf_r.tolist()], "o-", color="steelblue",
            label=f"real (slope {slope_r:+.2f})")
    ax.plot(xs, [math.log(max(v, 1e-6)) for v in acf_g.tolist()], "s--", color="crimson",
            label=f"gen (slope {slope_g:+.2f})")
    ax.set_title("roughness: log-log ACF(|r|)"); ax.set_xlabel("log lag"); ax.legend()

    ax = axes[1, 2]
    log = __import__("csv")
    with open(HERE / "models" / "loss_log.csv") as f:
        reader = list(log.DictReader(f))
    iters = [int(r["iter"]) for r in reader]
    ax.plot(iters, [float(r["d_loss"]) for r in reader], color="steelblue", lw=0.8, label="D loss")
    ax.plot(iters, [float(r["g_loss"]) for r in reader], color="crimson", lw=0.8, label="G loss")
    ax.plot(iters, [float(r["gp"]) for r in reader], color="olive", lw=0.8, alpha=0.7, label="GP")
    ax.set_title("training losses"); ax.set_xlabel("gen iter"); ax.legend(fontsize=7)

    out = REPORT / "wgan_report.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nplot saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())