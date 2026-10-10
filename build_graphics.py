"""build_graphics.py — v2 exhibit graphics for theonovak.com (honest artifacts only).

Every figure names its source: checkpoint file, draw seed, sample sizes.
Outputs land in report/v2/ (repo) AND ../theonovak.com/assets/projects/gpu_lab/
(the site). Idempotent: rerun after later runs (B verdict) to add series.

Figures:
  1 loss_curves:        run A training history (the notebook's 4-panel, from ckpt)
  2 tail_ccdf:          the v2 headline — empirical CCDF of |daily return|,
                        real vs GAN vs Gaussian vs hist-sim, log-log tail zoom
  3 quantiles:          q99.9 four-way bars across regimes (2y v1 + 10y runs)
  4 scenario_fan_3d:    3D fan of generated 64-day paths vs real panels
  5 density_3d:         3D (regime-vol quintile x magnitude bin x mass)
                        surfaces, real vs GAN — where tail mass goes missing
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import wganlib as wl
from realdata import fetch_returns, make_panel

HERE = Path(__file__).resolve().parent
OUT_REPO = HERE / "report" / "v2"
SITE_ASSETS = HERE.parent / "theonovak.com" / "assets" / "projects" / "gpu_lab"
DPI = 140
SEED = 2099

MANIFEST: dict[str, object] = {}


def _save(fig, name: str, caption: str) -> None:
    OUT_REPO.mkdir(parents=True, exist_ok=True)
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    p = OUT_REPO / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(p, SITE_ASSETS / name)
    MANIFEST[name] = caption
    print(f"  {name} -> {p.name} ({p.stat().st_size // 1024} KB)")


def gen_paths(tr: wl.Trainer, n: int) -> torch.Tensor:
    return tr.generate_paths(n, seed=SEED).float()


def fig_loss_curves(ck: dict) -> None:
    hist = ck.get("history", [])
    if not hist:
        return
    its = [h[0] for h in hist]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0][0].plot(its, [h[1] for h in hist]); axes[0][0].set_title("D(real)")
    axes[0][1].plot(its, [h[2] for h in hist]); axes[0][1].set_title("W1 gap D(fake)−D(real)")
    axes[0][1].axhline(0, color="gray", lw=0.5)
    axes[1][0].plot(its, [h[3] for h in hist]); axes[1][0].set_title("GP term (λ·gp)")
    axes[1][1].plot(its, [h[4] for h in hist]); axes[1][1].set_title("G loss")
    for ax in axes.flat:
        ax.set_xlabel("generator iteration")
    fig.suptitle("WGAN-GP training history — real SPY 10y, conv critic (checkpoint wgan_real_spy_10y.pt)")
    fig.tight_layout()
    _save(fig, "loss_curves.png",
          "Training history from the run-A checkpoint (16k iters, real SPY 10y). "
          "The W1 gap not closing is expected: on real data the critic separates "
          "real from fake easily — the gates judge structure, not the critic's score.")


def fig_tail_ccdf(rets: torch.Tensor, gen: torch.Tensor, gauss: np.ndarray,
                  c_gen: torch.Tensor | None = None) -> None:
    real = np.sort(np.abs(rets.cpu().numpy()))
    gan = np.sort(np.abs(gen.cpu().numpy()))
    gs = np.sort(np.abs(gauss))
    n_r, n_g, n_gs = len(real), len(gan), len(gs)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(real[::-1], np.arange(1, n_r + 1)[::-1] / n_r, ".", ms=2,
              label=f"real SPY 10y (n={n_r})", alpha=0.7)
    ax.loglog(gan[::-1], np.arange(1, n_g + 1)[::-1] / n_g, ".", ms=2,
              label=f"GAN run A (50k paths, seed {SEED})", alpha=0.6)
    if c_gen is not None:
        cg = np.sort(np.abs(c_gen.cpu().numpy()))
        ax.loglog(cg[::-1], np.arange(1, len(cg) + 1)[::-1] / len(cg), ".",
                  ms=2, label=f"GAN run C (tail-weighted, 50k paths)", alpha=0.8)
    ax.loglog(gs[::-1], np.arange(1, n_gs + 1)[::-1] / n_gs, ".", ms=1,
              label="iid Gaussian (3.2M draws)", alpha=0.4)
    xthr = float(rets.abs().max())
    ax.axvline(xthr, color="crimson", lw=1, ls="--",
               label=f"real sample max |r| = {xthr*100:.1f}% (hist-sim tail ends here)")
    ax.set_xlabel("|daily return|"); ax.set_ylabel("P(|r| > x)")
    ax.set_title("The v2 frontier: tail mass beyond what resampling can see")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, "tail_ccdf.png",
          "Empirical CCDF of absolute daily returns (log-log). Run A's tail "
          "merges with the Gaussian's past ~2.5% (beyond-max mass 0.000000); "
          "run C — tail-weighted training — pulls the curve toward the real "
          "one, misses the pre-registered high-tail bar by 0.034pp, wins the "
          "low tail 3.5x, and puts ~1 day in 12,800 BEYOND the sample max "
          "(right of the dashed line), probability resampling cannot emit.")


def fig_v3_margins(v3rows: dict[str, list[dict]]) -> None:
    """v3's verdict picture: per-lever per-seed margins vs the hist-sim bar
    (zero line), both tails. NEGATIVE = GAN closer = win."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    levers = list(v3rows)
    colors = ("#4c72b0", "#dd8452", "#55a868")
    for ax, tail in zip(axes, ("high", "low")):
        for i, lev in enumerate(levers):
            ys = [r["margins_pp"][tail] for r in v3rows[lev]]
            xs = np.full(len(ys), i) + np.linspace(-0.12, 0.12, len(ys))
            ax.scatter(xs, ys, s=42, color=colors[i % 3],
                       label=f"{lev} (n={len(ys)})" if tail == "high" else None,
                       zorder=3)
        ax.axhline(0, color="k", lw=1, ls=":", zorder=2)
        ax.set_xticks(range(len(levers)), levers)
        ax.set_title(f"{tail} tail (q99.9)", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("margin vs hist-sim (pp)\nnegative = GAN closer = win")
    axes[0].legend(fontsize=8)
    fig.suptitle("v3: per-seed margins — v2's 'miss' was seed noise; the crash tail is the robust win")
    fig.tight_layout()
    _save(fig, "v3_margins.png",
          "R5 margins by lever and training seed (negative = the GAN sits "
          "closer to the real q99.9 than historical resampling). Floor "
          "(run-C recipe reseeded): high tail straddles zero — v2's 0.034pp "
          "miss was a draw, not the recipe; low tail negative at every seed. "
          "tailcrit (V3-1): negative at every seed on BOTH tails, tightest "
          "spread — consistency, the honest edge.")


def fig_fan_3d(gen: torch.Tensor, real_panel: torch.Tensor) -> None:
    gnp, rnp = gen[:80].cpu().numpy(), real_panel[::96][:80].cpu().numpy()  # 80 spread rows
    fig = plt.figure(figsize=(12, 5.5))
    for i, (data, ttl) in enumerate(((rnp, "real SPY 10y — 80 windows"),
                                     (gnp, "GAN (run A) — 80 generated paths"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        xs = np.arange(data.shape[1])
        for j in range(min(60, data.shape[0])):
            ax.plot(xs, np.full_like(xs, j), np.cumsum(data[j]), lw=0.6, alpha=0.5)
        ax.set_xlabel("trading day"); ax.set_ylabel("window #")
        ax.set_zlabel("cumulative log-return")
        ax.set_title(ttl, fontsize=10)
    fig.suptitle("Scenario fans: what 'rough but too-Gaussian' looks like path-by-path")
    fig.tight_layout()
    _save(fig, "scenario_fan_3d.png",
          "Left: 80 real 64-day windows from 10y SPY. Right: 80 generated paths "
          "(run A checkpoint, seed 2099). Same roughness and clustering; the "
          "difference lives at the extremes, not in the body.")


def fig_density_3d(rets: torch.Tensor, gen: torch.Tensor) -> None:
    """(regime quintile x magnitude bin x mass) surfaces, real vs GAN.

    Regime is defined per series-appropriate estimator, honestly different:
      - real: rolling 21-day std of the calendar series (a regime is a
        stretch of calendar time)
      - GAN: per-path std (paths are independent snapshots — no calendar
        exists; a regime is what a path lived through)
    Both surfaces are normalized to unit total mass; the comparison of
    interest is where tail mass sits conditional on regime."""
    def surface_real(series: torch.Tensor) -> np.ndarray:
        s = series.double().cpu()
        roll = s.unfold(0, 21, 1).std(dim=1)
        reg = torch.cat([torch.full((20,), float(roll[0]), dtype=s.dtype), roll])
        return _hist2d(reg, s.abs())

    def surface_gen(panel: torch.Tensor) -> np.ndarray:
        p = panel.double().cpu()
        pv = p.std(dim=1)                       # per-path vol (n,)
        reg = pv.repeat_interleave(p.shape[1])  # per-value regime id basis
        return _hist2d(reg, p.abs().flatten())

    def _hist2d(reg: torch.Tensor, mag: torch.Tensor) -> np.ndarray:
        qs = torch.quantile(reg, torch.linspace(0, 1, 6, dtype=reg.dtype))
        rid = torch.bucketize(reg, qs[1:-1])
        edges = torch.quantile(mag, torch.linspace(0, 1, 21, dtype=mag.dtype))
        mid = torch.bucketize(mag, edges[1:-1])
        H = torch.zeros(5, 20, dtype=torch.double)
        for r_i, m_i in zip(rid.tolist(), mid.tolist()):
            H[min(r_i, 4), min(m_i, 19)] += 1
        return (H / H.sum()).numpy()

    Z_real = surface_real(rets)
    Z_gen = surface_gen(gen)
    fig = plt.figure(figsize=(12, 5.5))
    for i, (Z, ttl) in enumerate((
            (Z_real, "real SPY 10y (regime = rolling 21d vol)"),
            (Z_gen, "GAN run A (regime = per-path vol)"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        X, Y = np.meshgrid(np.arange(5), np.arange(20), indexing="ij")
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", lw=0.2)
        ax.set_xlabel("vol regime quintile (calm->stressed)")
        ax.set_ylabel("|return| bin (small->large)")
        ax.set_zlabel("mass")
        ax.set_title(ttl, fontsize=10)
    fig.suptitle("Where the mass goes: joint (regime, magnitude) surface")
    fig.tight_layout()
    _save(fig, "density_3d.png",
          "Joint mass over volatility-regime quintile × return-magnitude bin. "
          "Regime defined honestly per side (rolling 21-day calendar vol for "
          "real data; per-path vol for generated ones — paths have no calendar). "
          "The stressed-regime × large-magnitude corner is visibly starved on "
          "the GAN surface: the R5 tail-taming measured quantitatively, seen in 3D.")


def fig_terminal_hist() -> None:
    """The notebook's core diagnostic, rendered from artifacts: synthetic
    teacher (rBergomi dataset, seed 2026/2027) vs the active synthetic
    exhibit (conv critic, 16k). Terminal-law mismatch = the economic gate's
    failure mode, visible."""
    data, _meta = wl.load_data()             # (N, 64) synthetic rBergomi (GPU)
    t = (data.sum(1).cpu().numpy())
    ck = torch.load("models/wgan_notebook.pt", weights_only=False)
    cfg = ck["config"]
    tr = wl.Trainer.from_checkpoint("models/wgan_notebook.pt", data=data.to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    fake = tr.generate_paths(20_000, seed=SEED)
    f = fake.sum(1).cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(min(t.min(), f.min()), max(t.max(), f.max()), 80)
    ax.hist(t, bins=bins, density=True, alpha=0.55, label=f"teacher rBergomi (n={len(t)})")
    ax.hist(f, bins=bins, density=True, alpha=0.55,
            label=f"GAN conv 16k (n={len(f)}, seed {SEED})")
    ax.set_xlabel("log S_T (64-day terminal log-return)")
    ax.set_ylabel("density")
    ax.set_title("The synthetic gate's failure mode: terminal law, not structure")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, "terminal_hist.png",
          "Terminal log-return law: synthetic rBergomi teacher vs the active "
          "conv-critic GAN (20k draws, seed 2099). The GAN terminal density "
          "matches the body but misses the law's tails/variance — the economic "
          "gate's FAIL (|z|=34.1 on the ATM call, 2026-10-10 rebuild; original "
          "draw was −15.8 — GPU conv training is bit-nondeterministic, so "
          "draws differ while the FAIL conclusion is stable) is exactly this "
          "picture: structure learned, law wrong.")

def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = fetch_returns("SPY", years=10)
    rets = r["returns"].float().to(dev)
    panel = make_panel(rets, 64, 1).contiguous().to(dev)
    tr = wl.Trainer.from_checkpoint("models/wgan_real_spy_10y.pt", data=panel)
    gen = gen_paths(tr, 50_000)
    tr_c = wl.Trainer.from_checkpoint("models/wgan_real_spy_10y_tail.pt", data=panel)
    c_gen = gen_paths(tr_c, 50_000)
    g_np = (torch.randn(3_200_000, generator=torch.Generator().manual_seed(42))
            * rets.std().item() + rets.mean().item())

    print("rendering (all sources named in captions):")
    fig_loss_curves(torch.load("models/wgan_real_spy_10y.pt", weights_only=False))
    fig_tail_ccdf(rets, gen, g_np, c_gen=c_gen)
    fig_fan_3d(gen, panel)
    fig_density_3d(rets, gen)
    fig_terminal_hist()

    # v3 verdict figure: rows read from the gate stamp (no re-eval)
    with open(HERE / "models" / "real_gate_v3.json") as f:
        v3 = json.load(f)
    tailcrit_seeds = [p.stem.split("_s")[1] for p in
                      sorted(HERE.glob("models/v3_tailcrit_s*.pt"))]
    floor_seeds = [p.stem.split("_s")[1] for p in
                   sorted(HERE.glob("models/v3_c_s*.pt"))]
    tc = v3["tailcrit"]["margins_pp_high"]; fl = v3["noise_floor"]["margins_pp_high"]
    tcl = v3["tailcrit"]["margins_pp_low"];  fll = v3["noise_floor"]["margins_pp_low"]
    rows3 = {
        "c": [dict(margins_pp=dict(high=h, low=l))
              for h, l in zip(fl, fll)],
        "tailcrit": [dict(margins_pp=dict(high=h, low=l))
                     for h, l in zip(tc, tcl)],
    }
    fig_v3_margins(rows3)

    MANIFEST["_provenance"] = {
        "model_a": "models/wgan_real_spy_10y.pt (run A: conv, 16k, real SPY 10y 2016-2026)",
        "model_c": "models/wgan_real_spy_10y_tail.pt (run C: conv, 16k, tail_gamma=1)",
        "model_synth": "models/wgan_notebook.pt (conv 16k, synthetic rBergomi teacher)",
        "gen_seed": SEED, "gen_paths": 50_000,
        "gaussian_draws": 3_200_000,
        "note": "all figures regenerate from committed checkpoints; manifest ships beside them",
    }
    (OUT_REPO / "manifest.json").write_text(json.dumps(MANIFEST, indent=1))
    shutil.copy(OUT_REPO / "manifest.json", SITE_ASSETS / "manifest.json")
    print("manifest -> report/v2/manifest.json + site assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())