"""wganlib.py — the gpu-lab training pipeline as a notebook-callable library.

Same math as train_wgan.py (the CLI script stays the reference trainer);
this module restructures it so a Jupyter notebook can drive training in
reviewable chunks: load_data() -> Trainer.train(n) -> generate_paths() ->
price_call(). Every component keeps the exact seeds and conventions of the
CLI version, and the Trainer checkpoints after every chunk so a crashed or
interrupted run loses nothing.

Model: WGAN-GP (Gulrajani et al. 2017) on rBergomi daily log-return paths.
  G: MLP  noise_dim -> hidden -> ... -> n_steps   (returns path, O(1)-scaled)
  D: MLP  n_steps -> hidden -> ... -> 1           (critic; LayerNorm only —
     BatchNorm interacts badly with the gradient penalty)
  loss: D maximizes E[D(real)] - E[D(fake)], G minimizes -E[D(fake)], plus
     gradient penalty lambda=10 on random interpolations.
Data scaling: returns divided by global std (GP tuned for O(1) data); the
scale is stored in every checkpoint and inverted at generation time.

Honest-benchmarks rules encoded here:
  - fixed seeds everywhere (SEED, +1, +2 for G-noise, GP, batch sampling)
  - references are computed, never hand-waved constants
  - the final arbiter of "accurate paths" is economic: a call price from
    generated paths vs the exact rBergomi engine, MC error stated.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "rbergomi_returns.pt"
MODEL_DIR = HERE / "models"
REPORT_DIR = HERE / "report"

# --------------------------------------------------------------- defaults --
# The proven config (run-2 architecture): 3x256 G / 3x512 D, n_critic=5,
# lambda=10, Adam(1e-4, 0.5/0.999), batch 128. The notebook's "knobs" cell
# can override these before constructing a Trainer.
NOISE_DIM = 32
N_STEPS = 64
HIDDEN_G = 256
HIDDEN_D = 512
N_CRITIC = 5
GP_LAMBDA = 10.0
LR = 1e-4
BETAS = (0.5, 0.999)
BATCH = 128
SEED = 2026


# ----------------------------------------------------------------- models --
class Generator(nn.Module):
    """noise -> return path. Width/depth are notebook knobs."""

    def __init__(self, noise_dim=NOISE_DIM, hidden=HIDDEN_G, n_steps=N_STEPS, depth=3):
        super().__init__()
        layers = []
        d_in = noise_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2)]
            d_in = hidden
        layers += [nn.Linear(d_in, n_steps)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, hidden=HIDDEN_D, n_steps=N_STEPS, depth=3):
        super().__init__()
        layers = []
        d_in = n_steps
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2)]
            d_in = hidden
        layers += [nn.Linear(d_in, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def gradient_penalty(critic, real, fake, gp_rng):
    """(||grad D(interp)||_2 - 1)^2 on random real/fake mixtures."""
    eps = torch.rand(real.shape[0], 1, generator=gp_rng, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    out = critic(interp)
    grads = torch.autograd.grad(
        outputs=out, inputs=interp,
        grad_outputs=torch.ones_like(out),
        create_graph=True, retain_graph=True,
    )[0]
    norms = grads.flatten(start_dim=1).norm(dim=1)
    return ((norms - 1.0) ** 2).mean()


# ---------------------------------------------------- comparison metrics --
# Module-level so the notebook can run the SAME estimators on real market
# return panels — apples-to-apples with the generator's report card.

def acf(x, k):
    """ACF at lag k of a (n_paths, T) tensor, per column pair."""
    if k == 0:
        return torch.ones(x.shape[1] - k, device=x.device)
    a = x[:, :-k] - x[:, :-k].mean(0, keepdim=True)
    b = x[:, k:] - x[:, k:].mean(0, keepdim=True)
    return (a * b).mean(0) / (
        a.pow(2).mean(0).sqrt() * b.pow(2).mean(0).sqrt() + 1e-12
    )


def acf_curve(x, lags=range(1, 16)):
    """Mean ACF(|r|) per lag over columns; pads each lag's shrinking vector
    (T-k columns) with the lag's own mean so all lags stack to one length."""
    curves = []
    for k in lags:
        v = acf(x.abs(), k)
        m = v.mean()
        pad = torch.full((x.shape[1] - 1 - v.shape[0],), m, device=x.device)
        curves.append(torch.cat([v, pad]))
    return torch.stack(curves).mean(1)


def corr_next_abs(r):
    """Leverage proxy: Corr(r_i, |r_{i+1}|) averaged over columns."""
    a = r[:, :-1] - r[:, :-1].mean(0, keepdim=True)
    b = r[:, 1:].abs() - r[:, 1:].abs().mean(0, keepdim=True)
    corr = (a * b).sum(0) / (
        a.pow(2).sum(0).sqrt() * b.pow(2).sum(0).sqrt() + 1e-12
    )
    return corr.mean().item()


def lsq_slope(xs, ys):
    """Least-squares slope — plain Python, no numpy dependency."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )


def panel_metrics(panel):
    """Summary stats for a (n_paths, T) return panel — the same estimators
    the report card uses, so real-data panels score directly against the GAN."""
    p = panel.float()
    vol = p.std(0).mean().item() * math.sqrt(252)
    m, s = p.mean(0, keepdim=True), p.std(0, keepdim=True)
    kurt = (((p - m) / s) ** 4).mean(0).mean().item()
    curve = acf_curve(p.abs())
    xs = [math.log(k) for k in range(1, len(curve) + 1)]
    slope = lsq_slope(xs, [math.log(max(v, 1e-6)) for v in curve.tolist()])
    return dict(
        vol_ann=vol,
        kurtosis=kurt,
        roughness_slope=slope,
        leverage=corr_next_abs(p),
        acf1=curve[0].item(),
        acf5=curve[4].item(),
    )


def series_scalars(r):
    """Clean 1D-series scalars (vol, kurtosis) — for real single paths, where
    per-step panel estimates would double-count overlapping windows."""
    x = r.double()
    m, s = x.mean(), x.std()
    return dict(
        vol_ann=s.item() * math.sqrt(252),
        kurtosis=(((x - m) / s) ** 4).mean().item(),
    )


# ------------------------------------------------------------------- data --
def load_data(path=DATA):
    """Load the validated dataset. Returns (returns, meta); returns on GPU."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path)
    returns = payload["returns"].to(dev)
    meta = {k: v for k, v in payload.items() if k != "returns"}
    return returns, meta


# ---------------------------------------------------------------- trainer --
class Trainer:
    """Notebook-drivable WGAN-GP trainer: chunked training + checkpoints.

    Example:
        data, meta = load_data()
        tr = Trainer(data, hidden_g=256, hidden_d=512)
        tr.train(2000)          # a chunk; curves update live
        tr.train(2000)          # another chunk; state survives restarts
        paths = tr.generate_paths(10_000)
    """

    def __init__(
        self,
        data: torch.Tensor,
        hidden_g: int = HIDDEN_G,
        hidden_d: int = HIDDEN_D,
        depth: int = 3,
        noise_dim: int = NOISE_DIM,
        n_critic: int = N_CRITIC,
        gp_lambda: float = GP_LAMBDA,
        lr: float = LR,
        betas=BETAS,
        batch: int = BATCH,
        seed: int = SEED,
        center: bool = True,
        dt: float | None = None,
    ):
        self.data = data
        self.dev = data.device
        self.n_paths, self.n_steps = data.shape
        self.scale = data.std().item()
        # Gate 1 (Oct 2026): drift centering. At flat xi0 the per-step mean is
        # a KNOWN deterministic constant, -0.5*xi0*dt — so subtract it from the
        # training data (the GAN spends capacity on structure, not on placing
        # a constant it is entitled to know) and add it back analytically at
        # generation. Cumulatively the drift is -0.030 vs terminal std 0.244:
        # 12% of a std the GAN no longer has to learn.
        self.center = center
        self.mu = data.mean(0) if center else torch.zeros(self.n_steps, device=data.device)
        self.dt = float(dt) if dt is not None else float(torch.load(DATA)["dt"])
        self.config = dict(
            hidden_g=hidden_g, hidden_d=hidden_d, depth=depth,
            noise_dim=noise_dim, n_critic=n_critic, gp_lambda=gp_lambda,
            lr=lr, betas=betas, batch=batch, seed=seed, center=center,
        )
        self.g_rng = torch.Generator(device=self.dev.type); self.g_rng.manual_seed(seed)
        self.gp_rng = torch.Generator(device=self.dev.type); self.gp_rng.manual_seed(seed + 1)
        self.batch_rng = torch.Generator(device=self.dev.type); self.batch_rng.manual_seed(seed + 2)
        # Seed the GLOBAL torch RNG too: nn.Linear's default init draws from it,
        # and it advances between constructions — without this, two same-seed
        # trainers start with DIFFERENT weights (caught by the test suite Oct 2026;
        # the CLI masked it by building one trainer per process).
        torch.manual_seed(seed)
        self.G = Generator(noise_dim, hidden_g, self.n_steps, depth).to(self.dev)
        self.D = Critic(hidden_d, self.n_steps, depth).to(self.dev)
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=betas)
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=betas)
        self.iter = 0
        self.history = []   # rows: (iter, d_real, gap=d_fake-d_real, gp, g_loss)
        self._gmean = None  # generator per-step mean (scaled units) — lazy cache

        MODEL_DIR.mkdir(exist_ok=True)
        self.ckpt_path = MODEL_DIR / "wgan_notebook.pt"

    # ----------------------------------------------------------- training --
    def train(self, n_iters: int, log_every: int = 50, ckpt_every: int = 1000):
        """Run n_iters more generator iterations. Chunkable; state persists."""
        batch, n_critic = self.config["batch"], self.config["n_critic"]
        lam = self.config["gp_lambda"]
        data = (self.data - self.mu) / self.scale
        n_paths, dev = self.n_paths, self.dev

        t0 = time.perf_counter()
        for it in range(self.iter + 1, self.iter + n_iters + 1):
            if it == self.iter + 1:
                self._gmean = None   # weights moved: cached gen-mean is stale
            for _ in range(n_critic):
                idx = torch.randint(0, n_paths, (batch,), generator=self.batch_rng, device=dev)
                real = data[idx]
                with torch.no_grad():
                    z = torch.randn(batch, self.config["noise_dim"], generator=self.g_rng, device=dev)
                    fake = self.G(z)
                d_real = self.D(real).mean()
                d_fake = self.D(fake).mean()
                gp = gradient_penalty(self.D, real, fake, self.gp_rng)
                d_loss = d_fake - d_real + lam * gp
                self.opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                self.opt_d.step()

            z = torch.randn(batch, self.config["noise_dim"], generator=self.g_rng, device=dev)
            fake = self.G(z)
            g_loss = -self.D(fake).mean()
            self.opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            self.opt_g.step()

            self.iter = it
            if it % log_every == 0:
                self.history.append(
                    (it, d_real.item(), d_fake.item() - d_real.item(),
                     (lam * gp).item(), g_loss.item())
                )
            if it % ckpt_every == 0:
                self.save()
        self.save()
        secs = time.perf_counter() - t0
        print(f"trained {n_iters} iters in {secs:.0f}s "
              f"({secs / n_iters * 1000:.0f} ms/iter) | total {self.iter} iters")

    def save(self):
        torch.save(
            {
                "iter": self.iter,
                "G": self.G.state_dict(),
                "D": self.D.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(),
                "scale": self.scale,
                "mu": self.mu,
                "config": self.config,
                "history": self.history,
            },
            self.ckpt_path,
        )

    def resume(self):
        """Load the notebook checkpoint if it exists (state survives crashes)."""
        if not self.ckpt_path.exists():
            return self
        ck = torch.load(self.ckpt_path, weights_only=False)
        cfg = ck.get("config", {})
        if (cfg.get("hidden_g"), cfg.get("hidden_d"), cfg.get("depth")) == (
            self.config["hidden_g"], self.config["hidden_d"], self.config["depth"]
        ) and cfg.get("center", False) == self.center:
            self.iter = ck["iter"]
            self.G.load_state_dict(ck["G"])
            self.D.load_state_dict(ck["D"])
            self.opt_g.load_state_dict(ck["opt_g"])
            self.opt_d.load_state_dict(ck["opt_d"])
            self.history = list(ck.get("history", []))
            print(f"resumed from iter {self.iter}")
        else:
            print("checkpoint has a different architecture — starting fresh")
        return self

    @classmethod
    def from_checkpoint(cls, path=None, data=None):
        """Rebuild a Trainer exactly as the checkpoint was trained — so the
        CLI (and future sessions) can't silently mismatch architectures or
        the centering convention."""
        path = Path(path) if path is not None else MODEL_DIR / "wgan_notebook.pt"
        ck = torch.load(path, weights_only=False)
        cfg = ck["config"]
        if data is None:
            data, _ = load_data()
        tr = cls(
            data,
            hidden_g=cfg["hidden_g"], hidden_d=cfg["hidden_d"],
            depth=cfg["depth"], noise_dim=cfg["noise_dim"],
            n_critic=cfg["n_critic"], gp_lambda=cfg["gp_lambda"],
            lr=cfg["lr"], betas=tuple(cfg["betas"]), batch=cfg["batch"],
            seed=cfg["seed"], center=cfg.get("center", False),
        )
        tr.iter = ck["iter"]
        tr.G.load_state_dict(ck["G"])
        tr.D.load_state_dict(ck["D"])
        tr.opt_g.load_state_dict(ck["opt_g"])
        tr.opt_d.load_state_dict(ck["opt_d"])
        tr.history = list(ck.get("history", []))
        return tr

    # -------------------------------------------------------- evaluation --
    def _generator_mean(self) -> torch.Tensor:
        """G's own per-step mean output (scaled units), estimated once from
        a large FIXED-seed sample and cached.

        Why this exists (tail_diag.py, Oct 2026): the WGAN critic polices the
        path DISTRIBUTION, and it lets G park a systematic per-step mean
        offset (measured: up to ~70·se on individual steps, E[log S_T] off by
        0.19σ — which was the ENTIRE call-price miss at z=-93; tail shape
        matched to 0.0007). But the centered training target has per-step
        mean EXACTLY 0 — a known constant. Enforcing it analytically is the
        same principle as drift centering: known constants are not the GAN's
        job. Generation-time fresh-z sample noise is ~1e-4/step (100k paths),
        an order below the MC error of anything priced from these paths.
        """
        if self._gmean is None:
            self.G.eval()
            with torch.no_grad():
                rng = torch.Generator(device=self.dev.type)
                rng.manual_seed(self.config["seed"] + 13)
                sample = []
                for start in range(0, 100_000, 2**14):
                    m = min(2**14, 100_000 - start)
                    z = torch.randn(m, self.config["noise_dim"], generator=rng, device=self.dev)
                    sample.append(self.G(z))
                self._gmean = torch.cat(sample).mean(0)
            self.G.train()
        return self._gmean

    def generate_paths(self, n: int, seed: int | None = None, retarget_vol: float | None = None) -> torch.Tensor:
        """n log-return paths (n, n_steps) in return units.

        retarget_vol (gate 2): if given, rescale the CENTERED paths to this
        annualized vol. Correlation-based structure (ACF, roughness slope,
        leverage) is scale-invariant, so the learned signature survives; the
        drift is rebuilt analytically for the target vol, -0.5*sigma^2*dt
        per step, instead of the teacher's.
        """
        self.G.eval()
        rng = torch.Generator(device=self.dev.type)
        rng.manual_seed(seed if seed is not None else self.config["seed"] + 7)
        paths = []
        with torch.no_grad():
            for start in range(0, n, 2**14):
                m = min(2**14, n - start)
                z = torch.randn(m, self.config["noise_dim"], generator=rng, device=self.dev)
                paths.append(self.G(z))
        centered = (torch.cat(paths) - self._generator_mean()) * self.scale
        self.G.train()
        if retarget_vol is None:
            return centered + self.mu          # original units: centered + drift
        daily = retarget_vol / math.sqrt(252.0)
        s = centered.std().item()
        drift = -0.5 * retarget_vol * retarget_vol * self.dt
        return centered * (daily / s) + drift

    def plot_loss_curves(self, save_path=None):
        """Plot D(real), Wasserstein gap, GP term, G loss from history."""
        if not self.history:
            print("no history yet — run train() first")
            return None
        import matplotlib.pyplot as plt

        its = [h[0] for h in self.history]
        d_re = [h[1] for h in self.history]
        gap = [h[2] for h in self.history]
        gp = [h[3] for h in self.history]
        g = [h[4] for h in self.history]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes[0][0].plot(its, d_re); axes[0][0].set_title("D(real)")
        axes[0][1].plot(its, gap); axes[0][1].set_title("W1 gap D(fake) − D(real)")
        axes[0][1].axhline(0, color="gray", lw=0.5)
        axes[1][0].plot(its, gp); axes[1][0].set_title("GP term (λ·gp)")
        axes[1][1].plot(its, g); axes[1][1].set_title("G loss")
        for ax in axes.flat:
            ax.set_xlabel("generator iteration")
        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            fig.savefig(save_path, dpi=130)
            plt.close(fig)
            return str(save_path)
        return fig   # unclosed: the notebook's inline backend renders it

    def evaluate(self, real=None, n_gen: int = 100_000):
        """Honest report card on structure the critic never saw directly.

        Same metrics as evaluate_wgan.py (CLI), so numbers are comparable:
          per-step mean/std, terminal E/Var, ACF(|r|), roughness slope,
          leverage proxy. Returns a dict of metrics (printed too).
        """
        gen = self.generate_paths(n_gen)
        real = self.data if real is None else real
        n = min(len(gen), len(real))
        gen, real_ = gen[:n], real[:n]

        metrics = {}
        mr, mg = real_.mean(0), gen.mean(0)
        sr, sg = real_.std(0), gen.std(0)
        se = sr / math.sqrt(n)
        metrics["per-step mean (max |z| vs 4σ)"] = (
            (mg - mr).abs() / (4 * se)).max().item()
        metrics["per-step std (max rel err)"] = (
            (sg - sr).abs() / sr).max().item()

        tr, tg = real_.sum(1), gen.sum(1)
        metrics["Var[log S_T] (rel err)"] = abs(
            tg.var().item() - tr.var().item()) / tr.var().item()
        metrics["E[log S_T] (err/0.1σm)"] = abs(
            tg.mean().item() - tr.mean().item()) / (tr.std().item() / 10)

        acf_r, acf_g = acf_curve(real_), acf_curve(gen)
        metrics["ACF|r| (mean abs dev)"] = (acf_g - acf_r).abs().mean().item()
        xs = [math.log(k) for k in range(1, 16)]
        slope_r = lsq_slope(xs, [math.log(max(v, 1e-6)) for v in acf_r.tolist()])
        slope_g = lsq_slope(xs, [math.log(max(v, 1e-6)) for v in acf_g.tolist()])
        metrics["roughness slope real"] = slope_r
        metrics["roughness slope gen"] = slope_g
        metrics["roughness slope |Δ|"] = abs(slope_g - slope_r)
        lev_r, lev_g = corr_next_abs(real_), corr_next_abs(gen)
        metrics["leverage real"] = lev_r
        metrics["leverage gen"] = lev_g
        metrics["leverage |Δ|"] = abs(lev_g - lev_r)

        print("==== honest report card (same metrics as evaluate_wgan.py) ====")
        for k, v in metrics.items():
            print(f"  {k}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return metrics

    def price_call(self, n_paths: int = 200_000, K: float = 1.0, seed: int | None = None):
        """Price an ATM European call from GENERATED paths vs the exact engine.

        The decisive test is economic: paths are 'accurate' only if they
        price vanilla payoffs like the exact rBergomi engine does. Returns a
        dict with both prices, both MC errors, and the z-score between them.
        """
        import rbergomi as rb

        payload = torch.load(DATA)
        T, dt = payload["T"], payload["dt"]
        H, eta, rho, xi0 = payload["H"], payload["eta"], payload["rho"], payload["xi0"]

        gen_paths = self.generate_paths(n_paths, seed=seed)
        ST = torch.exp(gen_paths.sum(1))            # r=0, S0=1
        pay = torch.clamp(ST - K, min=0.0)
        gan_price = pay.mean().item()
        gan_err = pay.std().item() / math.sqrt(n_paths)

        v, dW, t = rb.simulate_variance(
            n_paths, self.n_steps, T, H=H, eta=eta,
            xi0=lambda tt: torch.full_like(tt, xi0), seed=777, dev=self.dev,
        )
        ref = rb.price_european_call(v, dW, t, rho=rho, K=K, dev=self.dev)
        z = (gan_price - ref["price"]) / math.sqrt(gan_err**2 + ref["mc_error"]**2)
        return {
            "K": K,
            "GAN price": gan_price,
            "GAN MC err": gan_err,
            "engine price": ref["price"],
            "engine MC err": ref["mc_error"],
            "z (GAN vs engine)": z,
            "note": "within 2σ ⇒ generated paths price vanilla payoffs like the engine",
        }