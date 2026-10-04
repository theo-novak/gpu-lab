"""Bucket 4: WGAN-GP training on the rBergomi return-path dataset.

Recipe (Gulrajani et al. 2017 defaults):
  - critic (D) and generator (G) MLPs, LayerNorm only (BatchNorm interacts
    badly with the gradient penalty)
  - n_critic = 5 D-steps per G-step, GP lambda = 10
  - Adam lr 1e-4, betas (0.5, 0.999)
  - data scaled to O(1) by the global std of returns (penalty is tuned for
    O(1) data); the scale is stored in the checkpoint and inverted at eval

Outputs:
  models/wgan.pt       — final weights + full config + scale
  models/loss_log.csv   — iter, d_real, d_fake, gp, d_loss, g_loss
  models/wgan_ckpt_N.pt — snapshots every 2000 iters
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

# ----------------------------- config ------------------------------------
NOISE_DIM = 32
N_STEPS = 64
HIDDEN_G = 256
HIDDEN_D = 512
N_CRITIC = 5
GP_LAMBDA = 10.0
LR = 1e-4
BETAS = (0.5, 0.999)
BATCH = 128
GEN_ITERS = 16000
CKPT_EVERY = 4000
LOG_EVERY = 50
SEED = 2026

HERE = Path(__file__).parent
DATA = HERE / "data" / "rbergomi_returns.pt"
MODEL_DIR = HERE / "models"


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, HIDDEN_G), nn.LayerNorm(HIDDEN_G), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_G, HIDDEN_G), nn.LayerNorm(HIDDEN_G), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_G, HIDDEN_G), nn.LayerNorm(HIDDEN_G), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_G, N_STEPS),   # raw output: scaled return path
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_STEPS, HIDDEN_D), nn.LayerNorm(HIDDEN_D), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_D, HIDDEN_D), nn.LayerNorm(HIDDEN_D), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_D, HIDDEN_D), nn.LayerNorm(HIDDEN_D), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_D, 1),
        )

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


def main() -> int:
    t_start = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_DIR.mkdir(exist_ok=True)

    payload = torch.load(DATA)
    returns = payload["returns"].to(dev)          # (100k, 64) float32
    n_paths, n_steps = returns.shape
    assert n_steps == N_STEPS, f"expected {N_STEPS} steps, data has {n_steps}"
    scale = returns.std().item()                  # global std, ~sqrt(xi0*dt)
    data = returns / scale                        # O(1) training data
    print(f"data: {n_paths} x {n_steps} | std scale {scale:.5f} -> O(1)")
    print(f"device: {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}")

    g_rng = torch.Generator(device=dev.type); g_rng.manual_seed(SEED)
    gp_rng = torch.Generator(device=dev.type); gp_rng.manual_seed(SEED + 1)
    batch_rng = torch.Generator(device=dev.type); batch_rng.manual_seed(SEED + 2)

    G, D = Generator().to(dev), Critic().to(dev)
    opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    def rand_batch(n):
        idx = torch.randint(0, n_paths, (n,), generator=batch_rng, device=dev)
        return data[idx]

    csv_path = MODEL_DIR / "loss_log.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(["iter", "d_real", "d_fake", "gp", "d_loss", "g_loss"])

    print(f"\ntraining: {GEN_ITERS} generator iterations, n_critic={N_CRITIC}, "
          f"lambda={GP_LAMBDA}, batch={BATCH}")
    for it in range(1, GEN_ITERS + 1):
        # ---- critic steps ----
        for _ in range(N_CRITIC):
            real = rand_batch(BATCH)
            with torch.no_grad():
                z = torch.randn(BATCH, NOISE_DIM, generator=g_rng, device=dev)
                fake = G(z)
            d_real = D(real).mean()
            d_fake = D(fake).mean()
            gp = gradient_penalty(D, real, fake, gp_rng)
            d_loss = d_fake - d_real + GP_LAMBDA * gp
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward(retain_graph=False)
            opt_d.step()

        # ---- generator step ----
        z = torch.randn(BATCH, NOISE_DIM, generator=g_rng, device=dev)
        fake = G(z)
        g_loss = -D(fake).mean()
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        if it % LOG_EVERY == 0:
            with csv_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    it,
                    f"{d_real.item():+.4f}", f"{d_fake.item():+.4f}",
                    f"{gp.item():.4f}", f"{d_loss.item():+.4f}", f"{g_loss.item():+.4f}",
                ])
            if it % 500 == 0:
                el = time.perf_counter() - t_start
                print(f"  iter {it:>5} | D(real) {d_real.item():+.3f} "
                      f"D(fake) {d_fake.item():+.3f} | GP {gp.item():7.3f} "
                      f"| G {g_loss.item():+.3f} | {el:6.0f}s")

        if it % CKPT_EVERY == 0:
            torch.save({"iter": it, "G": G.state_dict(), "D": D.state_dict(),
                        "scale": scale}, MODEL_DIR / f"wgan_ckpt_{it}.pt")

    torch.save(
        {
            "iter": GEN_ITERS, "G": G.state_dict(), "D": D.state_dict(),
            "scale": scale,
            "config": dict(noise_dim=NOISE_DIM, n_steps=N_STEPS, hidden_g=HIDDEN_G,
                           hidden_d=HIDDEN_D, n_critic=N_CRITIC, gp_lambda=GP_LAMBDA,
                           lr=LR, betas=BETAS, batch=BATCH, seed=SEED),
        },
        MODEL_DIR / "wgan.pt",
    )
    total = time.perf_counter() - t_start
    print(f"\ndone: {GEN_ITERS} iters in {total/60:.1f} min "
          f"({total/GEN_ITERS*1000:.0f} ms/gen-iter)")
    print(f"saved: {MODEL_DIR / 'wgan.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())