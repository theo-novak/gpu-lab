"""VRAM probe: where does 12 GB actually run out?

Runs REAL training steps (critic+GP double-backward, generator step) at each
config and records peak CUDA memory. OOM ends an axis, honestly.

Axes probed (one at a time, others held at proven settings):
  A. path length  n_steps 64 -> 2048   (model 256/512, batch 256)
  B. width        hidden up to OOM      (n_steps 64, batch 256)
  C. batch        up to OOM             (model 512/1024, n_steps 64)

Timing per config also measured -> honest training-time extrapolation.
"""
from __future__ import annotations

import gc
import math
import time

import torch
import torch.nn as nn

DEV = torch.device("cuda")
TOTAL_MB = torch.cuda.get_device_properties(0).total_memory / 2**20


class G(nn.Module):
    def __init__(self, noise_dim, hidden, n_steps, depth=3):
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


class D(nn.Module):
    def __init__(self, hidden, n_steps, depth=3):
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


def gradient_penalty(critic, real, fake):
    eps = torch.rand(real.shape[0], 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    out = critic(interp)
    grads = torch.autograd.grad(
        outputs=out, inputs=interp,
        grad_outputs=torch.ones_like(out),
        create_graph=True, retain_graph=True,
    )[0]
    norms = grads.flatten(start_dim=1).norm(dim=1)
    return ((norms - 1.0) ** 2).mean()


def probe(hg, hd, batch, n_steps, iters=8, n_critic=5, noise_dim=32):
    """Train `iters` generator iterations for real; return (peak_MB, ms/iter)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gen = None
    try:
        gen = G(noise_dim, hg, n_steps).to(DEV)
        dis = D(hd, n_steps).to(DEV)
        opt_g = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.999))
        data = torch.randn(batch, n_steps, device=DEV)  # synthetic: memory profile only

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            for _ in range(n_critic):
                idx = torch.randint(0, batch, (batch,), device=DEV)
                real = data[idx]
                with torch.no_grad():
                    fake = gen(torch.randn(batch, noise_dim, device=DEV))
                d_loss = dis(fake).mean() - dis(real).mean() \
                    + 10.0 * gradient_penalty(dis, real, fake)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
            fake = gen(torch.randn(batch, noise_dim, device=DEV))
            g_loss = -dis(fake).mean()
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20
        del dis, opt_g, opt_d, data
        return peak, secs / iters * 1000
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() and not isinstance(
            e, torch.cuda.OutOfMemoryError
        ):
            raise
        return None, None
    finally:
        if gen is not None:
            del gen
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def params_mb(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) * 4 / 2**20


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)} | {TOTAL_MB:.0f} MiB total")
    print("\n== Axis A: path length n_steps (G 256, D 512, batch 256) ==")
    for n_steps in [64, 128, 256, 512, 1024, 2048]:
        peak, ms = probe(256, 512, 256, n_steps)
        tag = f"{peak:7.0f} MiB  {ms:7.1f} ms/iter" if peak else "OOM"
        print(f"  n_steps {n_steps:>5}: {tag}")

    print("\n== Axis B: hidden width, G=hidden D=2*hidden (n_steps 64, batch 256) ==")
    for hidden in [256, 512, 1024, 2048, 4096, 8192]:
        peak, ms = probe(hidden, 2 * hidden, 256, 64)
        tag = f"{peak:7.0f} MiB  {ms:7.1f} ms/iter" if peak else "OOM"
        print(f"  G {hidden:>5} / D {2*hidden:>5}: {tag}")

    print("\n== Axis C: batch (G 512, D 1024, n_steps 64) ==")
    for batch in [128, 256, 512, 1024, 2048, 4096, 8192]:
        peak, ms = probe(512, 1024, batch, 64)
        tag = f"{peak:7.0f} MiB  {ms:7.1f} ms/iter" if peak else "OOM"
        print(f"  batch {batch:>5}: {tag}")

    # biggest-model sanity: how big is G(4096)/D(8192) on disk?
    big_g, big_d = G(32, 4096, 64), D(8192, 64)
    print(f"\nreference: G(4096)+D(8192) parameter count = "
          f"{(params_mb(big_g) + params_mb(big_d)):.0f} MiB float32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())