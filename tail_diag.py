"""Tail diagnostic: WHY does the GAN underprice the call?

A call is a right-tail bet. Decompose each panel's call price into:
  C_actual : mean of (exp(logST) - K)+ over the panel   [what the pricer sees]
  C_ln     : exact lognormal price fitted to the panel's OWN terminal (m, s^2)
             [what the terminal FIRST TWO MOMENTS alone are worth]

If C_actual ≈ C_ln for the GAN but C_actual >> C_ln for the teacher, the
GAN's terminal law has collapsed to near-lognormal (thin tails) and the
missing price is exactly the tail shape beyond the second moment.
"""
import math

import torch

import rbergomi as rb
import wganlib as wl

K = 1.0
N = 100_000
Phi = torch.distributions.Normal(0.0, 1.0).cdf


def lognormal_call(m: float, s: float, K: float) -> float:
    if s <= 0:
        return max(math.exp(m) - K, 0.0)
    d1 = (m + s * s - math.log(K)) / s
    d2 = (m - math.log(K)) / s
    return float(math.exp(m + s * s / 2) * Phi(torch.tensor(d1)).item()
                 - K * Phi(torch.tensor(d2)).item())


def decompose(name: str, logST: torch.Tensor):
    m, s = logST.mean().item(), logST.std().item()
    actual = torch.clamp(torch.exp(logST) - K, min=0).mean().item()
    ln = lognormal_call(m, s, K)
    kurt = (((logST - logST.mean()) / logST.std()) ** 4).mean().item()
    print(f"{name:<14} m={m:+.4f} s={s:.4f} kurt={kurt:6.2f} | "
          f"C_actual={actual:.4f} C_lognormal={ln:.4f} tail_shape={actual - ln:+.4f}")
    return actual, ln


print(f"K = {K} (ATM). N = {N:,} paths per panel. tail_shape = C_actual - C_lognormal\n")

data, _ = wl.load_data()
real_logST = data[:N].sum(1)

tr = wl.Trainer.from_checkpoint()
gen = tr.generate_paths(N, seed=2027)
gan_logST = gen.sum(1)

# the engine's OWN ground truth: price_european_call said 0.0812 on fresh sims;
# the dataset panel is an independent 100k sample of the same law
a_r, ln_r = decompose("rBergomi", real_logST)
a_g, ln_g = decompose("GAN (16k ctr)", gan_logST)

print(f"\nengine MC reference price : 0.0812")
print(f"GAN  vs engine gap        : {a_g - 0.0812:+.4f}  (the economic miss)")
print(f"moment-matched gap       : {ln_g - ln_r:+.4f}  (what matched (m,s) would still miss)")
print(f"tail-shape gap            : {(a_g - ln_g) - (a_r - ln_r):+.4f}  (beyond-second-moment part)")
print(f"\nGAN kurtosis of logST vs teacher: see kurt column — lognormal is 3.0")