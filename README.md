# gpu-lab

rBergomi scenario engine + WGAN-GP path generator, built and benchmarked on an
RTX 5070 Ti Laptop (Blackwell, 12 GB) — GPU matmul 3.9 TFLOP/s vs 545 GFLOP/s
CPU (7.2×); elementwise ops are memory-bound and roughly tie.

The core idea is the BFG insight: realised volatility is *rough*
(H ≈ 0.1), so a WGAN-GP trained on rBergomi return paths must learn genuine
temporal structure — volatility clustering, a rough ACF, leverage effect —
not just marginals. Every module is validated against exact references
before anything downstream trains on it.

## Layout

| file | what it is | status |
|---|---|---|
| `rbergomi.py` | exact rBergomi engine: hybrid-scheme kernel (exact-interval weights + row normalization so Var[X_t] = t^{2H} holds exactly on the grid), chunked GPU simulation, log-Euler pricing with same-noise leverage correlation | validated 8/8 |
| `validate_rbergomi.py` | 8-check battery: BS anchor (η=0 reduces to Black-Scholes), E[v_t]=ξ₀ at 3 horizons, martingale, **independent** roughness recovery (variogram slope 0.116 vs H=0.1) *and* kernel self-consistency, put-call parity on the same MC sample | green |
| `bench_gpu.py` | honest CPU-vs-GPU numbers, elementwise and matmul | done |
| `build_dataset.py` | 100k×64 daily log-return paths with fingerprint checks (drift, variance, leverage −0.569, E[log S_T] z=+0.30 vs exact) | done |
| `train_wgan.py` | WGAN-GP (λ=10, n_critic=5, Adam 1e-4), data scaled to O(1), checkpoints + loss CSV | prototype-trained; 16k run crashed ~12,150 |
| `evaluate_wgan.py` | honest report card on structure the critic never sees directly: roughness slope, ACF of \|returns\|, leverage proxy, terminal moments. Takes a model path as argv[1] | done |

## Work in progress

- The 16k-iteration run died silently at ~iter 12,150 (error output was
  swallowed by a pipe) — rerun with output logged to file to catch the cause;
  the 12k checkpoint is the current prototype (`models/wgan_prototype.pt`).
- Known WARNs: per-step mean at a strict 4σ bar, ACF absolute levels.
- Next experiments, in order: finish the final 4k iters from checkpoint;
  1D-conv critic over the path axis if structure needs more pressure;
  then the real prize — retrain the pipeline on actual market returns
  (the BofA scenario-generation idea, now on a verified engine).

## Reproduce

```bash
uv sync                                   # pulls torch from the cu130 index (see pyproject.toml)
uv run python validate_rbergomi.py       # 8/8 must pass
uv run python build_dataset.py           # data/rbergomi_returns.pt (seeded)
uv run python train_wgan.py              # ~20 min on a 5070 Ti Laptop
uv run python evaluate_wgan.py models/wgan.pt
```

## Results so far (honest)

Small net (G 2×128, D 2×256, 6k iters): marginals OK, **structure near zero**
— roughness slope −0.02 vs real −0.37. Wider/deeper critic + generator
(3×256 / 3×512, 16k iters): roughness **−0.329 vs real −0.370 (PASS)**,
leverage −0.365 vs −0.411 (PASS), Var[log S_T] within 0.9% (PASS) —
4 PASS / 2 WARN on the 12k-iteration checkpoint (`models/wgan_prototype.pt`).

Where it still loses: per-step mean at a strict 4σ bar and ACF absolute
levels both sit in WARN; and the 16k run crashed silently at iter ~12,150
(uncaught — rerun with output logged to file is the next step).

Verdict, candidly: the architecture is validated as a *prototype* — a small
WGAN-GP can learn rough-vol temporal structure, not just marginals — but it
is not yet a scenario engine you'd stress-test a book with.