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
| `wganlib.py` | the pipeline as a library: chunked `Trainer` (checkpoints, resume), module-level metrics, `generate_paths(retarget_vol=...)`, `price_call()`. Drift-centered + generator-mean-enforced (known constants are the engine's job, not the GAN's) | done |
| `train_wgan_notebook.ipynb` | VS Code notebook: knobs (proven/big/max), chunked training, honest report card, economic test, reality check vs SPY/AAPL/JPM | done, executed copy tracked |
| `realdata.py` | Yahoo chart API (no key): 2y daily returns + live spot, cached under `data/real/` | done |
| `price_cli.py` | **the ship**: `uv run python price_cli.py SPY --days 21` → live spot, realized vol, three prices side-by-side (BS baseline / exact engine / WGAN-GP with measured-verdict stamp) | shipped v1 |
| `train_centered.py` | gate-1 retrains (centered data; `proven` or `big` config), prints the economic verdict | run logs `train_*.log` |
| `tail_diag.py` | the decomposition that pinned the call-price miss on the generator's mean offset, not tails (tail-shape gap 0.0007) | diagnosis preserved |

## Ship status (v1, 2026-10-05)

Live option pricing works end-to-end: keyless spot fetch, realized-vol
calibration, BS baseline, exact engine, GAN exhibit — each priced from the
same MC budget with errors stated. The GAN's economic test **FAILs honestly
at |z|=16.5** (its verdict, measured and stamped in `models/gan_verdict.json`,
prints next to its price): structure matches the teacher, but the terminal
law near the money still differs. Per the ship rule: engine-priced truth,
GAN as a labeled exhibit, no silent asterisks. Levers queued: full big-config
run (killed at 5k by a session close — rerun `train_centered.py 16000 big`),
then a 1D-conv critic, then real-panel training.

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