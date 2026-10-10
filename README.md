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
| `wganlib.py` | the pipeline as a library: chunked `Trainer` (checkpoints, resume), module-level metrics, `generate_paths(retarget_vol=...)`, `price_call()`. Drift-centered + generator-mean-enforced (known constants are the engine's job, not the GAN's). `critic="conv"` selects the 60k-param Conv1d critic (architecture, not capacity) | done |
| `train_wgan_notebook.ipynb` | VS Code notebook: knobs (proven/big/max), chunked training, honest report card, economic test, reality check vs SPY/AAPL/JPM | done, executed copy tracked |
| `realdata.py` | Yahoo chart API (no key): 2y daily returns + live spot, cached under `data/real/` | done |
| `price_cli.py` | **the ship**: `uv run python price_cli.py SPY --days 21` → live spot, realized vol, three prices side-by-side (BS baseline / exact engine / WGAN-GP with measured-verdict stamp) | shipped v1 |
| `train_centered.py` | gate-1 retrains (centered data; `proven`, `big`, or `conv` critic), prints the economic verdict | three verdicts measured |
| `tail_diag.py` | the decomposition that pinned the call-price miss on the generator's mean offset, not tails (tail-shape gap 0.0007) | diagnosis preserved |
| `tests/` | 20-test CPU suite (~2 s, no GPU/data): determinism, centering, mean enforcement, retarget-vol, checkpoint round-trips + arch refusal, ConvCritic invariants; session-isolated from `models/` (a near-miss: tests once overwrote the active slot) | 20/20 |

## v2: the tail-extrapolation question (2026-10-10, contract ba090e2)

One question, pre-registered before any v2 training: **does the path-GAN
beat historical block-resampling precisely beyond the observed tail —
the one place resampling is structurally blind?** 10y of SPY (2,513
returns incl. COVID's −11.6%, ~2,386 independent windows vs 2y's ~7);
block-252 persistence-honest bootstrap CIs (the 64-block CIs failed their
own self-containment and were widened BEFORE any v2 weights moved). New
gates: R5 (closer to real q99.9 than hist-sim, high AND low) and R6
(generated kurtosis ≤ 1.5× real — no exploding tails). A three-rung
dispersion ladder was pre-declared: A plain retrain → B +variance aux →
C +tail-weighted sampling (γ=1).

Verdict (full table in `models/real_gate_v2.json`), two-sided as always:

- **The ladder worked**: dispersion 0.45 → 0.58 → **0.95** of real,
  kurtosis inside the cap, structure PASS on every rung. No
  rearchitecture needed — training-time shaping fixes what the objective
  under-weights.
- **The capability win is real**: run C puts ~1 day in 12,800 **beyond
  the observed sample extreme** (the COVID-magnitude event) and matches
  the low tail 3.5× better than the bootstrap (0.30 vs 1.07pp). The
  bootstrap cannot emit a day beyond its sample max — BY CONSTRUCTION.
- **The calibrated bar still says no**: R5's high-tail bar went to
  hist-sim by 0.034pp (0.905 vs 0.871 — noise-close, but the contract
  was pre-registered and the letter stands). Mid-tail honesty: γ=1
  overshot q99.5-high (3.87 vs real 3.27).
- **More data does not teach tails**: run A's 10y retrain produced
  Gaussian tails (q99.9 3.57% vs Gaussian's 3.56%) with the project's
  best-ever structure. Data scale is neutral-to-tails; the residual gap
  is objective-bound — same conclusion as the synthetic gate.

So: **resample for calibrated accuracy; train if you need probability
past your sample's horizon.** Both claims measured under one committed
contract. Reopen lever: the objective itself (tail/payoff-aware critic)
or regime-conditional generation — NOT more data (run A).

Incident worth its own line: a resume without a named slot stamped 1k
checkpoints into the default slot and clobbered the synthetic exhibit.
Library rule added: checkpoints remember `ckpt_file` provenance and
`from_checkpoint` restores the right slot. 31-test suite pins it. The
rebuild reproduced FAIL with a DIFFERENT magnitude (|z| −15.8 → −34.1):
GPU conv training is not bit-deterministic (atomic accumulation order in
the conv backward), so identical seeds reproduce the recipe, not the
weights — a second draw from the same training distribution, and run-to-run
spread is itself evidence the FAIL is not one unlucky draw.

## Ship status (v1 FINAL, 2026-10-05)

Live option pricing works end-to-end: keyless spot fetch, realized-vol
calibration, BS baseline, exact engine, GAN exhibit — each priced from the
same MC budget with errors stated, and the GAN's measured economic-test
verdict (stamped in `models/gan_verdict.json`) printed next to its price.

Gate 1 is closed, honestly: **three critic architectures at 16k iters
(drift-centered, generator-mean enforced) all FAIL the economic test** —
proven MLP |z|=+16.5, 4×-capacity MLP |z|=−24.2, 60k-param Conv1d |z|=−15.8
— each with a different error profile. The conv critic matches structure
closest (roughness −0.30 vs −0.37, leverage Δ 0.007, per-step mean inside
the 4σ bar) yet stays 9.4% low on terminal variance, which the call price
feels directly. Verdict across all three: the binding constraint is the
WGAN objective itself — the critic polices path distributions, never the
terminal law pointwise. Capacity moves the miss around; architecture moves
it; neither closes it.

On live quotes the conv GAN sits closest to the engine of any run (SPY
$769.64 ATM 21d: BS 14.58 / engine 12.78 / GAN 12.56, errors ±0.03).
Engine-priced truth, GAN as a labeled exhibit — no silent asterisks. If the
gate is ever reopened: a payoff-aware auxiliary loss (caveat stated: that
makes the training objective the test itself), or real-panel training
where no closed-form engine exists to arbitrate.

## Real-panel endgame (2026-10-05, the BofA regime actually run)

Conv critic, 16k iters, on 437 overlapping 64-day panels of real SPY 2y
returns — pre-registered gate (`evaluate_real.py`, committed BEFORE
training: moving-block-bootstrap CIs, Gaussian floor, memorization check):

- **R1 PASS 4/4**: roughness −0.36, ACF1 0.20, ACF5 0.12, leverage −0.11 —
  every one inside the data's own 90% sampling envelope. But that envelope
  is WIDE (one 2y series ≈ 7 independent 64-day stretches): pass means
  "consistent with what this data can certify", nothing stronger.
- **R2 PASS**: decisively beats iid-Gaussian on every clustering/leverage
  row (Gaussian sits at 0.000 by construction).
- **vs historical block-resampling — NO win**: the free, untrained
  baseline matches the real estimators as well or better on every row
  (roughness −0.49 vs GAN −0.36, real −0.51). On this data size the GAN
  buys nothing the bootstrap doesn't already give.
- **R3 FAIL (the carried signature)**: zero verbatim copies, but panel
  variance ratio 0.45 < the 0.5 floor — generated dispersion ≈ HALF the
  real, kurtosis 9.9 vs real 22.6. Same under-dispersion/tail-taming the
  synthetic gate showed (−9.4% there, −50% here), whether the teacher is a
  parametric model or reality itself: it's the objective+architecture's
  signature, not the teacher's.
- R4 (descriptive): ATM 21d at matched 16.45% vol — GAN 1.41c vs BS 1.89c:
  thinner tails price cheaper.

Verdict: on minimal real data the path-WGAN is a real structure learner
that fails dispersion — and at this sample size loses its raison d'être to
literal resampling. Two runs worth keeping: `models/wgan_real_spy.pt`
(gate file `models/real_gate.json`).

Two incidents from the arc, encoded as rules: `from_checkpoint` now
REQUIRES its training data, fingerprint-verified (a naive resume would
have finished training the real-panel model on synthetic data); and panels
must be moved to the GPU explicitly — the first 11k iters of this run
trained on CPU at 264 ms/iter because fetch_returns hands back a CPU
tensor (fixed: 60 ms/iter on resumé).

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

Where it loses, finally measured: the economic call-price gate FAILs across
all three critic architectures (see Ship status). Structure is not the
bottleneck — the terminal law is.

Verdict, candidly: the architecture is validated as a *prototype* — a small
WGAN-GP can learn rough-vol temporal structure, not just marginals — but it
is not yet a scenario engine you'd stress-test a book with.