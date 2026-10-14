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
| `scenario_cli.py` | **the other ship** (uses-branch): prompted ticker input (or argv list), per-ticker resampled 64-day path-bundle graphs + drawdown tables; the tool that beat the GAN, as a 1-command UX | shipped 2026-10-10 |
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

## v3: seeds, noise floor, and the crash-tail win (2026-10-10, contract e1bdf77)

v2 left 0.034pp of high-tail miss on the table — and the rebuild incident
had already taught us single draws aren't claims. So v3's contract was
amended BEFORE any v3 training: **lever verdicts at the median of >=3
fresh training seeds, spread reported**, plus a noise-floor lever (run
C's recipe re-seeded). Checkpoints: `models/v3_<lever>_s<seed>.pt`.

Verdict (`models/real_gate_v3.json`, three seeds per lever):

- **v2's calibrated-FAIL was seed noise.** The floor's high-tail margins
  straddle zero (−0.40 / −0.58 / +0.04pp); v2's letter-miss sits inside
  that spread. The LOW tail beats historical resampling at every seed
  (−0.65 / −0.67 / −1.06pp) — the crash tail is a real, robust GAN win,
  ~3× closer than the bootstrap.
- **V3-1 (tail-weighted critic loss) passes at 3/3 seeds** — the only
  lever with R5+R6+R1..R3 green everywhere — with spreads tighter than
  the recipe's own (0.41 / 0.19pp). Its median edge over the free lever
  is <=0.2pp, inside its own spread: it buys **consistency**, not
  magnitude. Honest cost: var-ratio drift 1.13–1.17 (slight overshoot).
- **V3-2 (`termq`) stays scaffolded and unrun** — declared, not fired; a
  lever that trains the terminal law directly is for a different
  question.

Updated claim: **resample for convenience; train when you need the crash
tail closer than resampling can see, or probability beyond your sample's
horizon.**

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

## Practical uses (uses-branch: `feature/practical-uses`)

The science lives on `main`; this branch accumulates **plain-English ways
to use what the project built.** Rule: a use ships only with its honesty
note — what the tool can't tell you is part of the tool.

### Getting the package running (one-time, ~2 minutes)

```bash
git clone https://github.com/theo-novak/gpu-lab && cd gpu-lab
uv sync                     # installs everything; torch comes from the cu130 index
uv run pytest tests/ -q     # sanity: 34/34 must pass
```

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and any
NVIDIA GPU (CPU works for everything except training speed; the market
data is fetched keyless from Yahoo on first use, then cached in
`data/real/`). Network needed on first fetch per ticker only.

### 1. `scenario_cli.py` — resampled scenario bundles (SHIPPED)

**The tool that beat the GAN, as a one-command UX.** Historical
block-resampling won v2/v3's calibrated gates; this wraps it so you never
touch the library.

```bash
uv run python scenario_cli.py            # prompts: "Ticker(s), comma-separated [SPY]:"
uv run python scenario_cli.py SPY QQQ    # or skip the prompt
uv run python scenario_cli.py --help     # also prints this honest preamble
# flags: --years 5  --paths 2000  --days 64  --seed 2026  --quiet
```

Input format (prompt or argv): comma-separated Yahoo tickers —
`SPY`, `SPY, QQQ, 2773.HK`, `BTC-USD`. Per ticker it prints a mini risk
table (p5/median/p95 of the N-day return, drawdown percentiles,
P(touch a 10% drawdown), the observed worst-day ceiling) and writes
`report/scenarios/<TICKER>_scenario_<date>.png` — sample paths + the
p5–p95 band + median + stats footer, reproducible by seed.

Example output (see `docs/scenario_cli_SPY_example.png`; from the first
ten-tech run, 5y history, seed 2026, 2,000 paths each): MSFT/AAPL are
the boring tails (P(DD>10%) 61–65%), NVDA pairs a −31% bad-quarter tail
with +52% upside, and TSLA touches a double-digit drawdown in **99.5%**
of its 2,000 resampled quarters.

**Honesty note (on every chart):** resampling's tail IS its sample —
no generated day can exceed the observed worst day. The v2/v3 verdict's
complement: calibrated accuracy comes from resampling; the crash-tail /
beyond-extreme capability belongs to the trained models (see v3).

### 2. `pulse_cli.py` — the market-pulse daily log (SHIPPED)

One row per run per ticker: date, spot, realized vol, ATM-21d Black-
Scholes vs the rough-vol engine price, and their spread — appended to
`report/pulse/pulse.csv`. Run it daily; the log becomes a growing
personal record of how the BS-vs-rough-vol disagreement moves through
regimes (a Oct-2025-style vol spike shows up loudly).

```bash
uv run python pulse_cli.py SPY QQQ       # snapshot now (any tickers)
uv run python pulse_cli.py --show        # pretty-print the log so far
uv run python pulse_cli.py --n 20        # last 20 rows
# flags: --days 21 (option horizon)  --vol-years 2 (realized-vol window)
# daily automation: Windows Task Scheduler -> "uv run python pulse_cli.py"
#   from the repo dir; or ask Hermes for a cronjob.
```

Example first row (2026-10-14): SPY spot 768.09, realized vol 11.1%,
BS 9.80 vs engine 8.68 → spread **−11.4%** (the rough-vol view prices
short ATM calls lower at calm tape). Read the spread as model
disagreement, not edge.

**Honesty note:** the engine prices under rough vol (H=0.1) calibrated
to realized vol, no IV surface (option chains are auth-gated keyless) —
both prices share the same input vol by design. It's a fixed-lens
observation of regimes, not a trading signal.

### 3. `mc_oracle.py` — rough-vol MC odds for arbitrary scenarios (SHIPPED)

The validated engine as an oracle: "odds of X over N days, under rough
volatility rather than GBM." Calibrated to any ticker's realized vol, or
run at any explicit vol level.

```bash
uv run python mc_oracle.py SPY --dd 10 --lose 5        # P(DD>=10%), P(ret<=-5%)
uv run python mc_oracle.py SPY --days 126 --gain 20    # longer horizon
uv run python mc_oracle.py --vol 45 --dd 20 --plot     # generic 45%-vol name
# flags: --days 64  --paths 65536  --vol-years 5  --seed 2026  --plot
```

Pass any combination of `--dd X` (P of a max-drawdown ≥ X), `--lose X`
(P of total return ≤ −X), `--gain X` (P of ≥ +X). Horizon is any number
of trading days — the engine is exact, no grid cap (that was the GAN's
limit, not this model's). `--plot` also renders the path fan with p5/
median/p95 to `report/oracle/`.

Sample answers (seed 2026, 65,536 paths): SPY at 17.1% realized vol,
64d: P(DD≥10%) 26.3% — vs 24.3% from scenario_cli's resampler. Two
independent tools, one question, ~2pp apart: a running cross-check.
A generic 45%-vol name over 126d: P(DD≥20%) 53.8%.

**Honesty note:** answers are under the rBergomi **model** view
(H=0.1, η=1.9, ρ=−0.9, flat forward variance) — a structured assumption,
not a forecast. Flat ξ₀ means no vol term structure; single-asset means
no correlations (use portfolio_cli for joint questions).

### 4. `portfolio_cli.py` — joint portfolio stress table (SHIPPED)

Scenario bundles for a WHOLE portfolio: weights + tickers in, the
portfolio's own drawdown distribution out. **Joint calendar bootstrap**:
each scenario re-samples one common historical window for every name,
so real cross-correlations survive (independent per-name resampling
would silently destroy them — the classic quiet mistake).

```bash
uv run python portfolio_cli.py                                  # prompts
uv run python portfolio_cli.py "SPY:40,QQQ:25,NVDA:15,V:10,CASH:10"
uv run python portfolio_cli.py SPY:.5 QQQ:.3 TSLA:.2 --days 126 --plot
# weights: percentages (40) or fractions (.4); CASH = 0-return damping
# flags: --days 64  --paths 3000  --years 5  --seed 2026  --plot
```

Output: joint-history window (names are date-aligned to their overlap,
printed), the market-weighted p5/median/p95 portfolio return, the
portfolio maxDD distribution (1-in-20 vs median), P(DD≥10%),
P(total≤−5%), and a path-fan chart to `report/portfolio/`.

Example (`SPY:40,QQQ:25,NVDA:15,V:20`, 5y joint history 2021-10..2026-10,
3,000 × 64d scenarios): p5 −12.1%, median +5.3%, 1-in-20 maxDD −24.4%,
P(DD≥10%) 38.1%. Adding CASH (0-return damped variance): P(DD≥10%)
drops to 28.4% — diversification you can *see*.

**Honesty note:** the portfolio's tail is the sample's tail (no day
beyond the joint history's observed extremes); today's weights are
static within each scenario window; CASH earns nothing (no rate).

### The four tools at a glance

| tool | question it answers | engine? | GPU? | output |
|---|---|---|---|---|
| `scenario_cli.py` | what do plausible N-day futures look like per ticker? | no (resampling) | no | per-ticker path-fan PNGs + risk table |
| `pulse_cli.py` | how does the rough-vol view deviate from BS over time? | yes | no (CPU MC) | appending CSV log |
| `mc_oracle.py` | what are the odds of X under rough vol? | yes | yes (seconds) | printed odds (+ optional PNG) |
| `portfolio_cli.py` | what does MY portfolio's bad quarter look like? | no (joint resampling) | no | portfolio path-fan PNG + DD table |

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