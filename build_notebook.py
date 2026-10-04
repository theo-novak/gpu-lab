"""Build train_wgan_notebook.ipynb — the VS Code-runnable training notebook.

Run: uv run python build_notebook.py
Rebuilds the notebook from scratch (idempotent, versioned by git).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells.append(md(
"""# Training the rBergomi WGAN-GP — on a 12 GB GPU, honestly

**The question this notebook answers:** *how far can a 5070 Ti Laptop (12 GB) actually go, and do the generated paths price options like the exact engine does?*

How to use: select the `.venv` kernel (`gpu-lab`), set `CONFIG` below, **Run All**. With the default config it's ~10–15 minutes end to end: two training chunks you can watch, the honest report card, and the economic verdict. Training is **chunked and checkpointed** — close the notebook mid-run, reopen, rerun the cell, and it resumes from the last checkpoint.

Everything mechanical lives in `wganlib.py` (smoke-tested end-to-end); this notebook is the story of one training run. The engine upstream of all this is validated 8/8 against exact references (`validate_rbergomi.py`), and the dataset on disk carries its own fingerprint checks (`build_dataset.py`)."""
))

cells.append(code(
"""import math, time, torch
import wganlib as wl

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert dev.type == "cuda", "this notebook wants the GPU (uv run jupyter ... from gpu-lab)"
print(f"device: {torch.cuda.get_device_name(0)}")
free, total = torch.cuda.mem_get_info()
print(f"VRAM: {total/2**20:,.0f} MiB total | {free/2**20:,.0f} MiB free")
torch.cuda.reset_peak_memory_stats()"""
))

cells.append(md(
"""## The 12 GB map — measured, not estimated

`vram_probe.py` ran real training steps (critic + gradient-penalty double-backward, the true memory profile) at each config until the card gave out. Three findings:

| axis | result |
|---|---|
| **path length** (n_steps 64 → 2048) | nearly **free**: 92 → 129 MiB. The MLP's memory lives in its hidden layers, not the sequence. Longer paths cost training-data regeneration, not VRAM. |
| **width** (G 256 → 8192, D = 2×G) | the real frontier. G 1024/D 2048: 315 MiB at 57 ms/iter. G 4096/D 8192: 3.7 GB at 603 ms/iter. |
| **batch** (128 → 8192) | cheap: batch 8192 is only 1.2 GB. Batch buys gradient quality, not model capacity. |

**The wall is not where you'd expect.** G 8192/D 16384 didn't OOM on a 12 GB card — it silently *spilled* to system RAM (Windows CUDA fallback) at **14.5 seconds per iteration, a 250× slowdown**. On this machine the honest ceiling is what fits without spilling; past it, the GPU is a fiction.

Configs this notebook offers (pick one below):

| CONFIG | G / D width | peak VRAM | ms/iter | honest note |
|---|---|---|---|---|
| `"proven"` | 256 / 512 | ~130 MiB | ~34 | the prototype that passed roughness + leverage (4 PASS / 2 WARN at 12k iters) |
| `"big"` | 1024 / 2048 | ~315 MiB | ~57 | same speed as proven, 4× the capacity — unproven quality, our quality ceiling bet |
| `"max"` | 4096 / 8192 | ~3.8 GB | ~600 | 644 MiB of weights; fits, but ~10× slower per iter — use with patience (16k iters ≈ 2.7 h) |
| `"cliff"` | 8192 / 16384 | "fits" | ~14,500 | **don't** — spills to system RAM, 250× slowdown; documented as the honest wall |"""
))

cells.append(code(
"""# ---- the knob ----
CONFIG = "proven"   # "proven" | "big" | "max"   (see the table above)

SPECS = {
    "proven": dict(hidden_g=256,  hidden_d=512),
    "big":    dict(hidden_g=1024, hidden_d=2048),
    "max":    dict(hidden_g=4096, hidden_d=8192),
}
spec = SPECS[CONFIG]
N_ITERS_PER_CHUNK = 4000
N_CHUNKS = 2                     # total iters = N_ITERS_PER_CHUNK * N_CHUNKS
print(f"config: {CONFIG}  {spec} | training {N_ITERS_PER_CHUNK * N_CHUNKS} iters in {N_CHUNKS} chunks")"""
))

cells.append(code(
"""# ---- the data (validated rBergomi return paths, see build_dataset.py) ----
data, meta = wl.load_data()
print(f"paths: {data.shape} on {data.device}")
print(f"model: H={meta['H']}, eta={meta['eta']}, rho={meta['rho']}, xi0={meta['xi0']}, T={meta['T']:.4f}y")"""
))

cells.append(code(
"""# ---- trainer (resumes automatically from the last checkpoint) ----
tr = wl.Trainer(data, depth=3, **spec)
tr.resume()"""
))

cells.append(md(
"""## Training, in reviewable chunks

Each `train(n)` call runs n generator iterations (5 critic steps each, gradient penalty λ=10), appends to the loss history, and checkpoints. Run more chunks if the curves say the run's still cooking; the checkpoint carries everything."""
))

cells.append(code(
"""tr.train(N_ITERS_PER_CHUNK)
tr.plot_loss_curves()"""
))

cells.append(code(
"""# second chunk — demonstrates resumability; add more cells if you want more
tr.train(N_ITERS_PER_CHUNK)
tr.plot_loss_curves()"""
))

cells.append(md(
"""## The honest report card

Metrics the critic was **never shown directly**: per-step moments, terminal distribution, the ACF of |returns| (volatility clustering), the roughness slope (the entire point of rough vol — theory says ≈ 2H−1 = −0.8 for the ACF of |r| at H=0.1), and the leverage proxy. Same metric code as `evaluate_wgan.py`, so numbers are comparable across runs."""
))

cells.append(code(
"""metrics = tr.evaluate(n_gen=100_000)"""
))

cells.append(md(
"""## The decisive test is economic, not statistical

Metrics are diagnostics; the arbiter is whether paths generated by the network **price a vanilla payoff like the exact engine does**. ATM European call, 200k paths each, MC errors stated, z-score between the two prices. Within 2σ: the generator didn't just learn statistics — it learned the part of the distribution a trader would actually pay for."""
))

cells.append(code(
"""verdict = tr.price_call(n_paths=200_000, K=1.0)
for k, v in verdict.items():
    print(f"  {k}: {v}")"""
))

cells.append(code(
"""peak = torch.cuda.max_memory_allocated() / 2**20
print(f"peak VRAM used by this notebook: {peak:,.0f} MiB of {total/2**20:,.0f} MiB")
print(f"headroom at CONFIG={CONFIG}: {total/2**20 - peak:,.0f} MiB — "
      f"{'comfortable' if total/2**20 - peak > 2000 else 'tight; next size up spills'}")"""
))

cells.append(md(
"""## Where this honestly stands

- **What "proven" gets you:** the prototype's verdict at 12k iters — roughness and leverage in PASS, marginal moments close, per-step mean profile at WARN. A *proof the pipeline works*, not a production scenario engine.
- **The realistic 12 GB ceiling:** `"max"` (G 4096 / D 8192) fits at 3.8 GB — capacity is not the binding constraint on this card. **Wall-clock is.** WGANs at MLP scale converge by iteration count, not FLOPs, so the honest frontier is how many ms/iter you'll tolerate: ~34 ms buys the proven size, ~600 ms buys max.
- **Length is free:** n_steps 64 → 2048 costs ~40 MiB. The expensive part of longer horizons is regenerating training data, not VRAM — an easy follow-up experiment.
- **Known next lever (not VRAM):** a 1D-convolutional critic over the path axis — structure-aware, and where the residual WARN metrics likely live.
- **The endgame this is all scaffolding for:** the same trainer, pointed at *real* return paths instead of rBergomi output. The exact engine is the practice run because it has ground truth; the market doesn't.

*The decisive test remains the one above: if the call price from generated paths doesn't match the engine, nothing else on this page matters.*"""
))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3 (gpu-lab)",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python"}

out = "train_wgan_notebook.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"built {out} with {len(cells)} cells")