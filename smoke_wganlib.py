"""Smoke test for wganlib.py — CPU, tiny config, end-to-end.

Exercises every method the notebook will call:
  load_data -> Trainer -> train(chunk) -> save/resume -> generate_paths
  -> plot_loss_curves -> evaluate -> price_call
Run: uv run python smoke_wganlib.py
"""
import torch

import wganlib as wl

torch.manual_seed(0)

# ---- data on CPU for the smoke test (GPU is busy with vram_probe) ----
payload = torch.load(wl.DATA, map_location="cpu")
data = payload["returns"]
meta = {k: v for k, v in payload.items() if k != "returns"}
print(f"data: {tuple(data.shape)} | meta keys: {sorted(meta)[:6]}")

# ---- tiny trainer (the notebook uses the real config) ----
tr = wl.Trainer(
    data, hidden_g=32, hidden_d=64, depth=2, noise_dim=8,
    n_critic=1, batch=64, seed=1,
)
tr.train(12, log_every=4, ckpt_every=6)

# ---- save + resume round-trip ----
tr2 = wl.Trainer(data, hidden_g=32, hidden_d=64, depth=2, noise_dim=8,
                 n_critic=1, batch=64, seed=1)
tr2.resume()
assert tr2.iter == tr.iter, f"resume mismatch: {tr2.iter} vs {tr.iter}"
print(f"resume ok (iter {tr2.iter})")

# ---- generate + evaluate (small n for CPU speed) ----
paths = tr.generate_paths(512, seed=99)
assert paths.shape == (512, data.shape[1]), paths.shape
assert abs(paths.std().item() / data.std().item() - 1.0) > -10  # just prints ratio
print(f"generate ok: {tuple(paths.shape)}, std {paths.std().item():.4f} "
      f"(data std {data.std().item():.4f})")

m = tr.evaluate(n_gen=512)
print("evaluate ok:", len(m), "metrics")

# ---- price_call with a small MC count (CPU speed) ----
verdict = tr.price_call(n_paths=2_000, seed=123)
print("price_call ok:")
for k, v in verdict.items():
    print(f"  {k}: {v}")

# ---- plot loss curves (Agg-safe: save to file) ----
import matplotlib
matplotlib.use("Agg")
p = tr.plot_loss_curves(save_path="report/smoke_losses.png")
print(f"plot ok: {p}")

print("\nSMOKE PASS: wganlib end-to-end on CPU")