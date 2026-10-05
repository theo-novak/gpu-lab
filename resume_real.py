"""Resume the real-panel SPY run from its 11k checkpoint to 16k, then gate.

The resume rule (learned the hard way, Oct 2026): from_checkpoint REQUIRES
the data the checkpoint was trained on — fingerprint-verified. The panel is
rebuilt EXACTLY as train_real.py built it (same fetch cache, same window).
"""
import torch

import wganlib as wl
from realdata import fetch_returns, make_panel

r = fetch_returns("SPY", years=2)
# data-poisoning rule: same panel, same fingerprint — but ON THE GPU.
# (First 11k iters of this run trained on CPU by accident — the panel was
# never moved off fetch_returns' CPU tensor, hence 264 ms/iter and a 0%-GPU
# mystery that took two wrong theories before this one-line diagnosis.
# Same math either way; the checkpoint is valid.)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
panel = make_panel(r["returns"].float(), 64, 1).contiguous().to(dev)
tr = wl.Trainer.from_checkpoint("models/wgan_real_spy.pt", data=panel)
print(f"resumed: iter {tr.iter} | target 16000 | center {tr.config['center']} "
      f"| critic {tr.config['critic']} | fingerprint verified")
tr.train(16000 - tr.iter)
print()
print("==== honest report card vs the real panel (same estimators) ====")
tr.evaluate(real=panel, n_gen=panel.shape[0])