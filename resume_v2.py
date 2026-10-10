"""resume_v2.py — resume run A from its checkpoint (fingerprint-verified, GPU).

Same panel construction as train_real_v2.py (byte-identical fetch cache,
window, stride), same checkpoint slot; the fingerprint check refuses to
load if any of that drifted.
"""
import torch

import wganlib as wl
from realdata import fetch_returns, make_panel

r = fetch_returns("SPY", years=10)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
panel = make_panel(r["returns"].float(), 64, 1).contiguous().to(dev)
tr = wl.Trainer.from_checkpoint("models/wgan_real_spy_10y.pt", data=panel)
print(f"resumed: iter {tr.iter} | target 16000 | {dev} | conv critic | fingerprint verified")
tr.train(16000 - tr.iter)

print("\n==== honest report card vs the real 10y panel ====")
tr.evaluate(real=panel, n_gen=panel.shape[0])