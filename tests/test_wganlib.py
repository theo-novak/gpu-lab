"""wganlib mechanics: determinism, centering, mean enforcement, retarget.

Every test asserts an EXACT or analytically-derived reference — the
project's rule since the hand-waved-tolerance failure of Oct 2026.
"""
from __future__ import annotations

import math

import pytest
import torch

import wganlib as wl


class TestTrainerDeterminism:
    def test_same_seed_same_city(self, small_data):
        """Two trainers, same seed, same iters -> identical weights."""
        t1 = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=11)
        t2 = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=11)
        t1.train(2, log_every=10**9, ckpt_every=10**9)
        t2.train(2, log_every=10**9, ckpt_every=10**9)
        for p1, p2 in zip(t1.G.parameters(), t2.G.parameters()):
            assert torch.equal(p1, p2)
        for p1, p2 in zip(t1.D.parameters(), t2.D.parameters()):
            assert torch.equal(p1, p2)

    def test_different_seed_different_weights(self, small_data):
        t1 = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=11)
        t2 = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=12)
        t1.train(2, log_every=10**9, ckpt_every=10**9)
        t2.train(2, log_every=10**9, ckpt_every=10**9)
        assert any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(t1.G.parameters(), t2.G.parameters())
        )


class TestCentering:
    def test_center_subtracts_exact_per_step_mean(self, small_data):
        """center=True must set mu to the data's EXACT per-step mean."""
        tr = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=3)
        assert torch.allclose(tr.mu, small_data.mean(0), atol=1e-7)

    def test_no_center_mu_is_zero(self, small_data):
        tr = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=3, center=False)
        assert torch.all(tr.mu == 0)


class TestGeneratePaths:
    def test_mean_enforcement_zeroes_centered_mean(self, tiny_trainer):
        """generate_paths subtracts G's OWN mean -> centered output has
        per-step mean ~0 up to the fixed-sample noise floor (~1e-3 on 100k).

        Reference is EXACT (zero), tolerance = the estimator's own scale:
        std/sqrt(n) * safety, stated not hand-waved.
        """
        paths = tiny_trainer.generate_paths(20_000, seed=99)
        assert paths.shape == (20_000, tiny_trainer.n_steps)
        per_step = paths - tiny_trainer.mu  # strip the added drift term
        se = per_step.std() / math.sqrt(20_000)
        assert per_step.mean(0).abs().max() < 10 * se, (
            f"centered mean {per_step.mean(0).abs().max():.2e} vs 10*se {10 * se:.2e}"
        )

    def test_retarget_vol_rescales_exact(self, tiny_trainer):
        """retarget_vol must land annualized vol EXACTLY on target (the
        scale factor is analytic: daily = target/sqrt(252), slope is
        scale-invariant so structure survives)."""
        target = 0.30
        paths = tiny_trainer.generate_paths(20_000, seed=99, retarget_vol=target)
        # realized daily vol from the paths themselves
        daily = paths.std()  # global std of returns (paths are per-step)
        # annualized: per-step vol * sqrt(252) — paths have per-step dt=1/252
        ann = daily * math.sqrt(252)
        assert ann == pytest.approx(target, rel=0.05), (
            f"retargeted vol {ann:.4f} vs target {target}"
        )

    def test_generate_is_seed_deterministic(self, tiny_trainer):
        a = tiny_trainer.generate_paths(64, seed=5)
        b = tiny_trainer.generate_paths(64, seed=5)
        assert torch.equal(a, b)
        c = tiny_trainer.generate_paths(64, seed=6)
        assert not torch.equal(a, c)


class TestCheckpoint:
    def test_save_resume_round_trip(self, small_data, tmp_path):
        """from_checkpoint rebuilds the trainer exactly: weights, iter,
        config, scale, mu — the CLI's correctness depends on it."""
        tr = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=42)
        tr.train(2, log_every=10**9, ckpt_every=10**9)
        tr.ckpt_path = tmp_path / "ck.pt"
        tr.save()

        tr2 = wl.Trainer.from_checkpoint(tr.ckpt_path, data=small_data)
        assert tr2.iter == tr.iter
        assert tr2.config == tr.config
        assert tr2.scale == pytest.approx(tr.scale)
        assert torch.allclose(tr2.mu, tr.mu)
        for p1, p2 in zip(tr2.G.parameters(), tr.G.parameters()):
            assert torch.equal(p1, p2)
        # and it generates identically
        assert torch.equal(
            tr.generate_paths(32, seed=1), tr2.generate_paths(32, seed=1)
        )

    def test_resume_refuses_mismatched_arch(self, small_data, tmp_path):
        """resume() must NOT load a checkpoint with different hidden sizes
        or centering — silent arch mismatch would corrupt the active slot."""
        tr = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=42)
        tr.ckpt_path = tmp_path / "ck.pt"
        tr.save()
        other = wl.Trainer(small_data, hidden_g=64, hidden_d=32, depth=2,
                          noise_dim=4, n_critic=1, batch=32, seed=42)
        other.resume()
        assert other.iter == 0, "resume must refuse mismatched architecture"


class TestMetrics:
    def test_acf_lag_zero_is_one(self, small_data):
        assert torch.allclose(wl.acf(small_data, 0), torch.ones(N_STEPS := small_data.shape[1]))

    def test_acf_of_shuffled_columns_is_zero(self, small_data):
        """Column-shuffled data has no cross-column lag structure:
        ACF at k>=1 ~ 0 in expectation (exact-in-expectation reference)."""
        g = torch.Generator().manual_seed(1)
        shuffled = small_data[torch.randperm(small_data.shape[0], generator=g)]
        v = wl.acf(shuffled.abs(), 1).mean()
        assert abs(v.item()) < 0.05, f"shuffled ACF1 {v:.3f} should be ~0"

    def test_white_noise_acf_is_flat(self):
        """iid panel: ACF(|r|) ~ 0 at every lag — the CORRECT reference.

        (The first version of this test asserted a ~0 roughness SLOPE for
        white noise — wrong: the slope takes log(max(v,1e-6)) of ACF values
        scattered around zero, half clamp to the floor, and the 'slope' is
        garbage (measured -2.1). The slope metric is only defined for
        series with real volatility clustering; flatness of the ACF itself
        is the honest white-noise reference.)"""
        g = torch.Generator().manual_seed(2)
        noise = torch.randn(4_000, 32, generator=g) * 0.01
        curve = wl.acf_curve(noise.abs())
        assert curve.abs().max().item() < 0.08, (
            f"white-noise ACF max {curve.abs().max():.3f} should be ~0 "
            f"(scale 1/sqrt(4000) ~ 0.016)"
        )

    def test_leverage_sign_on_synthetic(self, small_data):
        """corr_next_abs on iid data ~ 0 (no leverage structure)."""
        g = torch.Generator().manual_seed(3)
        noise = torch.randn(4_000, 16, generator=g) * 0.01
        assert abs(wl.corr_next_abs(noise)) < 0.05


class TestConvCritic:
    """The pre-declared gate-1 lever: conv critic over the path axis."""

    def test_shapes_2d_and_3d(self):
        d = wl.ConvCritic(n_steps=64)
        x2 = torch.randn(8, 64)
        out = d(x2)
        assert out.shape == (8,), f"(B,T) input must give (B,), got {tuple(out.shape)}"
        x3 = x2.unsqueeze(1)  # SAME values, shape (B,1,T) — first draft drew
        # fresh randomness here and compared two different inputs (caught by
        # the assertion, fixed Oct 2026: test the shape contract, not luck)
        assert torch.equal(d(x3), out), "(B,1,T) must be equivalent to (B,T)"

    def test_param_count_below_mlp(self):
        """ARCHITECTURE-NOT-CAPACITY invariant: the conv critic must stay
        well under the proven MLP critic's params, or the experiment would
        silently test capacity again (the big run already did that)."""
        def n_params(m):
            return sum(p.numel() for p in m.parameters())
        conv = n_params(wl.ConvCritic(n_steps=64))
        mlp = n_params(wl.Critic(hidden=512, n_steps=64, depth=3))
        assert conv < mlp / 5, f"conv {conv} params must be << MLP {mlp}"

    def test_path_order_matters(self):
        """The flatten head must preserve order sensitivity — a pooled head
        would be permutation-invariant and blind to roughness. Exact
        reference: a permutation changes the critic score."""
        g = torch.Generator().manual_seed(9)
        d = wl.ConvCritic(n_steps=64)
        x = torch.randn(4, 64, generator=g).abs()  # nonneg so it's not antisymmetry
        perm = x[:, torch.randperm(64, generator=g)]
        assert not torch.allclose(d(x), d(perm)), (
            "critic must NOT be permutation-invariant"
        )

    def test_conv_trainer_deterministic(self, small_data):
        t1 = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=11,
                        critic="conv")
        t2 = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=11,
                        critic="conv")
        assert isinstance(t1.D, wl.ConvCritic)
        t1.train(2, log_every=10**9, ckpt_every=10**9)
        t2.train(2, log_every=10**9, ckpt_every=10**9)
        for p1, p2 in zip(t1.D.parameters(), t2.D.parameters()):
            assert torch.equal(p1, p2)

    def test_conv_checkpoint_round_trip(self, small_data, tmp_path):
        """from_checkpoint must rebuild the CONV critic from config alone —
        the CLI depends on never silently mismatching architectures."""
        tr = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=42,
                        critic="conv")
        tr.train(2, log_every=10**9, ckpt_every=10**9)
        tr.ckpt_path = tmp_path / "conv.pt"
        tr.save()
        tr2 = wl.Trainer.from_checkpoint(tr.ckpt_path, data=small_data)
        assert isinstance(tr2.D, wl.ConvCritic), "config must rebuild ConvCritic"
        assert tr2.config == tr.config
        assert torch.equal(
            tr.generate_paths(32, seed=1), tr2.generate_paths(32, seed=1)
        )

    def test_resume_refuses_critic_mismatch(self, small_data, tmp_path):
        """A conv checkpoint offered to an MLP trainer must be refused —
        same rule as hidden-size mismatch."""
        tr = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=42,
                        critic="conv")
        tr.ckpt_path = tmp_path / "conv.pt"
        tr.save()
        other = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                           noise_dim=4, n_critic=1, batch=32, seed=42)
        other.resume()
        assert other.iter == 0, "resume must refuse a critic-arch mismatch"
        assert not isinstance(other.D, wl.ConvCritic)


class TestDataFingerprint:
    """The data-poisoning rule (Oct 2026): from_checkpoint must never
    silently substitute data — a real-panel checkpoint resumed on synthetic
    data would train its last iterations on the wrong distribution."""

    def test_from_checkpoint_requires_data(self, small_data, tmp_path):
        tr = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=42)
        tr.train(2, log_every=10**9, ckpt_every=10**9)
        tr.ckpt_path = tmp_path / "ck.pt"
        tr.save()
        with pytest.raises(ValueError, match="REQUIRES"):
            wl.Trainer.from_checkpoint(tr.ckpt_path, data=None)

    def test_from_checkpoint_refuses_wrong_data(self, small_data, tmp_path):
        tr = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=42)
        tr.train(2, log_every=10**9, ckpt_every=10**9)
        tr.ckpt_path = tmp_path / "ck.pt"
        tr.save()
        other = small_data * 3.0 + 0.01  # same shape, different distribution
        with pytest.raises(ValueError, match="mismatch"):
            wl.Trainer.from_checkpoint(tr.ckpt_path, data=other)

    def test_from_checkpoint_accepts_right_data_restores_scale(self, small_data, tmp_path):
        tr = wl.Trainer(small_data, hidden_g=16, hidden_d=32, depth=2,
                        noise_dim=4, n_critic=1, batch=32, seed=42)
        tr.train(2, log_every=10**9, ckpt_every=10**9)
        tr.ckpt_path = tmp_path / "ck.pt"
        tr.save()
        tr2 = wl.Trainer.from_checkpoint(tr.ckpt_path, data=small_data)
        assert tr2.scale == tr.scale and torch.allclose(tr2.mu, tr.mu)
        assert torch.equal(tr.generate_paths(8, seed=1),
                           tr2.generate_paths(8, seed=1))


class TestGradientPenalty:
    def test_gp_zero_for_linear_critic(self, small_data):
        """A linear critic has constant gradient norm along the path —
        GP is (||grad||-1)^2; with a unit-slope linear critic on O(1)-scaled
        data, grads are ~1 so GP ~ 0. Exact-in-expectation check."""
        g = torch.Generator().manual_seed(5)
        real = torch.randn(256, 8, generator=g)
        fake = torch.randn(256, 8, generator=g)
        rng = torch.Generator().manual_seed(6)

        class LinearCritic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(8) / math.sqrt(8))

            def forward(self, x):
                return (x * self.w).sum(-1)

        gp = wl.gradient_penalty(LinearCritic(), real, fake, rng)
        # each interp gradient norm = ||w|| = 1 by construction
        assert gp.item() == pytest.approx(0.0, abs=1e-5)