# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for native pi-Flow sampling of distilled Kandinsky 6 checkpoints.

The port's risk is not that the formulas are wrong — they come from the
reference — but that the *adaptation* is: the grid axis has to land next to the
packed token axis, per-request scalars have to be expanded per token, the
substep loop masks finished entries with ``torch.where``, and raw time has to be
distinguished from shifted time throughout. Any of those can be wrong while
every tensor still has a plausible shape.

So the central test is :meth:`PolicyRolloutTest.test_matches_naive_oracle`,
which compares the vectorized implementation against :func:`_oracle_rollout` —
a deliberately naive reimplementation written from the published recursion:
one Python loop per token, one per substep, scalar interpolation indices, and no
masking. It is far too slow for real latents and exists only to disagree with
the fast path when the fast path is wrong.
"""

from __future__ import annotations

import numpy as np
import torch
from absl.testing import absltest, parameterized

from vllm_omni.diffusion.models.kandinsky6.scheduling_kandinsky6_piflow import (
    DXPolicy,
    KandinskyPiflowScheduler,
    policy_rollout_fm,
    shift_timesteps,
    split_grid_prediction,
)

# The real distilled Pro head: in_visual_dim 16 -> out_visual_dim 160 (n_grid 10),
# in_audio_dim 40 -> out_audio_dim 400.
PRO_DISTILL_TRANSFORMER_CONFIG = {
    "in_visual_dim": 16,
    "out_visual_dim": 160,
    "in_audio_dim": 40,
    "out_audio_dim": 400,
}
PRO_TRANSFORMER_CONFIG = {
    "in_visual_dim": 16,
    "out_visual_dim": 16,
    "in_audio_dim": 40,
    "out_audio_dim": 40,
}
PRO_DISTILL_SCHEDULER_CONFIG = {
    "_class_name": "PiflowScheduler",
    "shift": 5.0,
    "n_grid": 10,
    "num_policy_substeps": 128,
    "final_step_size_scale": 0.5,
    "eps": 1e-6,
}


def _unwarp(sigma: float, shift: float) -> float:
    return sigma / (shift + (1 - shift) * sigma)


def _warp(tau: float, shift: float) -> float:
    return shift * tau / (1 + (shift - 1) * tau)


def _oracle_rollout(
    grid_velocity: np.ndarray,
    x_src: np.ndarray,
    sigma_src: np.ndarray,
    segment_size: float,
    tau_src: np.ndarray,
    tau_dst: np.ndarray,
    shift: float,
    total_substeps: int,
    eps: float,
) -> np.ndarray:
    """A naive, per-token, per-substep rollout written from the formulas.

    ``grid_velocity`` is ``(T, N, ...)``, ``x_src`` is ``(T, ...)``. Nothing here
    is vectorized across tokens or substeps, and the grid lookup uses plain
    Python indices, so it shares no code path with the implementation it checks.
    """
    out = np.empty_like(x_src)
    for token in range(x_src.shape[0]):
        sigma = float(sigma_src[token])
        raw_src = _unwarp(sigma, shift)
        raw_dst = max(raw_src - segment_size, 0.0)
        segment = max(raw_src - raw_dst, eps)

        # Each grid velocity becomes an x_0 prediction at the segment's source.
        x0_grid = x_src[token][None] - sigma * grid_velocity[token]
        n_points = x0_grid.shape[0]

        delta = float(tau_src[token]) - float(tau_dst[token])
        n_substeps = max(int(np.round(delta * total_substeps)), 1)
        step = delta / n_substeps

        x_t = x_src[token].astype(np.float64).copy()
        raw_t, sigma_t = float(tau_src[token]), sigma
        for _ in range(n_substeps):
            normalized = np.clip((_unwarp(sigma_t, shift) - raw_dst) / segment, 0.0, 1.0) * (n_points - 1)
            low = int(np.clip(np.floor(normalized), 0, n_points - 2))
            high = low + 1
            x_0 = (high - normalized) * x0_grid[low] + (normalized - low) * x0_grid[high]
            velocity = (x_t - x_0) / max(sigma_t, eps)

            raw_minus = max(raw_t - step, 0.0)
            sigma_minus = _warp(raw_minus, shift)
            x_t = x_t + velocity * (sigma_minus - sigma_t)
            raw_t, sigma_t = raw_minus, sigma_minus
        out[token] = x_t
    return out


class ScheduleTest(parameterized.TestCase):
    """The segment schedule, which decides where every step lands."""

    @parameterized.parameters(1, 2, 4, 10, 50)
    def test_segments_reach_zero(self, num_steps: int) -> None:
        scheduler = KandinskyPiflowScheduler(shift=5.0, n_grid=10)
        segments = scheduler.segments(num_steps)
        self.assertLen(segments, num_steps)
        self.assertEqual(segments[0].tau_src, 1.0)
        # The schedule must consume all of raw time: a short last step leaves
        # the latent part-way along the flow and desaturates the output.
        self.assertAlmostEqual(segments[-1].tau_dst, 0.0, places=5)
        for previous, following in zip(segments, segments[1:], strict=False):
            self.assertAlmostEqual(previous.tau_dst, following.tau_src, places=12)

    def test_final_step_is_scaled(self) -> None:
        scheduler = KandinskyPiflowScheduler(shift=5.0, n_grid=10, final_step_size_scale=0.5)
        segments = scheduler.segments(10)
        base = 1.0 / (10 - 0.5)
        for segment in segments[:-1]:
            self.assertAlmostEqual(segment.segment_size, base, places=12)
        self.assertAlmostEqual(segments[-1].segment_size, base * 0.5, places=12)

    def test_timesteps_are_shifted_and_model_scale(self) -> None:
        scheduler = KandinskyPiflowScheduler(shift=5.0, n_grid=10)
        scheduler.set_timesteps(10)
        self.assertLen(scheduler.timesteps, 10)
        # Step 0 sits at tau = 1, where the shift is the identity, so the DiT
        # sees 1000. A scheduler that forgot to shift would also pass here, so
        # the second step is checked against the shift explicitly.
        self.assertAlmostEqual(float(scheduler.timesteps[0]), 1000.0, places=3)
        expected = float(shift_timesteps(torch.tensor(scheduler.segments(10)[1].tau_src), 5.0) * 1000)
        self.assertAlmostEqual(float(scheduler.timesteps[1]), expected, places=3)
        self.assertTrue(bool(torch.all(scheduler.timesteps.diff() < 0)), "timesteps must decrease")

    @parameterized.parameters(0, -1)
    def test_rejects_non_positive_steps(self, num_steps: int) -> None:
        with self.assertRaises(ValueError):
            KandinskyPiflowScheduler(shift=5.0, n_grid=10).segments(num_steps)


class GridHeadTest(absltest.TestCase):
    """``n_grid`` is derived from the head, so it cannot be configured apart."""

    def test_derives_grid_points_from_distilled_head(self) -> None:
        self.assertEqual(KandinskyPiflowScheduler.grid_points_from_config(PRO_DISTILL_TRANSFORMER_CONFIG), 10)

    def test_plain_head_has_no_grid(self) -> None:
        self.assertEqual(KandinskyPiflowScheduler.grid_points_from_config(PRO_TRANSFORMER_CONFIG), 1)

    def test_rejects_half_converted_head(self) -> None:
        config = dict(PRO_DISTILL_TRANSFORMER_CONFIG, out_audio_dim=40)
        with self.assertRaisesRegex(ValueError, "out_audio_dim"):
            KandinskyPiflowScheduler.grid_points_from_config(config)

    def test_from_configs_reads_the_checkpoint(self) -> None:
        scheduler = KandinskyPiflowScheduler.from_configs(PRO_DISTILL_SCHEDULER_CONFIG, PRO_DISTILL_TRANSFORMER_CONFIG)
        self.assertEqual(scheduler.n_grid, 10)
        self.assertEqual(scheduler.shift, 5.0)
        self.assertEqual(scheduler.num_policy_substeps, 128)
        self.assertEqual(scheduler.final_step_size_scale, 0.5)

    def test_rejects_too_small_grid(self) -> None:
        with self.assertRaises(ValueError):
            KandinskyPiflowScheduler(n_grid=1)


class SplitGridPredictionTest(absltest.TestCase):
    """The grid must land on axis 1, beside the packed token axis."""

    def test_layout_and_values(self) -> None:
        tokens, height, width, channels, n_grid = 3, 2, 5, 4, 6
        # Build a tensor whose value encodes (grid point, channel) so a wrong
        # split or a transposed movedim changes the values, not only the shape.
        per_point = [
            torch.full((tokens, height, width, channels), float(point)) + torch.arange(channels) / 100.0
            for point in range(n_grid)
        ]
        packed = torch.cat(per_point, dim=-1)
        split = split_grid_prediction(packed, n_grid)
        self.assertEqual(tuple(split.shape), (tokens, n_grid, height, width, channels))
        for point in range(n_grid):
            torch.testing.assert_close(split[:, point], per_point[point])

    def test_rejects_indivisible_channels(self) -> None:
        with self.assertRaises(ValueError):
            split_grid_prediction(torch.zeros(2, 3, 7), n_grid=4)


class PolicyTest(absltest.TestCase):
    """Invariants of the policy itself, independent of the rollout."""

    def test_velocity_at_segment_start_is_the_last_grid_point(self) -> None:
        """At the segment's source time the policy must reproduce the DiT exactly.

        The normalized position runs 0 at ``tau_dst`` to 1 at ``tau_src``, so at
        the source the interpolation selects grid point ``n - 1``; converting
        that back through ``x_0`` must return the velocity the DiT predicted. If
        raw and shifted time were confused anywhere, this is where it shows.
        """
        torch.manual_seed(0)
        shift, n_grid, eps = 5.0, 6, 1e-6
        tokens, height, width, channels = 4, 3, 2, 5
        x_src = torch.randn(tokens, height, width, channels, dtype=torch.float64)
        grid = torch.randn(tokens, n_grid, height, width, channels, dtype=torch.float64)

        tau_src = 1.0
        sigma_src = torch.full((tokens,), float(shift_timesteps(torch.tensor(tau_src), shift)), dtype=torch.float64)
        segment = torch.full((tokens,), 0.1, dtype=torch.float64)
        policy = DXPolicy(grid, x_src, sigma_src, segment, shift, eps)

        velocity = policy.pi(x_src, sigma_src)
        torch.testing.assert_close(velocity, grid[:, n_grid - 1])


class PolicyRolloutTest(parameterized.TestCase):
    """The vectorized rollout against a naive oracle built from the formulas."""

    @parameterized.named_parameters(
        ("shift5_grid10_mid", 5.0, 10, 1.0, 0.1052631578947368, 128),
        ("shift5_grid10_final_half", 5.0, 10, 0.1052631578947368, 0.0526315789473684, 128),
        ("shift1_identity_warp", 1.0, 4, 1.0, 0.5, 32),
        ("shift3_few_substeps", 3.0, 5, 0.7, 0.2, 8),
        ("shift5_last_segment_to_zero", 5.0, 10, 0.0526315789473684, 1e-6, 128),
    )
    def test_matches_naive_oracle(
        self,
        shift: float,
        n_grid: int,
        tau_src_value: float,
        tau_dst_value: float,
        substeps: int,
    ) -> None:
        torch.manual_seed(1234)
        eps = 1e-6
        tokens, height, width, channels = 5, 3, 2, 4
        segment_size = tau_src_value - tau_dst_value

        x_src = torch.randn(tokens, height, width, channels, dtype=torch.float64)
        grid = torch.randn(tokens, n_grid, height, width, channels, dtype=torch.float64) * 0.5
        tau_src = torch.full((tokens,), tau_src_value, dtype=torch.float64)
        tau_dst = torch.full((tokens,), tau_dst_value, dtype=torch.float64)
        sigma_src = shift_timesteps(tau_src, shift)
        segment = torch.full((tokens,), segment_size, dtype=torch.float64)

        policy = DXPolicy(grid, x_src, sigma_src, segment, shift, eps)
        got = policy_rollout_fm(x_src, sigma_src, tau_src, tau_dst, substeps, policy)

        expected = _oracle_rollout(
            grid.numpy(),
            x_src.numpy(),
            sigma_src.numpy(),
            segment_size,
            tau_src.numpy(),
            tau_dst.numpy(),
            shift,
            substeps,
            eps,
        )
        torch.testing.assert_close(got, torch.from_numpy(expected), rtol=1e-9, atol=1e-9)

    def test_tokens_at_different_times_are_masked_independently(self) -> None:
        """Packed tokens can belong to requests at different times.

        Entries whose segment needs fewer substeps must stop moving while the
        others keep integrating. A rollout that ignored ``active_mask`` would
        over-integrate the short entries, which the oracle catches because it
        integrates each token for its own substep count.
        """
        torch.manual_seed(7)
        shift, n_grid, substeps, eps = 5.0, 6, 16, 1e-6
        tokens, channels = 4, 3
        x_src = torch.randn(tokens, channels, dtype=torch.float64)
        grid = torch.randn(tokens, n_grid, channels, dtype=torch.float64) * 0.5
        # Deliberately uneven: 0.8 -> 16 substeps, 0.1 -> 2, so the mask matters.
        tau_src = torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float64)
        tau_dst = torch.tensor([0.2, 0.9, 0.1, 0.4], dtype=torch.float64)
        sigma_src = shift_timesteps(tau_src, shift)
        segment = tau_src - tau_dst

        policy = DXPolicy(grid, x_src, sigma_src, segment, shift, eps)
        got = policy_rollout_fm(x_src, sigma_src, tau_src, tau_dst, substeps, policy)

        # The oracle takes one segment size, so each token is rolled on its own.
        expected = np.stack(
            [
                _oracle_rollout(
                    grid[i : i + 1].numpy(),
                    x_src[i : i + 1].numpy(),
                    sigma_src[i : i + 1].numpy(),
                    float(segment[i]),
                    tau_src[i : i + 1].numpy(),
                    tau_dst[i : i + 1].numpy(),
                    shift,
                    substeps,
                    eps,
                )[0]
                for i in range(tokens)
            ]
        )
        torch.testing.assert_close(got, torch.from_numpy(expected), rtol=1e-9, atol=1e-9)


class CachedGridTest(absltest.TestCase):
    """The x_0 grid a cached step uses in place of the DiT's."""

    def _previous(self, x_0_grid: torch.Tensor):
        from vllm_omni.diffusion.models.kandinsky6.scheduling_kandinsky6_piflow import DXPolicy

        tokens = x_0_grid.shape[0]
        # Raw segment [0.5, 0.6] at shift 1 (raw == shifted time).
        return DXPolicy(
            None,
            torch.zeros(tokens, 3),
            torch.full((tokens,), 0.6),
            torch.full((tokens,), 0.1),
            shift=1.0,
            x_0_grid=x_0_grid,
        )

    def test_reuse_holds_the_last_prediction(self) -> None:
        from vllm_omni.diffusion.models.kandinsky6.scheduling_kandinsky6_piflow import cached_x_0_grid

        grid = torch.randn(2, 5, 3)
        out = cached_x_0_grid(self._previous(grid), torch.full((2, 1), 0.5), torch.full((2, 1), 0.4), "reuse")
        self.assertEqual(tuple(out.shape), (2, 1, 3))
        torch.testing.assert_close(out[:, 0], grid[:, 0])

    def test_extrapolate_continues_a_linear_trajectory_exactly(self) -> None:
        from vllm_omni.diffusion.models.kandinsky6.scheduling_kandinsky6_piflow import cached_x_0_grid

        # x_0(tau) = a + b * tau, sampled on the previous segment's grid
        # (index 0 at tau 0.5, index 4 at tau 0.6).
        a, b = torch.randn(2, 1, 3), torch.randn(2, 1, 3)
        prev_taus = torch.linspace(0.5, 0.6, 5).reshape(1, 5, 1)
        out = cached_x_0_grid(
            self._previous(a + b * prev_taus), torch.full((2, 1), 0.5), torch.full((2, 1), 0.4), "extrapolate"
        )
        taus = torch.linspace(0.4, 0.5, 5).reshape(1, 5, 1)
        torch.testing.assert_close(out, a + b * taus, rtol=1e-4, atol=1e-5)

    def test_rejects_unknown_mode(self) -> None:
        from vllm_omni.diffusion.models.kandinsky6.scheduling_kandinsky6_piflow import cached_x_0_grid

        with self.assertRaisesRegex(ValueError, "cache mode"):
            cached_x_0_grid(self._previous(torch.randn(2, 5, 3)), torch.ones(2, 1), torch.ones(2, 1), "magic")

    def test_policy_takes_exactly_one_source(self) -> None:
        from vllm_omni.diffusion.models.kandinsky6.scheduling_kandinsky6_piflow import DXPolicy

        with self.assertRaisesRegex(ValueError, "exactly one"):
            DXPolicy(None, torch.zeros(2, 3), torch.full((2,), 0.5))


if __name__ == "__main__":
    absltest.main()
