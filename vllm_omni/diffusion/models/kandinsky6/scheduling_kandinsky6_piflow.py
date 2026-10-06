# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

r"""Native pi-Flow (``PiflowScheduler``) sampling for distilled Kandinsky 6.

``kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`` samples in 10 steps with
``PiflowScheduler``, which :mod:`scheduling_kandinsky6`'s Euler stepper cannot
express. pi-Flow does not take a step along a predicted velocity: each DiT call
returns a *policy* over the whole segment ahead, and the sampler then integrates
that policy without calling the network again.

The shape of one outer step:

1. At flow-matching time ``tau_src`` the DiT is called once. Its output is not a
   velocity but ``n_grid`` predictions of ``x_0``, spread over the segment
   ``[tau_dst, tau_src]`` — hence the distilled checkpoint's output head being
   ``n_grid`` times wider (``out_visual_dim: 160`` for ``in_visual_dim: 16``).
2. :class:`DXPolicy` turns that grid into a function ``pi(x_t, sigma_t)``: look
   up where ``t`` falls in the segment, interpolate between the two bracketing
   grid points to get ``x_0``, and return the velocity ``(x_t - x_0) / sigma_t``
   that points there.
3. :func:`policy_rollout_fm` integrates that velocity across the segment in
   ``num_substeps`` Euler substeps — all elementwise tensor math, no DiT.

So 10 DiT calls still buy roughly 130 integration substeps. That is what makes
the distilled checkpoint worth serving, and it is also why the substeps must be
cheap: they run on the full latent, 13 times per step.

Two details decide correctness, and both are easy to get wrong:

**Raw time versus shifted time.** The schedule is uniform in *raw* time ``tau``,
but the DiT is conditioned on *shifted* time ``sigma = shift*tau/(1+(shift-1)tau)``
(``shift`` 5.0 here). Segments are therefore computed in ``tau`` and the policy
is evaluated in ``sigma``; :meth:`DXPolicy._unwarp_t` inverts the shift to move
back. Mixing the two silently biases every step.

**The final step is half-length.** ``final_step_size_scale`` (0.5) shortens the
last segment, so the segment size is ``1/(num_steps - (1 - scale))`` and not
``1/num_steps``. Getting this wrong leaves the last step landing short of
``tau = 0`` and desaturates the output.

Ported from ``kandinskylab/kandinsky-6`` (``kandinsky/core/algo/piflow_math.py``
and ``piflow_sampler.py``, MIT) and kept deliberately close to it: the formulas
are the contract with the distilled weights, so this module stays readable
against the reference rather than refactored away from it. The packed-sequence
adaptation is ours — vLLM-Omni carries latents as ``(sum_T, H, W, C)`` with
``cu_seqlens`` instead of a leading batch dimension, so per-request scalars are
expanded per token with ``repeat_interleave``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

_MIN_GRID_POINTS = 2


def shift_timesteps(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Raw flow-matching time -> the shifted time the DiT is conditioned on."""
    return shift * t / (1 + (shift - 1) * t)


class DXPolicy:
    """A network-free velocity field over one flow-matching segment.

    Built from one DiT call's ``n_grid`` predictions of ``x_0``; evaluating
    :meth:`pi` at any time inside the segment costs an interpolation, not a
    forward pass.
    """

    def __init__(
        self,
        denoising_output: torch.Tensor,
        x_t_src: torch.Tensor,
        sigma_t_src: torch.Tensor,
        segment_size: float | torch.Tensor = 1.0,
        shift: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.shift = shift
        self.eps = eps

        # Trailing singleton dims so a per-token scalar broadcasts over H, W, C.
        self.sigma_t_src = sigma_t_src.reshape(*sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))
        self.raw_t_src = self._unwarp_t(self.sigma_t_src)

        segment = segment_size
        if isinstance(segment, torch.Tensor) and segment.dim() < self.raw_t_src.dim():
            segment = segment.reshape(*segment.size(), *((self.raw_t_src.dim() - segment.dim()) * [1]))
        self.raw_t_dst = (self.raw_t_src - segment).clamp(min=0)
        self.segment_size = (self.raw_t_src - self.raw_t_dst).clamp(min=eps)
        self.denoising_output_x_0 = self._u_to_x_0(denoising_output, self.x_t_src, self.sigma_t_src)

    def _unwarp_t(self, sigma_t: torch.Tensor) -> torch.Tensor:
        """Inverse of :func:`shift_timesteps`."""
        return sigma_t / (self.shift + (1 - self.shift) * sigma_t)

    @staticmethod
    def _u_to_x_0(denoising_output: torch.Tensor, x_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        """Grid of velocities -> grid of ``x_0`` predictions (grid axis is 1)."""
        return x_t.unsqueeze(1) - sigma_t.unsqueeze(1) * denoising_output

    @staticmethod
    def _interpolate(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate the grid axis of ``x`` at normalized ``t``."""
        n = x.size(1)
        if n < _MIN_GRID_POINTS:
            return x.squeeze(1)
        t = t.clamp(min=0, max=1) * (n - 1)
        t0 = t.floor().to(torch.long).clamp(min=0, max=n - 2)
        t1 = t0 + 1
        indices = torch.stack([t0, t1], dim=1)
        values = torch.gather(x, dim=1, index=indices.expand(-1, -1, *x.shape[2:]))
        return (t1 - t) * values[:, 0] + (t - t0) * values[:, 1]

    def pi(self, x_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        """The policy's velocity at ``(x_t, sigma_t)``."""
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        raw_t = self._unwarp_t(sigma_t)
        x_0 = self._interpolate(self.denoising_output_x_0, (raw_t - self.raw_t_dst) / self.segment_size)
        return (x_t - x_0) / sigma_t.clamp(min=self.eps)


def policy_rollout_fm(
    x_t_start: torch.Tensor,
    sigma_t_start: torch.Tensor,
    raw_t_start: torch.Tensor,
    raw_t_end: torch.Tensor,
    total_substeps: int,
    policy: DXPolicy,
) -> torch.Tensor:
    """Integrate ``policy.pi`` from ``raw_t_start`` down to ``raw_t_end``.

    The substep count is proportional to the segment length
    (``round(delta_tau * total_substeps)``), so the short final segment takes
    proportionally fewer substeps and every segment integrates at the same
    resolution in time. Entries that have finished early are frozen by
    ``active_mask`` rather than skipped, because in the packed layout different
    tokens can belong to requests at different times.

    The result keeps ``x_t_start``'s dtype. The schedule tensors are FP32 (a BF16
    sigma near the end of the schedule has too few mantissa bits to separate
    adjacent substeps), and plain promotion would hand the VAE an FP32 latent
    where the rest of the pipeline -- and the Euler scheduler, which casts its
    step to ``sample.dtype`` -- keeps BF16.
    """
    ndim = x_t_start.dim()
    shape = (x_t_start.size(0), *((ndim - 1) * [1]))
    raw_t = raw_t_start.reshape(shape)
    raw_t_end = raw_t_end.reshape(shape)
    sigma_t = sigma_t_start.reshape(shape)

    delta_raw_t = raw_t - raw_t_end
    num_substeps = (delta_raw_t * total_substeps).round().to(torch.long).clamp(min=1)
    substep_size = delta_raw_t / num_substeps

    x_t = x_t_start
    for substep_id in range(int(num_substeps.max().item())):
        velocity = policy.pi(x_t, sigma_t)
        raw_t_minus = (raw_t - substep_size).clamp(min=0)
        sigma_t_minus = shift_timesteps(raw_t_minus, policy.shift)
        x_t_minus = x_t + velocity * (sigma_t_minus - sigma_t)

        active = num_substeps > substep_id
        x_t = torch.where(active, x_t_minus.to(x_t.dtype), x_t)
        sigma_t = torch.where(active, sigma_t_minus, sigma_t)
        raw_t = torch.where(active, raw_t_minus, raw_t)

    return x_t.to(x_t_start.dtype)


@dataclass
class PiflowSegment:
    """One outer step's segment, in raw flow-matching time."""

    step_index: int
    tau_src: float
    tau_dst: float
    segment_size: float
    is_final: bool

    @property
    def sigma_src(self) -> float:
        """Unused by the loop (which needs tensors) but handy in tests."""
        return float(shift_timesteps(torch.tensor(self.tau_src), 1.0))


class KandinskyPiflowScheduler:
    """pi-Flow schedule and policy parameters for a distilled K6 checkpoint.

    This is a *schedule and configuration* holder, not a stepper: pi-Flow's
    update is a policy rollout over a whole segment and cannot be expressed as
    the duck-typed ``step()`` the Euler scheduler offers. Presenting a ``step()``
    here would invite the generic denoise loop to drive it and silently produce
    the wrong trajectory, so the loop dispatches on this type instead
    (:func:`vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6.piflow_denoise_loop`).

    ``shift`` comes from the checkpoint's ``scheduler/scheduler_config.json``;
    ``n_grid`` is derived from the DiT's own head width rather than configured
    twice (see :meth:`grid_points_from_config`), so a checkpoint whose head and
    scheduler config disagree cannot be loaded into a silently wrong sampler.
    """

    order = 1

    def __init__(
        self,
        *,
        shift: float = 5.0,
        n_grid: int = 10,
        num_policy_substeps: int = 128,
        final_step_size_scale: float = 0.5,
        eps: float = 1e-6,
        device: torch.device | str = "cpu",
    ) -> None:
        if n_grid < _MIN_GRID_POINTS:
            raise ValueError(f"PiFlow n_grid must be >= {_MIN_GRID_POINTS}, got {n_grid}")
        if num_policy_substeps < 1:
            raise ValueError(f"num_policy_substeps must be >= 1, got {num_policy_substeps}")
        self.shift = float(shift)
        self.n_grid = int(n_grid)
        self.num_policy_substeps = int(num_policy_substeps)
        self.eps = float(eps)
        self.final_step_size_scale = max(float(final_step_size_scale), self.eps)
        self.device = device
        self.config = SimpleNamespace(
            shift=self.shift,
            n_grid=self.n_grid,
            num_policy_substeps=self.num_policy_substeps,
            final_step_size_scale=self.final_step_size_scale,
            eps=self.eps,
        )
        self._num_inference_steps = 0
        self.timesteps = torch.empty(0, dtype=torch.float32)

    # -- schedule ------------------------------------------------------
    def segments(self, num_inference_steps: int) -> list[PiflowSegment]:
        """The segment each outer step covers, in raw time, from 1.0 to 0.

        The last segment is ``final_step_size_scale`` as long as the others, so
        the base size is ``1 / (num_steps - (1 - scale))`` and the schedule still
        reaches ``tau = 0``.
        """
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
        base = 1.0 / (float(num_inference_steps) - (1.0 - self.final_step_size_scale))
        out: list[PiflowSegment] = []
        tau_src = 1.0
        for step_index in range(num_inference_steps):
            is_final = step_index == num_inference_steps - 1
            size = base * (self.final_step_size_scale if is_final else 1.0)
            tau_dst = max(tau_src - size, self.eps)
            out.append(PiflowSegment(step_index, tau_src, tau_dst, size, is_final))
            tau_src = tau_dst
        return out

    def set_timesteps(self, num_inference_steps: int, device: torch.device | str | None = None, **_: object) -> None:
        """Model-scale times the DiT is conditioned on, one per outer step."""
        device = device or self.device
        self._num_inference_steps = int(num_inference_steps)
        taus = torch.tensor([s.tau_src for s in self.segments(num_inference_steps)], dtype=torch.float32)
        self.timesteps = (shift_timesteps(taus, self.shift) * 1000.0).to(device)

    def __len__(self) -> int:
        return int(self.timesteps.shape[0])

    # -- construction --------------------------------------------------
    @staticmethod
    def grid_points_from_config(transformer_config: dict) -> int:
        """``n_grid`` implied by the DiT head, or 1 when the head is not a grid.

        A distilled checkpoint widens its output head by ``n_grid``
        (``out_visual_dim = in_visual_dim * n_grid``). Deriving it here means the
        head and the sampler cannot disagree, and the audio head is checked
        against the same factor so a half-converted config is rejected rather
        than sampled wrongly.
        """
        in_visual = int(transformer_config.get("in_visual_dim", 0) or 0)
        out_visual = int(transformer_config.get("out_visual_dim", 0) or 0)
        if in_visual <= 0 or out_visual <= 0 or out_visual % in_visual:
            return 1
        n_grid = out_visual // in_visual
        in_audio = int(transformer_config.get("in_audio_dim", 0) or 0)
        out_audio = int(transformer_config.get("out_audio_dim", 0) or 0)
        if n_grid > 1 and in_audio > 0 and out_audio > 0 and out_audio != in_audio * n_grid:
            raise ValueError(
                "Kandinsky 6 PiFlow head is inconsistent: "
                f"out_visual_dim/in_visual_dim = {n_grid} but out_audio_dim {out_audio} "
                f"!= in_audio_dim {in_audio} * {n_grid}"
            )
        return n_grid

    @classmethod
    def from_configs(
        cls,
        scheduler_config: dict,
        transformer_config: dict,
        *,
        device: torch.device | str = "cpu",
    ) -> KandinskyPiflowScheduler:
        return cls(
            shift=float(scheduler_config.get("shift", 5.0)),
            n_grid=cls.grid_points_from_config(transformer_config),
            num_policy_substeps=int(scheduler_config.get("num_policy_substeps", 128)),
            final_step_size_scale=float(scheduler_config.get("final_step_size_scale", 0.5)),
            eps=float(scheduler_config.get("eps", 1e-6)),
            device=device,
        )


def split_grid_prediction(prediction: torch.Tensor, n_grid: int) -> torch.Tensor:
    """``(..., C*n_grid)`` -> ``(T, n_grid, ..., C)``, grid on axis 1.

    The distilled head emits its ``n_grid`` predictions concatenated on the
    channel axis. :class:`DXPolicy` wants the grid on axis 1, next to the packed
    token axis, which is also the layout the reference wrapper produces.
    """
    if prediction.shape[-1] % n_grid:
        raise ValueError(f"channel dim {prediction.shape[-1]} is not divisible by n_grid {n_grid}")
    per_point = prediction.shape[-1] // n_grid
    reshaped = prediction.view(*prediction.shape[:-1], n_grid, per_point)
    return reshaped.movedim(-2, 1)
