# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Ships Qwen3-Omni's first audio frame one talker step sooner
(VLLM_OMNI_FRAME0).

A talker step samples a frame's first codebook token. The runner runs the
code predictor for the frame's other 15 only at the start of the next step,
before that step's forward, and ships the frame after that step samples. The
first frame therefore waits out the talker's first decode step: its forward,
its sampling and the scheduling around it.

With the switch on, the talker's runner:

- after a request's first sample (its prefill), runs the code predictor at
  once on that token and its hidden state. The 16 codes go out with the
  prefill step's output, and the embedding sum the first decode step feeds
  the talker is kept;
- in the first decode step, adds the kept sum to its text step, as the code
  predictor's output would have been added, and ships no codes, since frame 0
  is already out.

Frame 0's codes are the same draws as before: the code predictor still
samples frame 0 first, from the request's own seeded generator when it has a
tts_local_seed, so every later frame draws what it drew before.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# A request whose frame 0 is out and consumed: it must never ship again.
_USED = object()

# Called as listener(runner, req_id, codes [1, code groups]) for each frame 0
# shipped, once it is in the step's output.
Frame0Listener = Callable[[Any, str, torch.Tensor], None]


def _unwrapped_model(runner) -> Any:
    # With full CUDA graphs the runner holds the model in vLLM's
    # CUDAGraphWrapper, which forwards attributes but not its type.
    model = getattr(runner, "model", None)
    return model.unwrap() if hasattr(model, "unwrap") else model


def serves(runner) -> bool:
    """Whether `runner` is Qwen3-Omni's talker stage."""
    model = getattr(runner, "model", None)
    return (
        getattr(runner, "has_talker_mtp", False)
        and getattr(model, "talker", None) is not None
        and type(_unwrapped_model(runner)).__name__ == "Qwen3OmniMoeForConditionalGeneration"
    )


def _per_request_audio(multimodal_outputs: Any, count: int, device: torch.device) -> list[torch.Tensor]:
    """The step's codes as one entry per request, as the runner ships them."""
    codes = multimodal_outputs.get("codes") if isinstance(multimodal_outputs, dict) else None
    audio = codes.get("audio") if isinstance(codes, dict) else None
    if isinstance(audio, list) and len(audio) == count:
        return list(audio)
    if isinstance(audio, torch.Tensor) and audio.ndim > 0 and audio.shape[0] == count:
        return [audio[i : i + 1] for i in range(count)]
    return [torch.empty(0, dtype=torch.long, device=device) for _ in range(count)]


class TalkerFrame0:
    """Frame 0 of each talker request, shipped with its prefill step.

    Owned by the talker stage's runner; `listeners` run on each frame 0
    shipped, in order.
    """

    def __init__(self) -> None:
        # Per request: frame 0's embedding sum while the first decode step
        # still needs it, then _USED so frame 0 never ships twice.
        self._state: dict[str, Any] = {}
        self.listeners: list[Frame0Listener] = []
        self._logged = False

    def _live_state(self, runner) -> dict[str, Any]:
        for req_id in [r for r in self._state if r not in runner.requests]:
            del self._state[req_id]
        return self._state

    def ship_with_prefill(
        self,
        runner,
        *,
        req_ids: list[str],
        valid_sampled_token_ids: list[list[int]],
        sampled_token_ids: torch.Tensor,
        invalid_req_indices: list[int],
        sample_hidden_states: torch.Tensor,
        multimodal_outputs: Any,
    ) -> Any:
        """`multimodal_outputs` with frame 0's codes for each request that
        sampled its first token this step."""
        state = self._live_state(runner)
        invalid = set(invalid_req_indices)
        is_async = bool(getattr(runner, "use_async_scheduling", False))
        rows = []
        for idx, req_id in enumerate(req_ids):
            if req_id in state:
                continue
            if is_async:
                sampled = idx not in invalid
            else:
                sampled = bool(idx < len(valid_sampled_token_ids) and valid_sampled_token_ids[idx])
            if sampled:
                rows.append(idx)
        if not rows:
            return multimodal_outputs
        device = sample_hidden_states.device
        generators = [runner._talker_mtp_row_generator(req_ids[i], device) for i in rows]
        if len(rows) > 1 and any(g is not None for g in generators):
            # One generator per call: ship the first request now and leave the
            # others to the runner's own path in their first decode step.
            rows = rows[:1]
            generators = generators[:1]
        mtp_kwargs = runner._subtalker_sampling_kwargs()
        if generators[0] is not None:
            mtp_kwargs["generator"] = generators[0]

        index = torch.tensor(rows, dtype=torch.long, device=device)
        ids = sampled_token_ids.index_select(0, index).reshape(-1).to(torch.long)
        model = runner.model
        embeds = model.talker.embed_input_ids(ids).to(runner.dtype)
        hidden = sample_hidden_states.index_select(0, index).to(runner.dtype)
        with current_omni_platform.set_forward_context(
            None, runner.vllm_config, cudagraph_runtime_mode=CUDAGraphMode.NONE, batch_descriptor=None
        ):
            summed, codes = model.talker_mtp(ids.view(-1, 1), embeds, hidden, torch.zeros_like(embeds), **mtp_kwargs)
        audio = _per_request_audio(multimodal_outputs, len(req_ids), device)
        for row, idx in enumerate(rows):
            req_id = req_ids[idx]
            frame = codes[row : row + 1].detach().clone()
            audio[idx] = frame
            state[req_id] = summed[row : row + 1].detach().clone()
            runner._update_intermediate_buffer(req_id, {"codes": {"audio": frame}})
        for idx in rows:
            for listener in self.listeners:
                listener(runner, req_ids[idx], audio[idx])
        if not self._logged:
            self._logged = True
            logger.info("VLLM_OMNI_FRAME0: frame 0 ships with the talker's prefill step")
        merged = dict(multimodal_outputs) if isinstance(multimodal_outputs, dict) else {}
        merged_codes = dict(merged.get("codes") or {})
        merged_codes["audio"] = audio
        merged["codes"] = merged_codes
        return merged

    def rows_with_frame0(self, runner, decode_req_ids: list[str]) -> list[int]:
        """Indices into `decode_req_ids` of first decode steps whose frame 0
        is out: the code predictor need not run for them."""
        state = self._live_state(runner)
        return [i for i, req_id in enumerate(decode_req_ids) if isinstance(state.get(req_id), torch.Tensor)]

    def feed_first_decode_step(
        self,
        runner,
        decode_req_ids: list[str],
        rows: list[int],
        inputs_embeds: torch.Tensor,
        start_offsets: list[int] | None,
    ) -> None:
        """Writes each kept frame-0 sum plus its text step into the first
        decode step's input embedding, for the `rows` rows_with_frame0
        found."""
        if start_offsets is None:
            index_of = runner.input_batch.req_id_to_index
            start_offsets = [int(runner.query_start_loc.cpu[index_of[r]]) for r in decode_req_ids]
        out_key = getattr(runner.model, "talker_mtp_output_key", ("codes", "audio"))
        for i in rows:
            req_id = decode_req_ids[i]
            step = self._state[req_id] + runner.text_step.gpu[i : i + 1]
            inputs_embeds[start_offsets[i] : start_offsets[i] + 1].copy_(step)
            self._state[req_id] = _USED
            no_codes = torch.zeros(
                1, runner.model.talker.num_code_groups, dtype=torch.long, device=inputs_embeds.device
            )
            runner._update_intermediate_buffer(req_id, {out_key[0]: {out_key[1]: no_codes}})
