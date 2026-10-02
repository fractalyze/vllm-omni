# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Prefills all but the last position of Qwen3-Omni's talker prompt while
the thinker prefills (VLLM_OMNI_TALKER_PREPREFILL, with
VLLM_OMNI_EARLY_CHUNK).

For a text prompt the talker's prompt is the user parts and nine assistant
rows, all projected from thinker embeddings of prompt tokens, except the last
row, which adds the reply's first token. With the early chunk that token
reaches the talker as chunk 0 the moment it is sampled, and the whole
prefill runs after it. With the switch on, only the last position waits:

- the thinker's scheduler, when it schedules a text prompt's whole prefill,
  looks the prompt and the TTS tokens up in the thinker's embedding table and
  queues them as a pre chunk ahead of the thinker's forward. The pre chunk's
  ids.all equals its ids.prompt: no reply token yet;
- the input processor turns the pre chunk into the talker payload chunk 0
  would carry, without the reply's rows; every later chunk goes to the
  processors behind it as if the pre chunk had not been sent, so the early
  chunk still goes out as their chunk 0;
- the talker's scheduler schedules a waiting request whose payload is a pre
  chunk, and whose prefill has not started, for all but its last placeholder
  token, by vLLM's long-prefill threshold for that step. Its next chunk, the
  first token's, then schedules the last position alone: the step that
  samples frame 0. The talker rebuilds its prompt from the newest payload on
  each prefill step and slices it at the computed tokens, so the two steps
  read the rows a single step would;
- the first token's chunk carries only the token's embedding row (as
  hidden_states.layers[0]) and the ids, since the talker already holds the
  prompt's rows; the talker's chunk adapter rebuilds the full chunk from the
  pre chunk as it receives it;
- the talker model's last-position step computes only the last row (the
  reply's first token's projection plus the codec BOS embedding), since the
  first step already left the rest in the request's buffer.

Each talker step consumes one chunk, so the pre chunk adds one step and the
chunks after it keep theirs. If the first token's chunk lands before the
talker schedules the pre chunk, the newest payload wins and the talker
prefills in one step as before.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.qwen3_omni.serving import thinker_embedding

logger = init_logger(__name__)

# The thinker's processor input key for a pre chunk's rows.
PRE_CHUNK_KEY = "qwen3_omni_pre_chunk"

# The assistant part's thinker rows (<|im_start|>assistant\n, then the
# reply's first token) and the talker rows it becomes
# (_get_talker_assistant_parts).
ASSISTANT_ROWS = 4
ASSISTANT_TALKER_ROWS = 9


def is_pre_payload(info: Any) -> bool:
    """Whether a received payload is a pre chunk: the prompt's ids with no
    reply token after them."""
    if not isinstance(info, Mapping):
        return False
    ids = info.get("ids")
    if not isinstance(ids, Mapping):
        return False
    prompt, all_ids = ids.get("prompt"), ids.get("all")
    return prompt is not None and all_ids is not None and len(all_ids) == len(prompt) > 0


# ------------------------------------------------------------ thinker stage


def pre_chunk_requests(adapter: Any) -> set[str]:
    """External ids of the requests the thinker sent a pre chunk for, until
    they finish."""
    requests = getattr(adapter, "_qwen3_omni_pre_chunk_requests", None)
    if requests is None:
        requests = adapter._qwen3_omni_pre_chunk_requests = set()
    return requests


def pre_rows(thinker: thinker_embedding.ThinkerEmbedding, prompt_ids: list[int], stream) -> tuple:
    """The prompt's and the TTS tokens' thinker embeddings on the host:
    ([prompt, hidden], [3, 1, 1, hidden])."""
    with torch.cuda.stream(stream):
        ids = torch.tensor(prompt_ids, device=thinker.device)
        prefill = thinker.embed_input_ids(ids).to("cpu", non_blocking=True)
        tts = thinker.embed_input_ids(thinker.tts_tokens.to(thinker.device)).to("cpu", non_blocking=True)
    if stream is not None:
        stream.synchronize()
    return prefill, tts.reshape(3, 1, 1, -1)


def _pre_rows_stream(adapter: Any, device: torch.device) -> torch.cuda.Stream | None:
    if device.type != "cuda":
        return None
    stream = getattr(adapter, "_qwen3_omni_pre_chunk_stream", None)
    if stream is None:
        stream = adapter._qwen3_omni_pre_chunk_stream = torch.cuda.Stream(device=device)
    return stream


def send_pre_chunks(scheduler: Any, scheduler_output: Any) -> None:
    """Queues a pre chunk for each text prompt whose whole prefill the
    thinker's scheduler just scheduled."""
    thinker = thinker_embedding.registered()
    adapter = getattr(scheduler, "chunk_transfer_adapter", None)
    if thinker is None or adapter is None or adapter.custom_process_next_stage_input_func is None:
        return
    sent = pre_chunk_requests(adapter)
    for new in scheduler_output.scheduled_new_reqs:
        request = scheduler.requests.get(new.req_id)
        if (
            request is None
            or getattr(request, "mm_features", None)
            or request.resumable
            or request.external_req_id in sent
            or adapter.put_req_chunk[request.external_req_id] != 0
            or scheduler_output.num_scheduled_tokens.get(new.req_id) != request.num_prompt_tokens
        ):
            continue
        try:
            rows = pre_rows(thinker, list(request.prompt_token_ids), _pre_rows_stream(adapter, thinker.device))
        except Exception:
            logger.exception("VLLM_OMNI_TALKER_PREPREFILL: the prompt's embeddings failed")
            return
        sent.add(request.external_req_id)
        adapter.save_async({PRE_CHUNK_KEY: rows}, request)


def compact(payload: Any, hidden_states_struct: type, payload_struct: type) -> Any:
    """The first token's chunk without the prompt's rows the pre chunk sent:
    the token's embedding row as hidden_states.layers[0], and the ids, meta,
    speaker and language. A chunk without prompt rows passes as it is."""
    embed = getattr(payload, "embed", None)
    rows = getattr(embed, "prefill", None)
    if rows is None:
        return payload
    return payload_struct(
        hidden_states=hidden_states_struct(layers={0: rows[-1:]}),
        ids=payload.ids,
        meta=payload.meta,
        speaker=payload.speaker,
        language=payload.language,
    )


# ------------------------------------------------------------- talker stage


def rebuild(pre: dict, payload: dict) -> bool:
    """Fills a compact first-token chunk, in place, with the pre chunk's rows
    as the full chunk would carry them: the prompt's embeddings and the
    token's, and as many hidden rows. Returns whether it was compact."""
    states = payload.get("hidden_states")
    layers = states.get("layers") if isinstance(states, Mapping) else None
    if not layers or payload.get("embed") is not None:
        return False
    row = layers.get(0, layers.get("0"))
    prefill = pre["embed"]["prefill"]
    hidden = pre["hidden_states"]["output"]
    payload["embed"] = dict(pre["embed"], prefill=torch.cat((prefill, row.to(prefill.dtype))))
    payload["hidden_states"] = {"output": torch.cat((hidden, hidden.new_zeros(1, hidden.shape[1])))}
    return True


def merge_received(stash: dict[str, dict], req_id: str, payload: Any) -> None:
    """Keeps a received pre chunk, or fills the compact chunk that follows it
    (the chunk adapter's receive side, before the chunk is committed)."""
    if not isinstance(payload, dict):
        return
    if is_pre_payload(payload):
        stash[req_id] = payload
    elif req_id in stash and rebuild(stash[req_id], payload):
        del stash[req_id]


def is_pre_request(request: Any) -> bool:
    """Whether a waiting request's payload is a pre chunk and its prefill
    has not started."""
    return (
        request.num_computed_tokens == 0
        and request.num_tokens > 1
        and is_pre_payload(getattr(request, "additional_information", None))
    )


def plan_step(waiting: list) -> tuple[int | None, list]:
    """(prefill cap, requests to keep out of this step) for the talker's
    waiting requests. vLLM caps prefills with one scheduler-wide threshold,
    so a pre request takes its all-but-last step only alone in the queue;
    next to others it sits out the step, keeping its ready chunk."""
    pre = [request for request in waiting if is_pre_request(request)]
    if not pre:
        return None, []
    if len(waiting) == 1:
        return pre[0].num_tokens - 1, []
    return None, pre


def last_row_part(
    segments: list[tuple[int, int, int]], rows: int, user: int, assistant: int, system: int
) -> tuple[int, int, int] | None:
    """(start, end, talker rows) for a text prompt's chat `segments` whose
    last part, the assistant's, holds exactly its header and the reply's
    first token: the part's thinker rows, and the talker prompt's length (its
    user parts' first `rows` rows, then the assistant rows). None for any
    other prompt."""
    if not segments or any(role not in (user, assistant, system) for _, _, role in segments):
        return None
    start, end, role = segments[-1]
    if role != assistant or end - start != ASSISTANT_ROWS:
        return None
    talker_rows = sum(max(0, min(e, rows) - s) for s, e, r in segments[:-1] if r == user)
    return start, end, talker_rows + ASSISTANT_TALKER_ROWS
