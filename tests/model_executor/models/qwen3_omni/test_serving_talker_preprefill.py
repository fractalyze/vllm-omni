# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VLLM_OMNI_TALKER_PREPREFILL: the thinker sends a pre chunk of the prompt's
rows, the talker prefills all but the last position from it, and the first
token's chunk carries only the token's row. CPU-only, on stand-ins.
"""

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_omni.model_executor.stage_input_processors.qwen3_omni as q3
from tests.model_executor.models.qwen3_omni.talker_stand_in import (
    ASSISTANT,
    IM_START,
    SYSTEM,
    THINKER_HIDDEN,
    USER,
    talker_model,
)
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.data_entry_keys import HiddenStatesStruct, IdsStruct, OmniPayloadStruct
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import chat_segments
from vllm_omni.model_executor.models.qwen3_omni.serving import talker_preprefill, thinker_embedding
from vllm_omni.model_executor.models.qwen3_omni.serving.talker_preprefill import (
    PRE_CHUNK_KEY,
    compact,
    is_pre_payload,
    last_row_part,
    merge_received,
    plan_step,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# system part, user part (5 rows), then the assistant header.
PROMPT = [IM_START, SYSTEM, 11, 12, IM_START, USER, 21, 22, 23, IM_START, ASSISTANT, 198]
FIRST_TOKEN = 31
# The user part's 5 rows and the assistant's 9.
TALKER_ROWS = 14


def test_is_pre_payload():
    assert is_pre_payload({"ids": {"prompt": [1, 2], "all": [1, 2]}})
    assert not is_pre_payload({"ids": {"prompt": [1, 2], "all": [1, 2, 3]}})
    assert not is_pre_payload({"ids": {"prompt": [], "all": []}})
    assert not is_pre_payload(None)


def test_last_row_part():
    segments = chat_segments(PROMPT, len(PROMPT) + 1, IM_START)
    assert last_row_part(segments, len(PROMPT) + 1, USER, ASSISTANT, SYSTEM) == (9, 13, TALKER_ROWS)
    # A reply further along has more than the header and one token.
    segments = chat_segments(PROMPT, len(PROMPT) + 2, IM_START)
    assert last_row_part(segments, len(PROMPT) + 2, USER, ASSISTANT, SYSTEM) is None


def _waiting(computed=0, tokens=TALKER_ROWS, pre=True):
    ids = {"prompt": PROMPT, "all": PROMPT if pre else PROMPT + [FIRST_TOKEN]}
    return SimpleNamespace(num_computed_tokens=computed, num_tokens=tokens, additional_information={"ids": ids})


def test_plan_step():
    pre, other = _waiting(), _waiting(pre=False)
    assert plan_step([pre]) == (TALKER_ROWS - 1, [])
    assert plan_step([pre, other]) == (None, [pre])
    assert plan_step([other]) == (None, [])
    assert plan_step([_waiting(computed=3)]) == (None, [])


def _full_chunk(rows):
    return OmniPayloadStruct(
        embed=q3.EmbeddingsStruct(prefill=torch.arange(rows * 2.0).view(rows, 2), tts_bos=torch.zeros(1, 2)),
        hidden_states=HiddenStatesStruct(output=torch.ones(rows, 2)),
        ids=IdsStruct(all=PROMPT + [FIRST_TOKEN], prompt=PROMPT),
    )


def test_compact_and_rebuild_round_trip():
    rows = len(PROMPT) + 1
    full = _full_chunk(rows)
    small = compact(full, HiddenStatesStruct, OmniPayloadStruct)
    assert small.embed is None and small.hidden_states.layers[0].shape == (1, 2)

    pre = {
        "embed": {"prefill": full.embed.prefill[:-1], "tts_bos": full.embed.tts_bos},
        "hidden_states": {"output": torch.zeros(rows - 1, 2)},
        "ids": {"prompt": PROMPT, "all": PROMPT},
    }
    stash: dict = {}
    merge_received(stash, "r", pre)
    received: dict[str, Any] = {
        "hidden_states": {"layers": {0: small.hidden_states.layers[0]}},
        "ids": {"all": PROMPT + [1]},
    }
    merge_received(stash, "r", received)

    assert stash == {}
    assert torch.equal(received["embed"]["prefill"], full.embed.prefill)
    assert received["embed"]["tts_bos"] is full.embed.tts_bos
    assert received["hidden_states"]["output"].shape == (rows, 2)


def _transfer_manager():
    return SimpleNamespace(
        put_req_chunk=defaultdict(int),
        request_payload={},
        _get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=24))
        ),
    )


def _request(tokens):
    return SimpleNamespace(
        external_req_id="r",
        prompt_token_ids=list(PROMPT),
        all_token_ids=list(tokens),
        mm_features=None,
        resumable=False,
        num_computed_tokens=len(PROMPT),
        additional_information=None,
    )


def _prefill_output(rows):
    return {
        "hidden_states": {"layers": {0: torch.ones(rows, 2), 24: torch.full((rows, 2), 2.0)}},
        "embed": {"tts_bos": torch.zeros(1, 2), "tts_eos": None, "tts_pad": None},
    }


@pytest.fixture
def switches(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_TALKER_PREPREFILL", "1")
    monkeypatch.setenv("VLLM_OMNI_EARLY_CHUNK", "1")
    lookup = thinker_embedding.ThinkerEmbedding(
        embed_input_ids=lambda ids: ids.float().unsqueeze(-1).expand(*ids.shape, 2),
        device=torch.device("cpu"),
        tts_tokens=torch.tensor([[1, 2, 3]]),
    )
    monkeypatch.setattr(thinker_embedding, "_registered", lookup)


def test_the_processor_sends_a_pre_chunk_then_a_compact_first_chunk(switches):
    tm = _transfer_manager()
    request = _request(PROMPT + [FIRST_TOKEN])
    rows = talker_preprefill.pre_rows(thinker_embedding.registered(), PROMPT, None)
    talker_preprefill.pre_chunk_requests(tm).add("r")

    pre = q3.thinker2talker_async_chunk(tm, {PRE_CHUNK_KEY: rows}, request)
    assert pre.ids.all == PROMPT and pre.embed.prefill.shape == (len(PROMPT), 2)
    assert pre.embed.tts_bos.shape == (1, 1, 2)
    tm.put_req_chunk["r"] += 1

    first = q3.thinker2talker_async_chunk(tm, _prefill_output(len(PROMPT)), request)

    # The early chunk went out as the processors' chunk 0, compacted.
    assert first.embed is None
    assert first.hidden_states.layers[0].tolist() == [[float(FIRST_TOKEN)] * 2]
    assert first.ids.all == PROMPT + [FIRST_TOKEN]
    assert tm.put_req_chunk["r"] == 1


def test_finishing_forgets_the_pre_chunk(switches):
    tm = _transfer_manager()
    talker_preprefill.pre_chunk_requests(tm).add("r")
    tm.put_req_chunk["r"] = 3
    q3.thinker2talker_async_chunk(tm, _prefill_output(1), _request(PROMPT + [FIRST_TOKEN, 5, 6]), is_finished=True)
    assert "r" not in talker_preprefill.pre_chunk_requests(tm)


def test_send_pre_chunks_queues_each_whole_text_prefill(switches):
    saved = []
    adapter = SimpleNamespace(
        custom_process_next_stage_input_func=object(),
        put_req_chunk=defaultdict(int),
        save_async=lambda output, request: saved.append((output, request.external_req_id)),
    )
    request = SimpleNamespace(
        external_req_id="r",
        mm_features=None,
        resumable=False,
        num_prompt_tokens=len(PROMPT),
        prompt_token_ids=PROMPT,
    )
    scheduler = SimpleNamespace(chunk_transfer_adapter=adapter, requests={"r0": request})
    output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="r0")], num_scheduled_tokens={"r0": len(PROMPT)}
    )

    talker_preprefill.send_pre_chunks(scheduler, output)
    talker_preprefill.send_pre_chunks(scheduler, output)

    ((chunk, req_id),) = saved
    prefill, tts = chunk[PRE_CHUNK_KEY]
    assert req_id == "r" and prefill.shape == (len(PROMPT), 2) and tts.shape == (3, 1, 1, 2)


def test_send_pre_chunks_skips_partial_prefills(switches):
    saved = []
    adapter = SimpleNamespace(
        custom_process_next_stage_input_func=object(),
        put_req_chunk=defaultdict(int),
        save_async=lambda output, request: saved.append(output),
    )
    request = SimpleNamespace(
        external_req_id="r", mm_features=None, resumable=False, num_prompt_tokens=len(PROMPT), prompt_token_ids=PROMPT
    )
    scheduler = SimpleNamespace(chunk_transfer_adapter=adapter, requests={"r0": request})
    output = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id="r0")], num_scheduled_tokens={"r0": 4})
    talker_preprefill.send_pre_chunks(scheduler, output)
    assert saved == []


class _Queue(list):
    def remove_requests(self, requests):
        for request in requests:
            self.remove(request)

    def prepend_request(self, request):
        self.insert(0, request)


def _scheduler(waiting):
    scheduler = object.__new__(OmniARScheduler)
    scheduler.waiting = _Queue(waiting)
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    scheduler.chunk_transfer_adapter = None
    scheduler.seen = []

    def schedule(throttle_prefills):
        scheduler.seen.append((list(scheduler.waiting), scheduler.scheduler_config.long_prefill_token_threshold))
        return SimpleNamespace(scheduled_new_reqs=[])

    scheduler._schedule = schedule
    return scheduler


def test_a_lone_pre_request_prefills_all_but_its_last_position():
    pre = _waiting()
    scheduler = _scheduler([pre])
    scheduler._schedule_with_preprefill(False)
    assert scheduler.seen == [([pre], TALKER_ROWS - 1)]
    assert scheduler.scheduler_config.long_prefill_token_threshold == 0


def test_a_pre_request_beside_others_sits_out_the_step():
    pre, other = _waiting(), _waiting(pre=False)
    scheduler = _scheduler([pre, other])
    scheduler._schedule_with_preprefill(False)
    assert scheduler.seen == [([other], 0)]
    assert list(scheduler.waiting) == [pre, other]


def _talker_payload(processed):
    rows = len(PROMPT) + 1
    torch.manual_seed(2)
    return {
        "embed": {
            "prefill": torch.randn(rows, THINKER_HIDDEN).to(torch.bfloat16),
            "tts_bos": torch.randn(1, 1, THINKER_HIDDEN).to(torch.bfloat16),
            "tts_eos": torch.randn(1, 1, THINKER_HIDDEN).to(torch.bfloat16),
            "tts_pad": torch.randn(1, 1, THINKER_HIDDEN).to(torch.bfloat16),
        },
        "hidden_states": {"output": torch.zeros(rows, THINKER_HIDDEN).to(torch.bfloat16)},
        "ids": {"all": PROMPT + [FIRST_TOKEN], "prompt": PROMPT},
        "meta": {"num_processed_tokens": processed},
    }


def test_the_last_position_matches_the_full_prefill():
    payload = _talker_payload(TALKER_ROWS - 1)
    embeds = torch.zeros(1, 6)
    full_ids, full_embeds, _ = talker_model().talker_preprocess_prefill(None, embeds, payload)

    ids, last, update = talker_model(preprefill=True).talker_preprocess_prefill(None, embeds, payload)

    assert torch.equal(ids, full_ids)
    assert last.dtype == full_embeds.dtype and torch.equal(last, full_embeds)
    assert update["meta"] == {"prefill_consumed_text_tokens": 1}


def test_other_steps_take_the_full_prefill():
    model = talker_model(preprefill=True)
    assert model._talker_prefill_last_row(torch.zeros(1, 6), _talker_payload(0)) is None
    assert model._talker_prefill_last_row(torch.zeros(2, 6), _talker_payload(TALKER_ROWS - 1)) is None
