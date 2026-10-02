# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VLLM_OMNI_EARLY_CHUNK: thinker2talker_async_chunk sends chunk 0 at the
thinker's first token, built from the token's embedding, and drops the next
step's chunk that would repeat it. CPU-only, on a stand-in thinker lookup.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.model_executor.stage_input_processors.qwen3_omni as q3
from vllm_omni.model_executor.models.qwen3_omni.serving import thinker_embedding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HIDDEN = 2
_ACCEPT_LAYER = 24
_PROMPT = [151644, 872, 5, 151645]
_FIRST = 9


@pytest.fixture
def thinker(monkeypatch):
    """A registered thinker whose embedding of token t is [t, t]."""
    lookup = thinker_embedding.ThinkerEmbedding(
        embed_input_ids=lambda ids: ids.float().unsqueeze(-1).expand(-1, _HIDDEN),
        device=torch.device("cpu"),
        tts_tokens=torch.tensor([[1, 2, 3]]),
    )
    monkeypatch.setattr(thinker_embedding, "_registered", lookup)
    return lookup


def _transfer_manager():
    return SimpleNamespace(
        put_req_chunk={"r": 0},
        request_payload={},
        _get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=_ACCEPT_LAYER))
        ),
    )


def _request(tokens, mm_features=None):
    return SimpleNamespace(
        external_req_id="r",
        prompt_token_ids=list(_PROMPT),
        all_token_ids=list(tokens),
        mm_features=mm_features,
        resumable=False,
        num_computed_tokens=len(_PROMPT),
        additional_information=None,
    )


def _output(rows):
    return {
        "hidden_states": {
            "layers": {0: torch.ones(rows, _HIDDEN), _ACCEPT_LAYER: torch.full((rows, _HIDDEN), 2.0)},
        },
        "embed": {"tts_bos": torch.zeros(1, _HIDDEN), "tts_eos": None, "tts_pad": None},
    }


def _prefill(transfer_manager, mm_features=None):
    request = _request(_PROMPT + [_FIRST], mm_features)
    return q3.thinker2talker_async_chunk(transfer_manager, _output(len(_PROMPT)), request)


def test_switch_off_holds_chunk0_for_the_first_token(monkeypatch, thinker):
    monkeypatch.delenv("VLLM_OMNI_EARLY_CHUNK", raising=False)
    tm = _transfer_manager()
    assert _prefill(tm) is None
    assert tm.request_payload["r"]["embed"]["prefill"].shape[0] == len(_PROMPT)


def test_chunk0_goes_out_at_the_first_token(monkeypatch, thinker):
    monkeypatch.setenv("VLLM_OMNI_EARLY_CHUNK", "1")
    tm = _transfer_manager()

    payload = _prefill(tm)

    assert payload is not None
    prefill = payload.embed.prefill
    assert prefill.shape == (len(_PROMPT) + 1, _HIDDEN)
    assert prefill[-1].tolist() == [float(_FIRST)] * _HIDDEN
    hidden = payload.hidden_states.output
    assert hidden.shape == (len(_PROMPT) + 1, _HIDDEN)
    assert hidden[-1].tolist() == [0.0] * _HIDDEN
    assert payload.ids.all == _PROMPT + [_FIRST]
    assert "r" not in tm.request_payload


@pytest.mark.parametrize("finished", [False, True])
def test_the_repeated_first_token_chunk_is_dropped_unless_it_finishes(monkeypatch, thinker, finished):
    monkeypatch.setenv("VLLM_OMNI_EARLY_CHUNK", "1")
    tm = _transfer_manager()
    _prefill(tm)
    tm.put_req_chunk["r"] = 1

    payload = q3.thinker2talker_async_chunk(tm, _output(1), _request(_PROMPT + [_FIRST]), is_finished=finished)

    assert (payload is not None) is finished
    # Later chunks take the stock path again.
    payload = q3.thinker2talker_async_chunk(tm, _output(1), _request(_PROMPT + [_FIRST, 4]))
    assert payload is not None and payload.embed.decode.shape == (1, _HIDDEN)


def test_multimodal_prompts_keep_the_stock_path(monkeypatch, thinker):
    monkeypatch.setenv("VLLM_OMNI_EARLY_CHUNK", "1")
    tm = _transfer_manager()
    assert _prefill(tm, mm_features=[object()]) is None
    assert tm.request_payload["r"]["embed"]["prefill"].shape[0] == len(_PROMPT)


def test_without_a_registered_thinker_the_stock_path_runs(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_EARLY_CHUNK", "1")
    monkeypatch.setattr(thinker_embedding, "_registered", None)
    tm = _transfer_manager()
    assert _prefill(tm) is None


def test_register_keeps_the_thinkers_lookup(monkeypatch):
    monkeypatch.setattr(thinker_embedding, "_registered", None)
    thinker = torch.nn.Embedding(10, _HIDDEN)
    thinker.embed_input_ids = thinker.forward
    tts_tokens = torch.tensor([[1, 2, 3]])

    thinker_embedding.register(thinker, tts_tokens)

    registered = thinker_embedding.registered()
    assert registered.device == torch.device("cpu")
    assert registered.tts_tokens is tts_tokens
    assert torch.equal(registered.embed_input_ids(torch.tensor([4])), thinker.weight[4:5])
