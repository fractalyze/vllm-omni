# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VLLM_OMNI_FRAME0: the talker ships frame 0 with its prefill step and its
first decode step reuses the kept embedding sum instead of running the code
predictor. CPU-only, on a stand-in runner and model.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.serving import frame0 as frame0_module
from vllm_omni.model_executor.models.qwen3_omni.serving.frame0 import TalkerFrame0
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HIDDEN = 4
_GROUPS = 3
# Token 7's stand-in embedding (70) plus its hidden state (1), kept from the
# prefill, plus the text-step row (0.5).
_FIRST_DECODE_EMBED = 71.5


class Qwen3OmniMoeForConditionalGeneration:
    """Stands in for the model by its type name, as the runner checks it."""

    talker_mtp_output_key = ("codes", "audio")

    def __init__(self, talker=True):
        self.talker = (
            SimpleNamespace(
                num_code_groups=_GROUPS,
                embed_input_ids=lambda ids: ids.float().unsqueeze(-1).expand(-1, _HIDDEN) * 10,
            )
            if talker
            else None
        )
        self.mtp_calls = []

    def talker_mtp(self, ids, embeds, hidden, text_step, **kwargs):
        self.mtp_calls.append((ids.flatten().tolist(), kwargs))
        codes = ids.view(-1, 1).expand(-1, _GROUPS) + 100
        return embeds + hidden, codes


class _Request:
    def __init__(self, seed=None):
        extra = {"tts_local_seed": seed} if seed is not None else None
        self.sampling_params = SimpleNamespace(extra_args=extra)


def _runner(req_ids, seeds=None, talker=True):
    seeds = seeds or {}
    runner = SimpleNamespace(
        model=Qwen3OmniMoeForConditionalGeneration(talker=talker),
        has_talker_mtp=True,
        requests={r: _Request(seeds.get(r)) for r in req_ids},
        dtype=torch.float32,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(subtalker_sampling_params={"temperature": 0.9})),
        use_async_scheduling=False,
        updates=[],
        text_step=SimpleNamespace(gpu=torch.full((8, _HIDDEN), 0.5)),
    )
    runner._update_intermediate_buffer = lambda req_id, upd: runner.updates.append((req_id, upd))
    runner._talker_mtp_row_generator = lambda req_id, device: OmniGPUModelRunner._talker_mtp_row_generator(
        runner, req_id, device
    )
    runner._subtalker_sampling_kwargs = lambda: OmniGPUModelRunner._subtalker_sampling_kwargs(runner)
    return runner


@pytest.fixture(autouse=True)
def _eager_context(monkeypatch):
    monkeypatch.setattr(
        frame0_module.current_omni_platform,
        "set_forward_context",
        lambda *args, **kwargs: contextlib.nullcontext(),
        raising=False,
    )


def _ship(frame0, runner, req_ids, sampled, valid=None):
    return frame0.ship_with_prefill(
        runner,
        req_ids=req_ids,
        valid_sampled_token_ids=valid if valid is not None else [[t] for t in sampled],
        sampled_token_ids=torch.tensor(sampled).view(-1, 1),
        invalid_req_indices=[],
        sample_hidden_states=torch.ones(len(req_ids), _HIDDEN),
        multimodal_outputs={},
    )


def test_serves_only_the_talker_stage():
    assert frame0_module.serves(_runner(["a"]))
    assert not frame0_module.serves(_runner(["a"], talker=False))


def test_switch_off_builds_nothing(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_FRAME0", raising=False)
    runner = _runner(["a"])
    OmniGPUModelRunner._init_talker_frame0(runner)
    assert getattr(runner, "talker_frame0", None) is None


def test_switch_on_builds_it_for_the_talker(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FRAME0", "1")
    runner = _runner(["a"])
    OmniGPUModelRunner._init_talker_frame0(runner)
    assert isinstance(runner.talker_frame0, TalkerFrame0)


def test_prefill_ships_frame0_once():
    runner = _runner(["a", "b"])
    frame0 = TalkerFrame0()
    shipped = []
    frame0.listeners.append(lambda r, req_id, codes: shipped.append((req_id, codes.tolist())))

    out = _ship(frame0, runner, ["a", "b"], [7, 9], valid=[[7], []])

    audio = out["codes"]["audio"]
    assert audio[0].tolist() == [[107] * _GROUPS]
    assert audio[1].numel() == 0  # "b" sampled nothing this step
    assert shipped == [("a", [[107] * _GROUPS])]
    assert runner.updates == [("a", {"codes": {"audio": audio[0]}})]
    assert runner.model.mtp_calls == [([7], {"do_sample": None, "temperature": 0.9, "top_k": None, "top_p": None})]

    # The request's next sample is a decode step's: frame 0 is already out.
    out = _ship(frame0, runner, ["a"], [8])
    assert out == {}


def test_seeded_requests_ship_one_per_step():
    runner = _runner(["a", "b"], seeds={"a": 1, "b": 2})
    frame0 = TalkerFrame0()

    out = _ship(frame0, runner, ["a", "b"], [7, 9])

    assert [a.numel() for a in out["codes"]["audio"]] == [_GROUPS, 0]
    assert isinstance(runner.model.mtp_calls[0][1]["generator"], torch.Generator)


def _mtp_forward_runner(frame0, req_ids):
    runner = _runner(req_ids)
    runner.talker_frame0 = frame0
    runner.ran_mtp_for = []
    runner._run_talker_mtp = lambda ids, embeds, offsets: runner.ran_mtp_for.append(list(ids))
    return runner


def test_first_decode_step_reuses_the_kept_sum():
    frame0 = TalkerFrame0()
    runner = _mtp_forward_runner(frame0, ["a"])
    _ship(frame0, runner, ["a"], [7])
    runner.updates.clear()
    embeds = torch.zeros(2, _HIDDEN)

    OmniGPUModelRunner._talker_mtp_forward(runner, ["a"], embeds, [1])

    assert runner.ran_mtp_for == []
    assert embeds[1].tolist() == [_FIRST_DECODE_EMBED] * _HIDDEN
    assert embeds[0].tolist() == [0.0] * _HIDDEN
    ((req_id, update),) = runner.updates
    assert req_id == "a" and update["codes"]["audio"].tolist() == [[0] * _GROUPS]

    # Later decode steps run the code predictor again.
    OmniGPUModelRunner._talker_mtp_forward(runner, ["a"], embeds, [1])
    assert runner.ran_mtp_for == [["a"]]


def test_mixed_batch_runs_the_code_predictor_then_overwrites_kept_rows():
    frame0 = TalkerFrame0()
    runner = _mtp_forward_runner(frame0, ["a", "b"])
    _ship(frame0, runner, ["a"], [7])
    embeds = torch.zeros(2, _HIDDEN)

    OmniGPUModelRunner._talker_mtp_forward(runner, ["b", "a"], embeds, [0, 1])

    assert runner.ran_mtp_for == [["b", "a"]]
    assert embeds[1].tolist() == [_FIRST_DECODE_EMBED] * _HIDDEN


def test_switch_off_runs_the_code_predictor():
    runner = _mtp_forward_runner(None, ["a"])
    OmniGPUModelRunner._talker_mtp_forward(runner, ["a"], torch.zeros(1, _HIDDEN), [0])
    assert runner.ran_mtp_for == [["a"]]
