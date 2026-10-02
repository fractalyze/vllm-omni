# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Qwen3-Omni talker and code-predictor megakernels on random weights at
their shapes.

The talker's decode steps and the code predictor's teacher-forced frames are
held to an fp32 PyTorch model, within FACTOR × the error of the same model in
bf16, and the code predictor's draws to the wrapper's "stored" sampler. The
talker's and the code predictor's forwards are then run with the megakernel
switched on, on stand-ins holding those weights as vLLM does after loading:
an eligible call must give exactly what the kernel gives when launched
directly, and every other talker step must reach the stock forward.

The kernels are built for sm_120a, so these tests need an RTX 5090-class GPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.platforms import current_omni_platform


def _is_sm120() -> bool:
    return current_omni_platform.is_cuda() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = [
    pytest.mark.local_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not _is_sm120(), reason="the talker megakernels are built for sm_120a"),
]

if _is_sm120():
    import vllm_omni.model_executor.models.qwen3_omni.megakernel.talker as megakernel_talker
    from vllm_omni.model_executor.models.qwen3_omni.megakernel import code_predictor as cp
    from vllm_omni.model_executor.models.qwen3_omni.megakernel import talker_decode as td
    from vllm_omni.model_executor.models.qwen3_omni.megakernel.talker import (
        CodePredictorMegakernel,
        TalkerMegakernel,
    )
    from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_attention import PagedCache, cos_sin_table
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_code_predictor_mtp import (
        Qwen3OmniMoeTalkerCodePredictor,
    )
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_talker import (
        Qwen3OmniMoeTalkerForConditionalGeneration,
    )

# A kernel's error may exceed bf16's own by this factor.
FACTOR = 1.25
TALKER_LAYERS, CP_LAYERS = 2, 5
BLOCKS, BLOCK_SIZE = 4, 16
EPS, ROPE_THETA = 1e-6, 1_000_000.0
TOP_K, TOP_P = 50, 0.8
_UNIFORM_EPS = 1e-20


def _rel_l2(x: torch.Tensor, ref: torch.Tensor) -> float:
    x, ref = x.float(), ref.float()
    return float((x - ref).norm() / ref.norm())


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _linear(gen, *shape: int) -> torch.Tensor:
    return (torch.randn(*shape, generator=gen, device="cuda") / shape[-1] ** 0.5).bfloat16()


def _norm(gen, n: int) -> torch.Tensor:
    return (1 + 0.1 * torch.randn(n, generator=gen, device="cuda")).bfloat16()


def _uniforms(*shape: int, gen: torch.Generator) -> torch.Tensor:
    return torch.rand(*shape, generator=gen, device="cuda").clamp(_UNIFORM_EPS, 1 - _UNIFORM_EPS)


# ---------------------------------------------------------------------------
#  Talker decode
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def talker_layers():
    gen = _generator(0)
    d = td.DIM
    return [
        td.LayerWeights(
            norm=_norm(gen, d),
            wqkv=_linear(gen, td.Q_DIM + 2 * td.KV_DIM, d),
            q_norm=_norm(gen, td.HEAD_DIM),
            k_norm=_norm(gen, td.HEAD_DIM),
            wo=_linear(gen, d, td.Q_DIM),
            moe_norm=_norm(gen, d),
            # Scaled up so the chosen experts' router logits stand clear of
            # bf16 rounding, as a trained router's do; random ones tie.
            router=(4 * _linear(gen, td.EXPERTS, d).float()).bfloat16(),
            w13=_linear(gen, td.EXPERTS, 2 * td.EXPERT_FFN, d),
            w2=_linear(gen, td.EXPERTS, d, td.EXPERT_FFN),
            shared_w13=_linear(gen, 2 * td.SHARED_FFN, d),
            shared_w2=_linear(gen, d, td.SHARED_FFN),
            shared_gate=_linear(gen, 1, d),
        )
        for _ in range(TALKER_LAYERS)
    ]


@pytest.fixture(scope="module")
def talker_norm():
    return _norm(_generator(1), td.DIM)


@pytest.fixture(scope="module")
def cos_sin():
    return cos_sin_table(256)


def _talker_inputs(pos: int):
    gen = _generator(100 + pos)
    embedding = torch.randn(td.DIM, generator=gen, device="cuda").bfloat16()
    return embedding, torch.tensor([pos, pos + 3, pos + 5], dtype=torch.int64, device="cuda")


@pytest.mark.parametrize("ctas", [96, 128])
def test_talker_steps_within_bf16_budget(talker_layers, talker_norm, cos_sin, ctas):
    caches = td.new_caches(TALKER_LAYERS, BLOCKS * BLOCK_SIZE, BLOCK_SIZE)
    decoder = td.TalkerDecoder(talker_layers, caches, talker_norm, cos_sin, ctas=ctas, splits=2)
    for pos in range(12):
        embedding, positions = _talker_inputs(pos)
        got = decoder.step(embedding, pos, positions).float()
        # The kernel wrote this position; the fp32 reference writes it last,
        # so all three read the same history.
        budget = td.reference_step(
            talker_layers, talker_norm, caches, embedding, pos, positions, cos_sin, torch.bfloat16
        )
        truth = td.reference_step(talker_layers, talker_norm, caches, embedding, pos, positions, cos_sin, torch.float32)
        error, allowed = _rel_l2(got, truth), _rel_l2(budget, truth)
        assert error <= FACTOR * allowed, f"pos {pos}: error {error:.3g} exceeds {FACTOR} × bf16's {allowed:.3g}"


class _StockTalkerModel:
    """The talker's language model: the layers the kernel reads, and a stock
    forward that records each call."""

    def __init__(self, layers, norm):
        self.layers = layers
        self.norm = norm
        self.calls: list[int] = []

    def __call__(self, input_ids, positions, intermediate_tensors, inputs_embeds=None, **kwargs):
        self.calls.append(inputs_embeds.shape[0])
        return torch.zeros_like(inputs_embeds), None


def _talker_stand_in(layers, final_norm, cos_sin):
    """A talker holding `layers` as vLLM's Qwen3MoeForCausalLM does after
    loading, with the megakernel on."""
    modules = []
    for w in layers:
        kv = torch.zeros(BLOCKS, td.KV_HEADS, BLOCK_SIZE, 2 * td.HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        modules.append(
            SimpleNamespace(
                input_layernorm=SimpleNamespace(weight=w.norm),
                post_attention_layernorm=SimpleNamespace(weight=w.moe_norm),
                self_attn=SimpleNamespace(
                    qkv_proj=SimpleNamespace(weight=w.wqkv),
                    o_proj=SimpleNamespace(weight=w.wo),
                    q_norm=SimpleNamespace(weight=w.q_norm),
                    k_norm=SimpleNamespace(weight=w.k_norm),
                    rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin.float()),
                    attn=SimpleNamespace(kv_cache=kv),
                ),
                mlp=SimpleNamespace(
                    gate=SimpleNamespace(weight=w.router),
                    experts=SimpleNamespace(routed_experts=SimpleNamespace(w13_weight=w.w13, w2_weight=w.w2)),
                    shared_expert=SimpleNamespace(
                        gate_up_proj=SimpleNamespace(weight=w.shared_w13),
                        down_proj=SimpleNamespace(weight=w.shared_w2),
                    ),
                    shared_expert_gate=SimpleNamespace(weight=w.shared_gate),
                ),
            )
        )
    talker = object.__new__(Qwen3OmniMoeTalkerForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.language_model = SimpleNamespace(model=_StockTalkerModel(modules, SimpleNamespace(weight=final_norm)))
    talker.megakernel = TalkerMegakernel(ctas=96)
    return talker


def _table():
    """The one request's blocks reversed, so positions cross blocks out of order."""
    return torch.arange(BLOCKS, dtype=torch.int32, device="cuda").flip(0).view(1, BLOCKS)


def _decode_metadata(pos: int, table: torch.Tensor):
    slot = int(table[0, pos // BLOCK_SIZE]) * BLOCK_SIZE + pos % BLOCK_SIZE
    return SimpleNamespace(
        block_table=table,
        seq_lens=torch.tensor([pos + 1], dtype=torch.int32, device="cuda"),
        slot_mapping=torch.tensor([slot], dtype=torch.int64, device="cuda"),
    )


def test_talker_decode_step_runs_on_the_kernel(talker_layers, talker_norm, cos_sin, monkeypatch):
    talker = _talker_stand_in(talker_layers, talker_norm, cos_sin)
    pos = 20
    metadata = _decode_metadata(pos, _table())
    monkeypatch.setattr(megakernel_talker, "_attn_metadata", lambda: metadata)
    embeds = torch.randn(1, td.DIM, device="cuda").bfloat16()
    # vLLM's M-RoPE buffer is [3, max_tokens + 1], so rows are not adjacent.
    rows = torch.zeros(3, 8, dtype=torch.int64, device="cuda")
    rows[:, 0] = torch.tensor([pos, pos + 1, pos + 2])

    hidden = talker.forward(None, rows[:, :1], inputs_embeds=embeds)
    current_omni_platform.synchronize()
    assert talker.language_model.model.calls == []

    # The same step on a decoder over caches of its own.
    caches = td.new_caches(TALKER_LAYERS, BLOCKS * BLOCK_SIZE, BLOCK_SIZE)
    caches = [PagedCache(c.key, c.value, metadata.block_table[0]) for c in caches]
    decoder = td.TalkerDecoder(talker_layers, caches, talker_norm, cos_sin, ctas=96, splits=2)
    want = torch.empty(td.DIM, dtype=torch.bfloat16, device="cuda")
    decoder.launch(embeds, metadata.seq_lens, metadata.slot_mapping, rows[:, :1], final_hidden=want)
    current_omni_platform.synchronize()
    assert torch.equal(hidden.view(-1), want)
    for layer, expected in zip(talker.language_model.model.layers, caches):
        key, value = layer.self_attn.attn.kv_cache.transpose(1, 2).split(td.HEAD_DIM, dim=-1)
        got = PagedCache(key, value, metadata.block_table[0])
        got_k, _ = got.gather(pos + 1)
        want_k, _ = expected.gather(pos + 1)
        assert torch.equal(got_k[pos], want_k[pos])
        assert got_k[pos].abs().sum() > 0


def test_talker_other_steps_keep_the_stock_forward(talker_layers, talker_norm, cos_sin, monkeypatch):
    talker = _talker_stand_in(talker_layers, talker_norm, cos_sin)
    table = _table()
    monkeypatch.setattr(megakernel_talker, "_attn_metadata", lambda: _decode_metadata(0, table))
    # A prompt of several tokens.
    talker.forward(None, torch.zeros(3, 5, dtype=torch.long, device="cuda"), inputs_embeds=torch.zeros(5, td.DIM))
    # One token without M-RoPE positions.
    talker.forward(None, torch.zeros(1, dtype=torch.long, device="cuda"), inputs_embeds=torch.zeros(1, td.DIM))
    assert talker.language_model.model.calls == [5, 1]
    assert talker.megakernel._decoder is None


def test_talker_refuses_quantized_experts(talker_layers, talker_norm, cos_sin, monkeypatch):
    talker = _talker_stand_in(talker_layers, talker_norm, cos_sin)
    experts = talker.language_model.model.layers[0].mlp.experts.routed_experts
    experts.w13_weight = experts.w13_weight.view(torch.uint8)
    table = _table()
    monkeypatch.setattr(megakernel_talker, "_attn_metadata", lambda: _decode_metadata(0, table))
    with pytest.raises(RuntimeError, match="bf16 experts"):
        talker.forward(
            None, torch.zeros(3, 1, dtype=torch.long, device="cuda"), inputs_embeds=torch.zeros(1, td.DIM).cuda()
        )


# ---------------------------------------------------------------------------
#  Code predictor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cp_weights():
    gen = _generator(2)
    d = cp.DIM
    layers = [
        cp.Layer(
            wqkv=_linear(gen, cp.Q_DIM + 2 * cp.KV_DIM, d),
            wo=_linear(gen, d, cp.Q_DIM),
            w13=_linear(gen, 2 * cp.FFN, d),
            w2=_linear(gen, d, cp.FFN),
            input_layernorm=_norm(gen, d),
            post_attention_layernorm=_norm(gen, d),
            q_norm=_norm(gen, cp.HEAD_DIM),
            k_norm=_norm(gen, cp.HEAD_DIM),
        )
        for _ in range(CP_LAYERS)
    ]
    embeddings = torch.randn(cp.HEADS, cp.VOCAB, d, generator=gen, device="cuda").bfloat16()
    # Scaled up so the logits spread like a trained head's and top-p cuts.
    heads = (4 * _linear(gen, cp.HEADS, cp.VOCAB, d).float()).bfloat16()
    return cp.Weights(layers, _norm(gen, d), heads, embeddings, EPS, ROPE_THETA)


def _frames(weights, count: int, seed: int):
    """(talker_hidden, code0_embed, uniforms, forced codes) for `count` frames."""
    gen = _generator(seed)
    hidden = torch.randn(count, cp.DIM, generator=gen, device="cuda").bfloat16()
    code0 = torch.randint(0, cp.VOCAB, (count,), generator=gen, device="cuda")
    code0_embed = weights.codec_embeddings[0][code0].contiguous()
    uniforms = _uniforms(count, cp.HEADS, cp.VOCAB, gen=gen)
    forced = torch.randint(0, cp.VOCAB, (count, cp.HEADS), generator=gen, device="cuda")
    return hidden, code0_embed, uniforms, forced


@pytest.mark.parametrize("ctas", [None, 64])
def test_code_predictor_frames_within_bf16_budget(cp_weights, ctas):
    predictor = cp.CodePredictor(cp_weights, TOP_K, TOP_P, ctas=ctas)
    hidden, code0_embed, uniforms, forced = _frames(cp_weights, count=3, seed=1)
    codes, logits = predictor.run(hidden, code0_embed, uniforms, forced_codes=forced)
    fp32 = cp_weights.to(torch.float32)
    for i in range(hidden.shape[0]):
        inputs = (hidden[i], code0_embed[i], uniforms[i], TOP_K, TOP_P, forced[i])
        _, truth = cp.reference(fp32, *inputs)
        _, budget = cp.reference(cp_weights, *inputs)
        error, allowed = _rel_l2(logits[i], truth), _rel_l2(budget, truth)
        assert error <= FACTOR * allowed, f"frame {i}: error {error:.3g} exceeds {FACTOR} × bf16's {allowed:.3g}"
        # The kernel draws from its own logits as the wrapper draws from bf16 logits.
        drawn = cp.reference_sample(logits[i].bfloat16().float(), uniforms[i], TOP_K, TOP_P)
        assert torch.equal(codes[i], drawn)


def test_code_predictor_frames_are_independent_and_deterministic(cp_weights):
    predictor = cp.CodePredictor(cp_weights, TOP_K, TOP_P)
    hidden, code0_embed, uniforms, _ = _frames(cp_weights, count=3, seed=3)
    codes, logits = predictor.run(hidden, code0_embed, uniforms)
    again_codes, again_logits = predictor.run(hidden, code0_embed, uniforms)
    alone_codes, alone_logits = predictor.run(hidden[1:2], code0_embed[1:2], uniforms[1:2])
    assert torch.equal(again_logits, logits) and torch.equal(again_codes, codes)
    assert torch.equal(alone_logits[0], logits[1]) and torch.equal(alone_codes[0], codes[1])


def _code_predictor_stand_in(weights, ctas=None):
    """A Qwen3-Omni code predictor holding `weights` as CodePredictorWrapper
    does after loading (fused qkv_proj and gate_up_proj, one head and one
    codec embedding per code), with the megakernel on and built."""
    layers = [
        SimpleNamespace(
            input_layernorm=SimpleNamespace(weight=w.input_layernorm),
            post_attention_layernorm=SimpleNamespace(weight=w.post_attention_layernorm),
            self_attn=SimpleNamespace(
                qkv_proj=SimpleNamespace(weight=w.wqkv),
                o_proj=SimpleNamespace(weight=w.wo),
                q_norm=SimpleNamespace(weight=w.q_norm),
                k_norm=SimpleNamespace(weight=w.k_norm),
            ),
            mlp=SimpleNamespace(
                gate_up_proj=SimpleNamespace(weight=w.w13),
                down_proj=SimpleNamespace(weight=w.w2),
            ),
        )
        for w in weights.layers
    ]
    wrapper = object.__new__(Qwen3OmniMoeTalkerCodePredictor)
    nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(rms_norm_eps=weights.eps, rope_parameters={"rope_theta": weights.rope_theta})
    wrapper.model = SimpleNamespace(
        layers=layers,
        norm=SimpleNamespace(weight=weights.norm),
        codec_embedding=[SimpleNamespace(weight=t.clone()) for t in weights.codec_embeddings],
    )
    wrapper.lm_head = [SimpleNamespace(weight=t.clone()) for t in weights.lm_heads]
    wrapper._top_k, wrapper._top_p = TOP_K, TOP_P
    wrapper.megakernel = CodePredictorMegakernel(ctas=ctas)
    wrapper.megakernel.load_weights(wrapper)
    return wrapper


@pytest.mark.parametrize("draw", ["given", "generator", "generators"])
def test_code_predictor_forward_runs_on_the_kernel(cp_weights, draw):
    wrapper = _code_predictor_stand_in(cp_weights, ctas=96)
    hidden, code0_embed, uniforms, _ = _frames(cp_weights, count=2, seed=5)
    layer0_code = torch.tensor([7, 11], device="cuda")
    kwargs = {"sample_uniforms": uniforms}
    if draw == "generator":
        kwargs = {"generator": _generator(9)}
        uniforms = torch.empty_like(uniforms).uniform_(_UNIFORM_EPS, 1 - _UNIFORM_EPS, generator=_generator(9))
    elif draw == "generators":
        kwargs = {"generators": [_generator(9), _generator(10)]}
        for row, seed in enumerate((9, 10)):
            uniforms[row].uniform_(_UNIFORM_EPS, 1 - _UNIFORM_EPS, generator=_generator(seed))

    all_codes, proj_buf = wrapper.forward(layer0_code, code0_embed, hidden.view(2, 1, -1), **kwargs)
    want_codes, _ = cp.CodePredictor(cp_weights, TOP_K, TOP_P, ctas=96).run(hidden, code0_embed, uniforms)

    assert all_codes.shape == (2, cp.CODE_GROUPS, 1)
    assert torch.equal(all_codes[:, 0, 0], layer0_code)
    assert torch.equal(all_codes[:, 1:, 0], want_codes)
    assert proj_buf.shape == (2, cp.POSITIONS + 1, cp.DIM)
    assert torch.equal(proj_buf[:, 0], hidden)
    assert torch.equal(proj_buf[:, 1], code0_embed)
    tables = torch.arange(cp.HEADS, device="cuda")
    assert torch.equal(proj_buf[:, 2:], cp_weights.codec_embeddings[tables, want_codes])


def test_code_predictor_aliases_the_modules_heads_and_embeddings(cp_weights):
    wrapper = _code_predictor_stand_in(cp_weights)
    stacked = wrapper.megakernel._predictor.weights
    for i, head in enumerate(wrapper.lm_head):
        assert head.weight.data_ptr() == stacked.lm_heads[i].data_ptr()
    for i, embedding in enumerate(wrapper.model.codec_embedding):
        assert embedding.weight.data_ptr() == stacked.codec_embeddings[i].data_ptr()
    assert torch.equal(stacked.lm_heads, cp_weights.lm_heads)
    assert torch.equal(stacked.codec_embeddings, cp_weights.codec_embeddings)
