# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Qwen3-Omni thinker megakernels on a few random layers.

Each block, the decode step and the prefill chunk are held to an fp32 PyTorch
model on the dequantized weights, within FACTOR × the error of the same model
in bf16. The thinker's forward is then run with the megakernel switched on, on
a stand-in holding those layers as vLLM does after loading, so the kernels run
without a checkpoint: an eligible step must give exactly what the kernel gives
when launched directly, and every other step must reach the stock forward.

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
    pytest.mark.skipif(not _is_sm120(), reason="the thinker megakernels are built for sm_120a"),
]

if _is_sm120():
    import vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker as megakernel_thinker
    import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker as thinker_module
    from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker import ThinkerMegakernel
    from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_attention import (
        DIM,
        HEAD_DIM,
        KV_HEADS,
        AttentionWeights,
        PagedCache,
        ThinkerAttention,
        cos_sin_table,
    )
    from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_attention import (
        reference as attention_reference,
    )
    from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_decode import (
        ThinkerDecoder,
        ThinkerWeights,
        new_caches,
    )
    from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_moe import MoeWeights, ThinkerMoe
    from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_moe import reference as moe_reference
    from vllm_omni.model_executor.models.qwen3_omni.megakernel.thinker_prefill import (
        ThinkerPrefiller,
        reference_prefill,
    )
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

# A kernel's error may exceed bf16's own by this factor.
FACTOR = 1.25
LAYERS, BLOCKS, BLOCK_SIZE = 3, 12, 16


def _rel_l2(x: torch.Tensor, ref: torch.Tensor) -> float:
    x, ref = x.float(), ref.float()
    return float((x - ref).norm() / ref.norm())


def _within_budget(got, truth, budget, what: str) -> None:
    error, allowed = _rel_l2(got, truth), _rel_l2(budget, truth)
    assert error <= FACTOR * allowed, f"{what}: error {error:.3g} exceeds {FACTOR} × bf16's {allowed:.3g}"


@pytest.fixture(scope="module")
def layers():
    return [(AttentionWeights.random(seed=2 * i), MoeWeights.random(seed=2 * i + 1)) for i in range(LAYERS)]


@pytest.fixture(scope="module")
def final_norm():
    gen = torch.Generator(device="cuda").manual_seed(0)
    return (1 + 0.1 * torch.randn(DIM, generator=gen, device="cuda")).bfloat16()


@pytest.fixture(scope="module")
def cos_sin():
    return cos_sin_table(512)


def _caches(table: torch.Tensor) -> list[PagedCache]:
    """Empty caches in vLLM's FlashAttention layout: one [blocks, KV_HEADS,
    block_size, 2 × HEAD_DIM] tensor per layer, so a slot's heads are not
    adjacent."""
    caches = []
    for _ in range(LAYERS):
        kv = torch.zeros(BLOCKS, KV_HEADS, BLOCK_SIZE, 2 * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        key, value = kv.transpose(1, 2).split(HEAD_DIM, dim=-1)
        caches.append(PagedCache(key, value, table))
    return caches


# (pos, M-RoPE positions): the first token; text, whose three positions agree;
# and an audio or image token, whose height and width differ.
@pytest.mark.parametrize("pos,positions", [(0, (0, 0, 0)), (37, (37, 37, 37)), (150, (140, 151, 173))])
def test_attention_block_within_bf16_budget(layers, cos_sin, pos, positions):
    weights = layers[0][0]
    gen = torch.Generator(device="cuda").manual_seed(pos)
    residual = 4 * torch.randn(DIM, generator=gen, device="cuda")
    kv = torch.randn(BLOCKS, KV_HEADS, BLOCK_SIZE, 2 * HEAD_DIM, generator=gen, device="cuda").bfloat16()
    key, value = kv.transpose(1, 2).split(HEAD_DIM, dim=-1)
    cache = PagedCache(key, value, torch.randperm(BLOCKS, generator=gen, device="cuda").int())
    positions = torch.tensor(positions, dtype=torch.int32, device="cuda")
    truth = attention_reference(weights, residual, cache, pos, positions, cos_sin, torch.float32)
    budget = attention_reference(weights, residual, cache, pos, positions, cos_sin, torch.bfloat16)

    got = ThinkerAttention(weights, cos_sin).run(residual, cache, pos, positions)

    _within_budget(got - residual, truth.residual - residual, budget.residual - residual, "attention")
    written_k, written_v = cache.gather(pos + 1)
    torch.testing.assert_close(written_k[pos].float(), truth.key.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(written_v[pos].float(), truth.value.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("seed", [1, 2])
def test_moe_block_within_bf16_budget(layers, seed):
    weights = layers[0][1]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    residual = 4 * torch.randn(DIM, generator=gen, device="cuda")

    got = ThinkerMoe(weights).run(residual)

    truth = moe_reference(weights, residual, torch.float32)
    budget = moe_reference(weights, residual, torch.bfloat16)
    assert torch.equal(got.experts.cpu(), truth.experts.cpu())
    _within_budget(got.residual - residual, truth.residual - residual, budget.residual - residual, "moe")


# None is one CTA per SM; 64 leaves SMs to the stages beside the thinker.
@pytest.mark.parametrize("ctas", [None, 64])
def test_decode_step_matches_blocks_launched_one_by_one(layers, final_norm, cos_sin, ctas):
    """The step stacks the blocks' own device code in one launch, so it must
    match them bit for bit, even where routing sits on a bf16 tie."""
    steps, positions_held = 6, 64
    weights = ThinkerWeights(embed=None, layers=layers, final_norm=final_norm, lm_head=None)
    decoder = ThinkerDecoder(weights, new_caches(LAYERS, positions_held), cos_sin, ctas=ctas)
    caches = new_caches(LAYERS, positions_held)
    blocks = [(ThinkerAttention(a, cos_sin, ctas=ctas), ThinkerMoe(m, ctas=ctas)) for a, m in layers]
    gen = torch.Generator(device="cuda").manual_seed(0)
    block_size = caches[0].key.shape[1]
    for pos in range(steps):
        embedding = torch.randn(DIM, generator=gen, device="cuda").bfloat16()
        positions = torch.tensor([pos, pos + 3, pos + 5], dtype=torch.int64, device="cuda")
        table = decoder.caches[0].block_table
        slot = table[pos // block_size].long().view(1) * block_size + pos % block_size
        hidden = torch.zeros(LAYERS + 1, DIM, device="cuda")
        decoder.launch(
            embedding,
            torch.tensor([pos + 1], dtype=torch.int32, device="cuda"),
            slot,
            positions.view(3, 1),
            hidden=hidden,
        )
        current_omni_platform.synchronize()

        x = embedding.float()
        for i, ((attention, moe), cache) in enumerate(zip(blocks, caches)):
            assert torch.equal(hidden[i], x), f"step {pos}: layer {i}'s input"
            x = moe.run(attention.run(x, cache, pos, positions.int())).residual
        assert torch.equal(hidden[LAYERS], x), f"step {pos}: the last layer's output"


# (tokens, pos0, ctas): a partial n-tile, the most a chunk takes, a chunk
# after 9 cached positions, and 20 CTAs, the fewest that take q, k, v's 5120
# rows at 256 a CTA.
@pytest.mark.parametrize("tokens,pos0,ctas", [(7, 0, None), (64, 0, None), (13, 9, None), (23, 0, 20)])
def test_prefill_chunk_within_bf16_budget(layers, final_norm, cos_sin, tokens, pos0, ctas):
    """The references take the kernel's own routing: the router rounds logits
    to bf16, where a near-tie decides an expert, and the error measured is the
    kernel's arithmetic."""
    weights = ThinkerWeights(embed=None, layers=layers, final_norm=final_norm, lm_head=None)
    table = torch.randperm(BLOCKS, device="cuda").int()
    kernel_caches, truth_caches, bf16_caches = _caches(table), _caches(table), _caches(table)

    def chunk(n, start, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        embeddings = torch.randn(n, DIM, generator=gen, device="cuda").bfloat16().float()
        base = torch.arange(start, start + n, device="cuda")
        return embeddings, torch.stack([base, base + 2, base + 5])

    if pos0:
        prefix, prefix_positions = chunk(pos0, 0, seed=100)
        reference_prefill(weights, truth_caches, prefix, 0, prefix_positions, cos_sin, torch.float32)
        for src in (kernel_caches, bf16_caches):
            for s, d in zip(truth_caches, src):
                d.key.copy_(s.key)
                d.value.copy_(s.value)
    embeddings, positions = chunk(tokens, pos0, seed=tokens)
    prefiller = ThinkerPrefiller(ThinkerDecoder(weights, kernel_caches, cos_sin), ctas=ctas)

    got = prefiller.run(embeddings, pos0, positions).float()

    routes = [prefiller.routing(layer, tokens) for layer in range(LAYERS)]
    truth = reference_prefill(weights, truth_caches, embeddings, pos0, positions, cos_sin, torch.float32, routes)
    budget = reference_prefill(weights, bf16_caches, embeddings, pos0, positions, cos_sin, torch.bfloat16, routes)
    for i in range(tokens):
        _within_budget(got[i], truth[i], budget[i], f"token {i}")


class _StockModel:
    """The thinker's language model: the layers the kernels read, and a stock
    forward that records each call."""

    def __init__(self, layers, norm):
        self.layers = layers
        self.norm = norm
        self.calls: list[int] = []

    def __call__(self, input_ids, positions, intermediate_tensors, inputs_embeds=None, **kwargs):
        self.calls.append(inputs_embeds.shape[0])
        return torch.zeros_like(inputs_embeds), None


def _stand_in(layers, final_norm, cos_sin, prefill=False, marlin_experts=False):
    """A thinker holding `layers` as vLLM's does after loading (the experts as
    moe_backend triton keeps them, as uint8 bytes) with the megakernel on."""
    modules = []
    for aw, mw in layers:
        kv = torch.zeros(BLOCKS, KV_HEADS, BLOCK_SIZE, 2 * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        w13 = mw.w13_packed.view(torch.uint8)
        if marlin_experts:
            w13 = torch.zeros(128, 128, 3072, dtype=torch.int32, device="cuda")
        modules.append(
            SimpleNamespace(
                input_layernorm=SimpleNamespace(weight=aw.norm, variance_epsilon=1e-6),
                post_attention_layernorm=SimpleNamespace(weight=mw.norm, variance_epsilon=1e-6),
                self_attn=SimpleNamespace(
                    qkv_proj=SimpleNamespace(weight_packed=aw.wqkv_packed, weight_scale=aw.wqkv_scales),
                    o_proj=SimpleNamespace(weight_packed=aw.wo_packed, weight_scale=aw.wo_scales),
                    q_norm=SimpleNamespace(weight=aw.q_norm),
                    k_norm=SimpleNamespace(weight=aw.k_norm),
                    rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin.float()),
                    attn=SimpleNamespace(kv_cache=kv),
                ),
                mlp=SimpleNamespace(
                    gate=SimpleNamespace(weight=mw.router),
                    experts=SimpleNamespace(
                        routed_experts=SimpleNamespace(
                            w13_weight_packed=w13,
                            w13_weight_scale=mw.w13_scales,
                            w2_weight_packed=mw.w2_packed.view(torch.uint8),
                            w2_weight_scale=mw.w2_scales,
                        )
                    ),
                ),
            )
        )
    thinker = object.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
    nn.Module.__init__(thinker)
    thinker.use_deepstack = False
    thinker.language_model = SimpleNamespace(model=_StockModel(modules, SimpleNamespace(weight=final_norm)))
    thinker.megakernel = ThinkerMegakernel(decode_ctas=None, prefill=prefill, prefill_ctas=None)
    thinker.megakernel.load_weights(thinker)
    return thinker


@pytest.fixture
def first_rank(monkeypatch):
    monkeypatch.setattr(thinker_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))


def _use_metadata(monkeypatch, metadata):
    monkeypatch.setattr(megakernel_thinker, "_attn_metadata", lambda: metadata)


def _table():
    """The one request's blocks reversed, so positions cross blocks out of order."""
    return torch.arange(BLOCKS, dtype=torch.int32, device="cuda").flip(0).view(1, BLOCKS)


def _kernel_caches(thinker, table):
    caches = []
    for layer in thinker.language_model.model.layers:
        key, value = layer.self_attn.attn.kv_cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
        caches.append(PagedCache(key, value, table))
    return caches


def _decode_metadata(pos):
    table = _table()
    slot = int(table[0, pos // BLOCK_SIZE]) * BLOCK_SIZE + pos % BLOCK_SIZE
    return SimpleNamespace(
        block_table=table,
        num_actual_tokens=1,
        max_query_len=1,
        seq_lens=torch.tensor([pos + 1], dtype=torch.int32, device="cuda"),
        slot_mapping=torch.tensor([slot], dtype=torch.int64, device="cuda"),
    )


def _prefill_metadata(tokens, padded, requests=1):
    """A prefill step of one request's `tokens` prompt tokens from position 0,
    padded to `padded` as a piecewise CUDA graph runs it."""
    table = _table()
    pos = torch.arange(padded, device="cuda")
    slots = table[0].long()[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE
    slots[tokens:] = -1
    return SimpleNamespace(
        block_table=table,
        num_actual_tokens=tokens,
        max_query_len=tokens if requests == 1 else tokens - 1,
        seq_lens=torch.tensor([tokens] * requests, dtype=torch.int32, device="cuda"),
        slot_mapping=slots,
    )


def test_thinker_decode_step_runs_on_the_kernel(layers, final_norm, cos_sin, monkeypatch, first_rank):
    thinker = _stand_in(layers, final_norm, cos_sin)
    pos = 20
    metadata = _decode_metadata(pos)
    _use_metadata(monkeypatch, metadata)
    embeds = torch.randn(1, DIM, device="cuda").bfloat16()
    # vLLM's M-RoPE buffer is [3, max_tokens + 1], so rows are not adjacent.
    rows = torch.zeros(3, 8, dtype=torch.int64, device="cuda")
    rows[:, 0] = pos

    hidden, captured = thinker.forward(
        None, rows[:, :1], inputs_embeds=embeds, capture_layer_indices=[0, 2], return_hidden_states=True
    )
    current_omni_platform.synchronize()
    assert thinker.language_model.model.calls == []

    # The same step on a decoder over caches of its own.
    caches = _caches(metadata.block_table[0])
    decoder = ThinkerDecoder(ThinkerWeights(None, layers, final_norm, None), caches, cos_sin.float().bfloat16())
    want = torch.empty(DIM, dtype=torch.bfloat16, device="cuda")
    want_hidden = torch.zeros(LAYERS + 1, DIM, device="cuda")
    decoder.launch(embeds, metadata.seq_lens, metadata.slot_mapping, rows[:, :1], final_hidden=want, hidden=want_hidden)
    current_omni_platform.synchronize()
    assert torch.equal(hidden.view(-1), want)
    captured_layers = captured["hidden_states"]["layers"]
    assert torch.equal(captured_layers[0], embeds)
    assert torch.equal(captured_layers[2], want_hidden[2].bfloat16().view(1, -1))
    for got, expected in zip(_kernel_caches(thinker, metadata.block_table[0]), caches):
        got_k, _ = got.gather(pos + 1)
        want_k, _ = expected.gather(pos + 1)
        assert torch.equal(got_k[pos], want_k[pos])
        assert got_k[pos].abs().sum() > 0


def test_thinker_prefill_step_runs_on_the_prefill_kernel(layers, final_norm, cos_sin, monkeypatch, first_rank):
    thinker = _stand_in(layers, final_norm, cos_sin, prefill=True)
    tokens, padded = 23, 24
    metadata = _prefill_metadata(tokens, padded)
    _use_metadata(monkeypatch, metadata)
    embeds = torch.randn(padded, DIM, device="cuda").bfloat16()
    positions = torch.arange(padded, device="cuda").repeat(3, 1)

    hidden, captured = thinker.forward(
        None, positions, inputs_embeds=embeds, capture_layer_indices=[0, 2], return_hidden_states=True
    )
    current_omni_platform.synchronize()
    assert thinker.language_model.model.calls == []

    caches = _caches(metadata.block_table[0])
    prefiller = ThinkerPrefiller(
        ThinkerDecoder(ThinkerWeights(None, layers, final_norm, None), caches, cos_sin.float().bfloat16())
    )
    want = torch.empty(tokens, DIM, dtype=torch.bfloat16, device="cuda")
    want_hidden = torch.zeros(tokens, DIM, device="cuda")
    prefiller.launch(
        embeds[:tokens].float(),
        metadata.seq_lens,
        metadata.slot_mapping,
        positions,
        want,
        hidden=want_hidden,
        hidden_layer=2,
    )
    current_omni_platform.synchronize()
    assert hidden.shape == (padded, DIM)
    assert torch.equal(hidden[:tokens], want)
    captured_layers = captured["hidden_states"]["layers"]
    assert torch.equal(captured_layers[0], embeds)
    assert torch.equal(captured_layers[2][:tokens], want_hidden.bfloat16())
    for got, expected in zip(_kernel_caches(thinker, metadata.block_table[0]), caches):
        assert torch.equal(got.gather(tokens)[0], expected.gather(tokens)[0])


def test_thinker_other_steps_keep_the_stock_forward(layers, final_norm, cos_sin, monkeypatch, first_rank):
    embeds = torch.randn(8, DIM, device="cuda").bfloat16()
    positions = torch.arange(8, device="cuda").repeat(3, 1)

    # A prompt chunk without the prefill switch, and a batch of two requests.
    thinker = _stand_in(layers, final_norm, cos_sin)
    _use_metadata(monkeypatch, _prefill_metadata(8, 8))
    thinker.forward(None, positions, inputs_embeds=embeds)
    assert thinker.language_model.model.calls == [8]
    thinker = _stand_in(layers, final_norm, cos_sin, prefill=True)
    _use_metadata(monkeypatch, _prefill_metadata(8, 8, requests=2))
    thinker.forward(None, positions, inputs_embeds=embeds)
    # A prompt with vision inputs: the kernels do not add deepstack embeddings.
    thinker.deepstack_input_embeds_num_tokens = 8
    _use_metadata(monkeypatch, _prefill_metadata(8, 8))
    thinker.forward(None, positions, inputs_embeds=embeds)
    # Longer than a chunk takes.
    thinker.deepstack_input_embeds_num_tokens = 0
    long_embeds = torch.randn(65, DIM, device="cuda").bfloat16()
    _use_metadata(monkeypatch, _prefill_metadata(65, 65))
    thinker.forward(None, torch.arange(65, device="cuda").repeat(3, 1), inputs_embeds=long_embeds)
    # The profile run, which has no KV caches.
    _use_metadata(monkeypatch, None)
    thinker.forward(None, positions[:, :1], inputs_embeds=embeds[:1])

    assert thinker.language_model.model.calls == [8, 8, 65, 1]


def test_thinker_refuses_marlin_experts(layers, final_norm, cos_sin, monkeypatch, first_rank):
    thinker = _stand_in(layers, final_norm, cos_sin, marlin_experts=True)
    _use_metadata(monkeypatch, _decode_metadata(0))
    embeds = torch.randn(1, DIM, device="cuda").bfloat16()
    with pytest.raises(RuntimeError, match="moe_backend triton"):
        thinker.forward(None, torch.zeros(3, 1, dtype=torch.int64, device="cuda"), inputs_embeds=embeds)
