# Attention arms

One file per arm of the Kandinsky 6 attention backend race
(`../attn_race.py`). Each is a value for vLLM-Omni's
`--diffusion-attention-config`, so adopting a winner is a **config change,
not a code change**:

```bash
vllm serve kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers --omni \
    --diffusion-attention-config "$(cat showcase/kandinsky6/compute/arms/sage2.json)"
```

`AttentionConfig` resolves a backend per call site in this order: an exact
`per_role` match, then the role's category (`self` / `cross`), then
`default`, then the platform default. Kandinsky 6's five call sites are
`kandinsky6.visual_self`, `kandinsky6.text_cross`,
`kandinsky6.video_audio_cross`, `kandinsky6.audio_video_cross` and
`kandinsky6.audio_self`, named in `kandinsky6_transformer.py`.

Only `visual_self` is set here, for three reasons that point the same way.

1. It is 98.8% of a block's attention FLOPs (41.3 of 41.8 TFLOP), so it is
   where a faster kernel pays.
2. **It is the only mask-free call site, and the FP8/FP4 kernels require
   that.** `SageAttentionImpl.forward_cuda` raises
   `"SAGE_ATTN does not support attn_mask"` outright. Both Kandinsky 6
   bundles set `text_token_padding: true`, and the fused block hands
   `attn_mask` to `cross_attention` for text and to the audio branch, while
   visual self-attention is called with only `rotary_emb` and
   `sparse_params`. So a `default:` arm — one that pointed every role at
   SAGE_ATTN — would not be slower, it would raise on the first padded
   prompt. Per-role is not tidiness here; it is the only form that runs.
3. One variable per arm means a quality regression has exactly one cause.

The cheap roles are raced separately by `attn_race.py --all-roles`, and that
race is what justifies `tuned.json`'s second variable.

## `tuned.json`

The single-backend arms answer "which kernel for the call that dominates".
`tuned.json` answers "which kernel for each call", and the answer is not the
same one twice:

| call site | q | kv | winner | vs `CUDNN_ATTN` | saved a block |
|---|---:|---:|---|---:|---:|
| `visual_self` | 50,220 | 50,220 | `SAGE_ATTN` | 2.52x | 107.13 ms |
| `audio_video_cross` | 218 | 50,220 | `TORCH_SDPA` | 4.21x | 1.91 ms |
| `video_audio_cross` | 50,220 | 218 | `SAGE_ATTN` | 1.20x | 0.19 ms |
| `audio_self` | 218 | 218 | `TORCH_SDPA` | 1.36x | 0.01 ms |
| `text_self`, `text_cross` | — | — | *unset* | — | — |

`audio_video_cross` is the interesting one: reverse the shape and
SageAttention goes from 2.5x ahead to 3.2x behind, because its per-block
quantization prologue is paid per *query* block and 218 queries give it no mma
work to amortize against. cuDNN, the platform default, is the worst of the
three there.

The two text roles stay unset because they are the two that receive a padding
mask (`test_role_masks.py` asserts exactly which four do not), and of the
mask-capable arms cuDNN was already the fastest on `text_cross`
(1.127 ms vs `TORCH_SDPA`'s 1.176 ms).

> The per-call numbers are measured (`k6c-a01`); the 109.24 ms a block they
> sum to is **arithmetic**, not a measurement of this file. The four calls were
> timed one at a time, so the sum assumes they do not interact — fair for the
> 107 ms that is one mma-bound kernel, less obviously so for the 2.1 ms of the
> three cheap ones. Whether `tuned.json` beats `sage2.json` end to end is for
> the harness to say.

| file | what it pins |
|---|---|
| `control.json` | nothing — platform default (`CUDNN_ATTN` on sm_120) |
| `sage2.json` | `visual_self` → `SAGE_ATTN`, SageAttention 2++ (INT8 QK, FP8 PV) |
| `sage3.json` | `visual_self` → `SAGE_ATTN_3`, SageAttention 3 (FP4, Blackwell only) |
| `flash.json` | `visual_self` → `FLASH_ATTN`, FlashAttention 4 (`flash_attn.cute`) |
| `tuned.json` | every mask-free call site on whatever won it in the race (below) |

`control.json` sets `"auto"`, which `AttentionConfig` normalizes to "no
override" rather than to a backend named auto — so the control arm is written
down and version-controlled like the others instead of being "pass no flag".

**No file here claims to be the winner.** The race decides, and the result
goes in `showcase/kandinsky6/measurements.md` with the arm it names.
