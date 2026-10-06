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

Only `visual_self` is set here. It is 98.8% of a block's attention FLOPs, and
leaving the four cheap call sites on the platform default keeps each arm a
one-variable change: a quality regression then has exactly one cause. The
cheap roles are raced separately by `attn_race.py --all-roles`, which is how a
second variable would be justified.

| file | `kandinsky6.visual_self` |
|---|---|
| `control.json` | platform default — `CUDNN_ATTN` on sm_120 |
| `sage2.json` | `SAGE_ATTN` — SageAttention 2++, INT8 QK with FP8 PV |
| `sage3.json` | `SAGE_ATTN_3` — SageAttention 3, FP4, Blackwell only |
| `flash.json` | `FLASH_ATTN` — FlashAttention 4 (`flash_attn.cute`) on Blackwell |

`control.json` sets `"auto"`, which `AttentionConfig` normalizes to "no
override" rather than to a backend named auto — so the control arm is written
down and version-controlled like the others instead of being "pass no flag".

**No file here claims to be the winner.** The race decides, and the result
goes in `showcase/kandinsky6/measurements.md` with the arm it names.
