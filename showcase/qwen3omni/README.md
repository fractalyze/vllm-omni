# Qwen3-Omni speech showcase

Qwen3-Omni-30B-A3B (AWQ-4bit) speech output on one RTX 5090, served by stock
vLLM-Omni from this branch. `control_marlin.yaml` is the deploy config: three
stages (thinker, talker, code2wav) on GPU 0, with the thinker's MoE on Marlin.

## Install

From a clone of this branch:

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install vllm==0.30.0 --torch-backend=auto
VIRTUAL_ENV=.venv uv pip install -e .
```

If the host has a system cuDNN newer than the venv's, code2wav fails at
start-up with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`: the venv's cuDNN
lacks `libcudnn_engines_tensor_ir`, so the system copy is loaded beside it.
Install the venv cuDNN at the system's version, for example
`VIRTUAL_ENV=.venv uv pip install 'nvidia-cudnn-cu13==9.23.*'`.

## Serve

```bash
PATH=$PWD/.venv/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit --omni --port 8091 \
    --deploy-config showcase/qwen3omni/control_marlin.yaml
```

The server is ready when `curl -sf localhost:8091/health` succeeds. An
`expandable_segments: memory mapping failed with OOM` warning at start-up is
harmless: the server comes up regardless.

## Thinker megakernels

`thinker_megakernel.yaml` serves the same pipeline with the thinker's decode
steps, and its prompt chunks of 2 to 64 tokens, each on one kernel launch
(`vllm_omni/model_executor/models/qwen3_omni/megakernel/`). The first start
compiles the kernels with the venv's CUDA toolkit, so `.venv/bin` must be on
`PATH` for its `ninja`.

```bash
PATH=$PWD/.venv/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_OMNI_THINKER_MEGAKERNEL=1 VLLM_OMNI_THINKER_MEGAKERNEL_CTAS=64 \
VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL=1 VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS=128 \
vllm serve cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit --omni --port 8091 \
    --deploy-config showcase/qwen3omni/thinker_megakernel.yaml
```

| Switch | Effect |
| --- | --- |
| `VLLM_OMNI_THINKER_MEGAKERNEL=1` | One-token decode steps run on the decode kernel |
| `VLLM_OMNI_THINKER_MEGAKERNEL_CTAS` | SMs a decode step takes, leaving the rest to the talker and code2wav; unset is every SM |
| `VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL=1` | With the first switch, one request's 2–64-token prompt chunks run on the prefill kernel |
| `VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS` | SMs a prefill chunk takes; unset is every SM |

- The kernels read the thinker's experts in the checkpoint's packing, so
  stage 0 must use `moe_backend: triton`; on Marlin the first step raises.
- Prompts with image or video inputs, chunks over 64 tokens and batches of
  two requests keep the stock forward.
- The CTA caps change how the kernels split their sums, so a prompt's text
  depends on them; at fixed caps every repeat gives the same text.
- The kernels are built for sm_120a (RTX 5090).

## Request

```bash
curl -s localhost:8091/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit",
  "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
  "modalities": ["text", "audio"]
}' > reply.json
jq -r '.choices[0].message.content' reply.json
jq -r '.choices[0].message.audio.data' reply.json | base64 -d > reply.wav
```

`reply.wav` is 24 kHz mono 16-bit PCM.
