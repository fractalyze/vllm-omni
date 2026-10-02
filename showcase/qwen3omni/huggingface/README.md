---
license: apache-2.0
base_model: Qwen/Qwen3-Omni-30B-A3B-Instruct
pipeline_tag: any-to-any
tags:
  - vllm-omni
  - text-to-speech
  - awq
  - rtx-5090
  - megakernel
  - inference-optimization
---

# Qwen3-Omni speech on one RTX 5090: first audio in 23 ms

Qwen3-Omni-30B-A3B (AWQ-4bit) text and speech on a single RTX 5090 with [vLLM-Omni](https://github.com/vllm-project/vllm-omni): **first audio about nine times sooner** and **text nearly four times faster** than stock vLLM-Omni on the same GPU, with no higher word error rate on Qwen3-ASR ([Results](#results)).

- **Code:** [fractalyze/vllm-omni @ `qwen3omni/showcase`](https://github.com/fractalyze/vllm-omni/tree/qwen3omni/showcase)
- **This repo:** results and how to reproduce them. No model weights: use [cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit), a quantization of [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) (their own licenses apply).

## What's inside

- **Megakernels for RTX 5090 (sm_120a):** the thinker's decode steps and short prompt chunks, the talker's decode steps and its code predictor each run on one kernel launch, reading the checkpoint's own weight packing.
- **Deterministic Marlin MoE:** a patch of vLLM's Marlin MoE so identical requests give the same text.
- **A shorter path to first audio:** the talker starts at the thinker's first token, ships its first audio frame with its prefill step, and decodes it straight to the API; the three stages share the GPU under CUDA MPS.

Every change is behind a `VLLM_OMNI_*` environment switch, off by default.

## Results

Three text prompts, five times each, after one warm-up request; RTX 5090, batch 1, one server per arm, all three arms served one after another in one session. Median (min–max).

| Configuration | TTFT | TTFA | TTFA, probes | Text, tok/s | RTF |
| --- | --: | --: | --: | --: | --: |
| stock vLLM-Omni | 57 ms (49–60) | 213 ms (206–220) | 214 ms (206–215) | 64.1 (64.1–64.9) | 0.112 (0.104–0.114) |
| deterministic Marlin | 18 ms (17–19) | 42 ms (39–43) | 41 ms (40–43) | 185.2 (185.2–188.7) | 0.058 (0.057–0.060) |
| **kernels** | **16 ms** (15–17) | **23 ms** (21–24) | **23 ms** (21–24) | **238.1** (238.1–238.1) | **0.052** (0.052–0.053) |

- TTFT and TTFA are time to the first streamed text token and the first audio chunk. RTF is generation time over audio length.
- TTFA, probes: 15 more requests on the same server right after the timed ones, the three prompts cycled five times.

Quality: each reply's audio transcribed by [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) and scored against that reply's own text, over all 15 replies.

| Configuration | WER | Prompt 0 | Prompt 1 | Prompt 2 | Distinct texts per prompt |
| --- | --: | --: | --: | --: | --: |
| stock vLLM-Omni | 1.59% | 4.4% | 0.0% | 0.3% | 1, 1, 1 |
| deterministic Marlin | 2.64% | 6.3% | 0.0% | 0.2% | 1, 1, 1 |
| **kernels** | **1.01%** | 2.2% | 0.0% | 0.6% | 1, 1, 1 |

Each prompt repeats one text, so a prompt's errors count five times. The protocol, commits, configs and environment of every arm are in [`measurements.md`](https://github.com/fractalyze/vllm-omni/blob/9122317a9f62fff0bec41ebda94662020f73109c/showcase/qwen3omni/measurements.md).

## Quick start

Requires an RTX 5090, CUDA 13 and Python 3.12.

```bash
git clone --branch qwen3omni/showcase https://github.com/fractalyze/vllm-omni.git && cd vllm-omni
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install vllm==0.30.0 --torch-backend=auto
VIRTUAL_ENV=.venv uv pip install -e .
```

If code2wav fails at start-up with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, the host's cuDNN is newer than the venv's: install the venv cuDNN at the host's version, for example `VIRTUAL_ENV=.venv uv pip install 'nvidia-cudnn-cu13==9.23.*'`.

Both arms run the three stages on one GPU under CUDA MPS:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-$USER/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps-$USER/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
```

**Kernel arm.** The first start compiles the kernels with the venv's CUDA toolkit, so `.venv/bin` must be on `PATH`.

```bash
PATH=$PWD/.venv/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_OMNI_THINKER_MEGAKERNEL=1 VLLM_OMNI_THINKER_MEGAKERNEL_CTAS=64 \
VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL=1 VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS=128 \
VLLM_OMNI_TALKER_MEGAKERNEL=1 \
VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL=1 VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS=96 \
VLLM_OMNI_DETERMINISTIC_MARLIN=1 \
VLLM_OMNI_CODE2WAV_STREAM_GRAPHS=1 VLLM_OMNI_CODE2WAV_COMPILE=1 \
VLLM_OMNI_FRAME0=1 VLLM_OMNI_EARLY_CHUNK=1 VLLM_OMNI_FRAME0_AUDIO=1 \
VLLM_OMNI_TALKER_PREP=1 VLLM_OMNI_FAST_POLL=1 \
VLLM_OMNI_THINKER_YIELD=1 VLLM_OMNI_TALKER_PREPREFILL=1 \
VLLM_OMNI_EVENT_DRIVEN_ORCH=1 \
vllm serve cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit --omni --port 8091 \
    --deploy-config showcase/qwen3omni/kernels.yaml
```

**Deterministic Marlin arm.** The thinker on vLLM's Marlin MoE, patched to give the same bits for identical requests:

```bash
PATH=$PWD/.venv/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_OMNI_DETERMINISTIC_MARLIN=1 \
VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL=1 VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS=96 \
VLLM_OMNI_CODE2WAV_STREAM_GRAPHS=1 VLLM_OMNI_FRAME0=1 VLLM_OMNI_EVENT_DRIVEN_ORCH=1 \
vllm serve cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit --omni --port 8091 \
    --deploy-config showcase/qwen3omni/control_marlin.yaml
```

The server is ready when `curl -sf localhost:8091/health` succeeds. Ask for text and speech:

```bash
curl -s localhost:8091/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit",
  "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
  "modalities": ["text", "audio"]
}' > reply.json
jq -r '.choices[0].message.content' reply.json
jq -r '.choices[0].message.audio.data' reply.json | base64 -d > reply.wav   # 24 kHz mono 16-bit PCM
```

Stop the MPS daemon after the server with `echo quit | nvidia-cuda-mps-control`. Every switch is described in the branch's [`showcase/qwen3omni/README.md`](https://github.com/fractalyze/vllm-omni/blob/qwen3omni/showcase/showcase/qwen3omni/README.md).

## Limitations

- **One GPU, batch 1.** The kernels are built for sm_120a (RTX 5090) and serve one request at a time; a batch of two requests falls back to the stock forward. Multi-user throughput was not measured.
- **Measured on text prompts only.** The thinker kernels keep the stock forward for prompts with image or video inputs and for prompt chunks over 64 tokens. The switches that start the talker early (`VLLM_OMNI_EARLY_CHUNK`, `VLLM_OMNI_TALKER_PREP`, `VLLM_OMNI_TALKER_PREPREFILL`) keep the stock path for prompts with audio, image or video inputs.
- **Text is deterministic per prompt; audio per request sequence.** Every repeat of a prompt gives the same text. The talker's code predictor samples from the talker stage's shared CUDA random stream rather than the request's own generator, so a fresh server replays the same audio for the same sequence of requests, but repeats of one prompt sound different.
- **Kernel settings change the output.** The CTA caps (`*_CTAS`) set how the kernels split their sums, so a prompt's text and audio depend on them; at fixed caps every repeat gives the same text.
- **The kernel arm needs the triton MoE.** The thinker kernels read the checkpoint's own expert packing, so `kernels.yaml` serves stage 0 on `moe_backend: triton`.
