# Qwen3-Omni speech showcase

Qwen3-Omni-30B-A3B (AWQ-4bit) speech output on one RTX 5090, served by stock
vLLM-Omni from this branch. `control_marlin.yaml` is the deploy config: three
stages (thinker, talker, code2wav) on GPU 0, with the thinker's MoE on Marlin.
[measurements.md](measurements.md) compares the kernel arm with stock
vLLM-Omni and the deterministic Marlin arm.

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

`kernels.yaml` serves the same pipeline with the thinker's decode steps, and
its prompt chunks of 2 to 64 tokens, each on one kernel launch
(`vllm_omni/model_executor/models/qwen3_omni/megakernel/`), and code2wav on
CUDA graphs. Those graphs fit only at the streaming sizes, so the config
needs `VLLM_OMNI_CODE2WAV_STREAM_GRAPHS=1` (below). The first start compiles
the kernels with the venv's CUDA toolkit, so `.venv/bin` must be on `PATH`
for its `ninja`.

```bash
PATH=$PWD/.venv/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_OMNI_THINKER_MEGAKERNEL=1 VLLM_OMNI_THINKER_MEGAKERNEL_CTAS=64 \
VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL=1 VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS=128 \
VLLM_OMNI_CODE2WAV_STREAM_GRAPHS=1 \
vllm serve cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit --omni --port 8091 \
    --deploy-config showcase/qwen3omni/kernels.yaml
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

## Talker and code-predictor megakernels

Two more switches put stage 1 on kernels from the same build: the talker's
one-token decode steps, and its code predictor (codes 1 to 15 of each audio
frame). They work with either deploy config; beside the thinker kernels:

```bash
PATH=$PWD/.venv/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_OMNI_THINKER_MEGAKERNEL=1 VLLM_OMNI_THINKER_MEGAKERNEL_CTAS=64 \
VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL=1 VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS=128 \
VLLM_OMNI_TALKER_MEGAKERNEL=1 \
VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL=1 VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS=96 \
VLLM_OMNI_CODE2WAV_STREAM_GRAPHS=1 \
vllm serve cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit --omni --port 8091 \
    --deploy-config showcase/qwen3omni/kernels.yaml
```

| Switch | Effect |
| --- | --- |
| `VLLM_OMNI_TALKER_MEGAKERNEL=1` | The talker's one-token decode steps run on the talker kernel |
| `VLLM_OMNI_TALKER_MEGAKERNEL_CTAS` | SMs a talker step takes; unset is 96, which leaves the thinker's decode its SMs |
| `VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL=1` | Every code-predictor call runs on the code-predictor kernel |
| `VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS` | SMs a code-predictor launch takes, leaving the rest to code2wav; unset is every SM |

- The talker kernel reads the talker's bf16 weights and paged KV cache in
  place; prompts and batches of two requests keep the stock forward.
- The code predictor kernel is built when stage 1 loads its weights, so
  vLLM's memory profile counts it. It draws each frame's 15 codes from
  uniforms drawn up front: the same distribution as the stock sampler, not
  the same random stream, so the audio differs from the stock arm's.
- As with the thinker, the CTA counts change how the kernels split their
  sums, so a prompt's audio depends on them. Which SMs the CTAs run on does
  not, so CUDA MPS leaves the audio unchanged.
- The talker calls its code predictor without the request's generator
  (`code_predictor_forward`), so codes 1 to 15 are drawn from the talker
  stage's own CUDA random stream, on the stock path as on the kernel. A
  request's audio therefore depends on every talker step the server took
  before it: a fresh server replays the same audio for the same sequence of
  requests, and repeats of one prompt differ.
- The talker's prompt runs on vLLM's compiled forward, loaded from vLLM's
  `torch.compile` cache. Inductor picks some of its kernels' launch
  configurations by timing them, so two cache entries (another environment,
  or a recompile) can round differently. One different code then shifts the
  audio of every later request while the text stays the same, so two arms
  compare byte for byte only when they load the same compiled talker.

## Serving switches

These switches shorten the path from a request to its first audio around
the kernels (`vllm_omni/model_executor/models/qwen3_omni/serving/` and the
call sites it names). Each is off unless set to `1`.

| Switch | Effect |
| --- | --- |
| `VLLM_OMNI_DETERMINISTIC_MARLIN` | vLLM's Marlin MoE gives the same bits across identical requests: each expert's rows are sorted by row before a prefill's grouped GEMM (`vllm_omni/patch.py`) |
| `VLLM_OMNI_CODE2WAV_STREAM_GRAPHS` | code2wav captures CUDA graphs only at the frame counts a streaming decode uses |
| `VLLM_OMNI_CODE2WAV_COMPILE` | code2wav decodes a one-frame first chunk on `torch.compile` |
| `VLLM_OMNI_FRAME0` | The talker's first audio frame ships with its prefill step, one step sooner |
| `VLLM_OMNI_EARLY_CHUNK` | The talker gets its prefill input at the thinker's first token, one thinker step sooner |
| `VLLM_OMNI_FRAME0_AUDIO` | With `VLLM_OMNI_FRAME0`, the talker decodes frame 0 on code2wav's weights (CUDA IPC) and sends the first chunk straight to the API |
| `VLLM_OMNI_TALKER_PREP` | The talker builds its prefill input from a text-only prompt without host syncs |
| `VLLM_OMNI_FAST_POLL` | A stage wakes when an upstream chunk lands instead of on its next 1 ms poll |
| `VLLM_OMNI_THINKER_YIELD` | With `VLLM_OMNI_FRAME0`, the thinker's decode waits while the talker makes a request's frame 0 (at most 12 ms) |
| `VLLM_OMNI_TALKER_PREPREFILL` | With `VLLM_OMNI_EARLY_CHUNK`, the talker prefills all but its last prompt position while the thinker prefills |
| `VLLM_OMNI_QWEN3_OMNI_RUN_DIR` | The directory a server's stages and API share for `VLLM_OMNI_FRAME0_AUDIO` and `VLLM_OMNI_THINKER_YIELD`; unset is one per user under the system temp directory, so servers sharing a host need one each |

- `VLLM_OMNI_DETERMINISTIC_MARLIN` serves `control_marlin.yaml`'s Marlin
  thinker; with the kernels the thinker runs on triton and the sort is
  idle. The sort adds a few kernels to each prefill layer; decode steps skip
  it. The fix belongs in vLLM: the branch pins upstream `vllm==0.30.0`, so it
  is carried as a patch of `marlin_moe.moe_align_block_size`.
- Prompts with audio, image or video keep the stock path under
  `VLLM_OMNI_EARLY_CHUNK`, `VLLM_OMNI_TALKER_PREP` and
  `VLLM_OMNI_TALKER_PREPREFILL`.
- `VLLM_OMNI_CODE2WAV_COMPILE` changes the first chunk's samples by a few
  units of the 16-bit range; every later sample is the same.
- `VLLM_OMNI_TALKER_PREPREFILL` runs the talker's prompt in two steps, which
  changes its audio.

### The kernel arm

Every kernel and serving switch, with the stages sharing the GPU under CUDA
MPS so their kernels run at once instead of taking time slices:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-$USER/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps-$USER/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d

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

Stop the daemon after the server with `echo quit | nvidia-cuda-mps-control`.

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
