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
