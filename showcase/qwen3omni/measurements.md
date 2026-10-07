# Qwen3-Omni speech measurements

Stock vLLM-Omni, the deterministic Marlin arm and the kernel arm, served one
after another in one session on one RTX 5090, from one venv. The latest
session is first; earlier sessions are kept below as history.

## 2026-10-06

This branch at `1f2cae18`
([#15](https://github.com/fractalyze/vllm-omni/pull/15)): Marlin MoE always
aligns routes with vLLM PR 48032's kernels, and `VLLM_OMNI_DETERMINISTIC_MARLIN`
no longer exists.

Every column but WER is median (min–max).

| Arm | TTFT | TTFA | TTFA, probes | Text, tok/s | RTF | Distinct texts per prompt | WER |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock | 56 ms (49–60) | 214 ms (206–226) | 213 ms (206–215) | 64.1 (64.1–64.9) | 0.111 (0.104–0.114) | 1, 1, 1 | 1.59% |
| deterministic Marlin | 16 ms (14–16) | 39 ms (37–41) | 38 ms (37–43) | 188.7 (188.7–188.7) | 0.059 (0.057–0.061) | 1, 1, 1 | 2.64% |
| kernels | 16 ms (15–21) | **23 ms** (21–25) | **23 ms** (22–25) | **238.1** (238.1–238.1) | **0.052** (0.051–0.053) | 1, 1, 1 | 1.01% |

- The protocol is the 2026-10-02 session's (below): three prompts five times
  each after one warm-up request, then 15 probe requests on the same server;
  WER with Qwen3-ASR-1.7B over the 15 timed replies.
- Every arm's 15 timed texts and audio files are byte-identical to the
  2026-10-02 session's, so WER and its split by prompt are unchanged
  (stock 4.4%, 0.0%, 0.3%; Marlin 6.3%, 0.0%, 0.2%; kernels 2.2%, 0.0%,
  0.6%).
- The Marlin arm's TTFT and TTFA are 2–3 ms lower than in 2026-10-02, when a
  sort after vLLM's alignment made it deterministic.
- Text speed is the inverse of the server's pace between streamed text
  deltas: 15.6, 5.3 and 4.2 ms a token at the medians.

| Arm | vLLM-Omni commit | Deploy config | CUDA MPS | Switches |
|---|---|---|---|---|
| stock | `69de153f` | decode-mk's `control/qwen3omni/production.yaml` | no | none |
| deterministic Marlin | `1f2cae18` | `showcase/qwen3omni/control_marlin.yaml` | yes | `VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL=1` `VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS=96` `VLLM_OMNI_CODE2WAV_STREAM_GRAPHS=1` `VLLM_OMNI_FRAME0=1` `VLLM_OMNI_EVENT_DRIVEN_ORCH=1` |
| kernels | `1f2cae18` | `showcase/qwen3omni/kernels.yaml` | yes | `VLLM_OMNI_THINKER_MEGAKERNEL=1` `VLLM_OMNI_THINKER_MEGAKERNEL_CTAS=64` `VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL=1` `VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS=128` `VLLM_OMNI_TALKER_MEGAKERNEL=1` `VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL=1` `VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS=96` `VLLM_OMNI_CODE2WAV_STREAM_GRAPHS=1` `VLLM_OMNI_CODE2WAV_COMPILE=1` `VLLM_OMNI_FRAME0=1` `VLLM_OMNI_EARLY_CHUNK=1` `VLLM_OMNI_FRAME0_AUDIO=1` `VLLM_OMNI_TALKER_PREP=1` `VLLM_OMNI_FAST_POLL=1` `VLLM_OMNI_THINKER_YIELD=1` `VLLM_OMNI_TALKER_PREPREFILL=1` `VLLM_OMNI_EVENT_DRIVEN_ORCH=1` |

Each arm's stage 1 loaded the same talker compile artifact as in 2026-10-02:

| Arm | Artifact | Files | sha256 of its files |
|---|---|---:|---|
| stock | `2ba9d68299c4757b…` | 45 | `daf25361c036de0c…` |
| deterministic Marlin | `8b3fbc760decae24…` | 1505 | `f1a3b750695e5131…` |
| kernels | `8b3fbc760decae24…` | 1505 | `f1a3b750695e5131…` |

Loading `8b3fbc76…` again logs the four `Cubin file saved by TritonBundler
not found` warnings in both arms.

## 2026-10-02

The first session, at `379804a6`, before the deterministic Marlin MoE fix
moved into vLLM PR 48032's alignment kernels. Its texts and audio are
byte-identical to the 2026-10-06 session's.

Every column but WER is median (min–max).

| Arm | TTFT | TTFA | TTFA, probes | Text, tok/s | RTF | Distinct texts per prompt | WER |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock | 57 ms (49–60) | 213 ms (206–220) | 214 ms (206–215) | 64.1 (64.1–64.9) | 0.112 (0.104–0.114) | 1, 1, 1 | 1.59% |
| deterministic Marlin | 18 ms (17–19) | 42 ms (39–43) | 41 ms (40–43) | 185.2 (185.2–188.7) | 0.058 (0.057–0.060) | 1, 1, 1 | 2.64% |
| kernels | 16 ms (15–17) | **23 ms** (21–24) | **23 ms** (21–24) | **238.1** (238.1–238.1) | **0.052** (0.052–0.053) | 1, 1, 1 | 1.01% |

- Each arm is three prompts five times each after one warm-up request, then
  15 probe requests: the same three prompts cycled five times on the same
  server. The columns are over the first 15 replies, except the probe column.
- TTFT of the Marlin and kernel arms overlaps: their ranges meet at 17 ms.
- Text speed is the inverse of the server's pace between streamed text
  deltas, one token each: 15.6, 5.4 and 4.2 ms a token at the medians.
- Distinct texts counts each prompt's replies over all 20 requests (timed and
  probes). Every arm gives one text per prompt.
- WER scores each timed reply's audio against that reply's own text with
  Qwen3-ASR-1.7B, over all 15 replies. Each prompt repeats one text, so a
  prompt's errors count five times. By prompt:

| Arm | Prompt 0 | Prompt 1 | Prompt 2 |
|---|---:|---:|---:|
| stock | 4.4% | 0.0% | 0.3% |
| deterministic Marlin | 6.3% | 0.0% | 0.2% |
| kernels | 2.2% | 0.0% | 0.6% |

### Arms

Every arm serves `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` (snapshot
`d6e1eff8`) with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

| Arm | vLLM-Omni commit | Deploy config | CUDA MPS | Switches |
|---|---|---|---|---|
| stock | `69de153f` | decode-mk's `control/qwen3omni/production.yaml` | no | none |
| deterministic Marlin | `379804a6` | `showcase/qwen3omni/control_marlin.yaml` | yes | `VLLM_OMNI_DETERMINISTIC_MARLIN=1` `VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL=1` `VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS=96` `VLLM_OMNI_CODE2WAV_STREAM_GRAPHS=1` `VLLM_OMNI_FRAME0=1` `VLLM_OMNI_EVENT_DRIVEN_ORCH=1` |
| kernels | `379804a6` | `showcase/qwen3omni/kernels.yaml` | yes | `VLLM_OMNI_THINKER_MEGAKERNEL=1` `VLLM_OMNI_THINKER_MEGAKERNEL_CTAS=64` `VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL=1` `VLLM_OMNI_THINKER_MEGAKERNEL_PREFILL_CTAS=128` `VLLM_OMNI_TALKER_MEGAKERNEL=1` `VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL=1` `VLLM_OMNI_CODE_PREDICTOR_MEGAKERNEL_CTAS=96` `VLLM_OMNI_DETERMINISTIC_MARLIN=1` `VLLM_OMNI_CODE2WAV_STREAM_GRAPHS=1` `VLLM_OMNI_CODE2WAV_COMPILE=1` `VLLM_OMNI_FRAME0=1` `VLLM_OMNI_EARLY_CHUNK=1` `VLLM_OMNI_FRAME0_AUDIO=1` `VLLM_OMNI_TALKER_PREP=1` `VLLM_OMNI_FAST_POLL=1` `VLLM_OMNI_THINKER_YIELD=1` `VLLM_OMNI_TALKER_PREPREFILL=1` `VLLM_OMNI_EVENT_DRIVEN_ORCH=1` |

- `69de153f`'s own deploy config, `vllm_omni/deploy/qwen3_omni_moe.yaml`, is
  written for two H100s. `production.yaml` (decode-mk `d48ef71a`) is that
  config with every stage on one GPU and only the memory, length and batch
  limits it takes to fit 32 GB.
- The stock arm runs the triton thinker; the Marlin arm runs Marlin; the
  kernel arm runs its own thinker kernels on the checkpoint's packing.
- `VLLM_OMNI_DETERMINISTIC_MARLIN` has since been removed: Marlin MoE
  alignment is now always deterministic, through vLLM PR #48032's kernels
  instead of the sort this table's runs used
  ([#15](https://github.com/fractalyze/vllm-omni/pull/15)). To serve an arm
  now, drop that switch.

### Talker compile artifact

The talker's prompt runs on vLLM's compiled forward, and two compile
artifacts can round differently (see the README's talker notes). Each arm's
stage 1 loaded this artifact from `~/.cache/vllm/torch_compile_cache/torch_aot_compile/`:

| Arm | Artifact | Files | sha256 of its files |
|---|---|---:|---|
| stock | `2ba9d68299c4757b…` | 45 | `daf25361c036de0c…` |
| deterministic Marlin | `8b3fbc760decae24…` | 1505 | `f1a3b750695e5131…` |
| kernels | `8b3fbc760decae24…` | 1505 | `f1a3b750695e5131…` |

- The two `379804a6` arms load the same compiled talker; the stock arm
  compiles a different talker forward.
- Loading `8b3fbc76…` logs four `Cubin file saved by TritonBundler not found`
  warnings: four of its autotuned kernels reload without their saved cubin.
  The warnings are the same in both arms.
- The digest is `find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum`
  run inside the artifact directory.

## Environment

| | |
|---|---|
| GPU | RTX 5090, driver 595.84 |
| venv | `vllm==0.30.0`, `torch==2.13.0+cu132`, `nvidia-cudnn-cu13==9.23.2.1` |
| Harness | decode-mk `d48ef71a`, `control/qwen3omni/bench.py` and `serve.sh` |
| ASR | `Qwen/Qwen3-ASR-1.7B` (snapshot `7278e1e7`) on `vllm serve`, `/v1/audio/transcriptions` |
| WER text normalization | openai-whisper's `EnglishTextNormalizer`, as in SGLang-Omni `b97f66d9` `benchmarks/tasks/asr.py` |

## Reproduce

From one clone of this branch, with the [install](README.md#install) venv and
a checkout of decode-mk at `d48ef71a` (`$DMK`):

```bash
git worktree add --detach ../stock 69de153f
export VLLM_OMNI_VENV=$PWD/.venv S2MK_RUN_DIR=/tmp/mps-$USER
uv pip install -e ../stock --no-deps     # stock arm; `-e .` for the other two
VLLM_OMNI=../stock $DMK/control/qwen3omni/serve.sh start \
  $DMK/control/qwen3omni/production.yaml stock/server.log
python $DMK/control/qwen3omni/bench.py time --repeats 5 --out stock
$DMK/control/qwen3omni/serve.sh stop
```

- The probes run on the same server right after `bench.py time`:

  ```bash
  PYTHONPATH=$DMK/control/qwen3omni python -c '
  import bench, statistics
  ttfa = [bench.request(8091, p, bench.SEED)["ttfa"] * 1e3
          for _ in range(5) for p in bench.PROMPTS]
  print(f"probe ttfa: {statistics.median(ttfa):.0f} ms ({min(ttfa):.0f}-{max(ttfa):.0f})")'
  ```

- The deterministic Marlin and kernel arms set their switches (the latest
  session's table) in the environment of `serve.sh start` and pass it
  `--mps`.
- Keep `S2MK_RUN_DIR` short: `serve.sh` puts the MPS control socket under it,
  and a Unix socket path over 108 bytes stops the daemon.
- `bench.py wer --out <arm> --asr-url http://127.0.0.1:8000` scores an arm
  once `vllm serve Qwen/Qwen3-ASR-1.7B --port 8000` answers; it imports
  `benchmarks.tasks.asr.normalize_text` from SGLang-Omni.
