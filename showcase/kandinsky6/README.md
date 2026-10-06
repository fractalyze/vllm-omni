# Kandinsky 6 Pro on one RTX 5090

Kandinsky 6.0 **Pro-distill** (29B joint video+audio DiT) serving W1 --
864x480, 121 frames at 24 fps, 10 PiFlow steps, guidance 1.0, audio on -- on a
single RTX 5090 (32 GB, 575 W) in a host with 60 GB of RAM, from this branch of
vLLM-Omni.

Nothing about that fits as shipped. The DiT is **56.14 GiB in BF16**, which
exceeds the board and the host RAM together, and upstream's recipe needs an
H100 and ~90 GB of free host memory for `--enable-cpu-offload`.
[PLAN.md](PLAN.md) states the goal and the rules; [measurements.md](measurements.md)
has every number and how it was taken.

## Install

From a clone of this branch:

```bash
uv venv --python 3.12 /data/jooman/k6/venv
VIRTUAL_ENV=/data/jooman/k6/venv uv pip install setuptools_scm
VIRTUAL_ENV=/data/jooman/k6/venv uv pip install vllm==0.30.0 --torch-backend=auto
VIRTUAL_ENV=/data/jooman/k6/venv uv pip install -e .
```

SageAttention is optional and only needed for the attention arms:

```bash
VIRTUAL_ENV=/data/jooman/k6/venv uv pip install sageattention==2.2.0
```

Put the Hugging Face cache somewhere with room for a 60 GB checkpoint:
`export HF_HOME=/data/jooman/hf`.

## Serve

Each script is one arm, in the sense `bench/serve.py` uses: a checkpoint, a set
of flags and a set of environment switches, complete enough to rebuild from the
ledger alone.

| script | weights | where they live | notes |
|---|---|---|---|
| `serve/serve_pro_bf16_ref.sh` | BF16, exact | streamed from the mmapped checkpoint | the quality gate's reference; slowest |
| `serve/serve_pro_fp8.sh` | FP8 E4M3, per-tensor scales | pinned host memory, staged per block | fails the user's quality gate |
| `serve/serve_pro_int8.sh` | INT8, per-output-row scales | quantized at load from the BF16 checkpoint | **does not run on sm_120** — kept because the format is the right one and the kernel is the only thing missing |

The INT8 script is in that table as a signpost, not an option: INT8 at per-row
scales costs 2.9x less weight error than FP8 at the same one byte per weight,
and consumer Blackwell has no INT8 GEMM to spend it on (CUTLASS's
`dispatch_scaled_mm`: "Int8 not supported on SM120"). On a datacenter Blackwell
or an Ada card it is the arm to try first.

The server is ready when `curl -sf localhost:$PORT/health` succeeds. Add an
attention arm to any of them:

```bash
serve/serve_pro_int8.sh --diffusion-attention-config "$(cat compute/arms/tuned.json)"
```

`compute/arms/*.json` are per-role attention configurations. The roles matter:
one backend for the whole model is wrong whichever one it is, because the
ranking inverts between the 50,220-query visual self-attention and the
218-query audio self-attention (measurements.md).

## Measuring

```bash
# One arm's W1 outputs for a whole prompt set, as MP4s, from a real served request.
compute/gate_pro.py --arm compute/arms/tuned.json --name tuned --out-dir runs/tuned

# G1 (the user's gate) and G2 (floor-relative), against the BF16 reference.
compute/gate_pro_score.py --arm-dir runs/tuned --reference-dir /data/jooman/k6/ref/setA \
    --g2-floor-mean <floor> --g2-floor-max <floor>

# G3 (distributional) and a contact sheet.
compute/clip_gate.py --arm-dir runs/tuned --reference-dir /data/jooman/k6/ref/setA \
    --contact-sheet runs/tuned/sheet.jpg

# Two arms interleaved A B B A in one session, which is how every speed claim here was taken.
bench/abba.py --control /data/jooman/k6/arms/control-cudnn.json \
              --candidate /data/jooman/k6/arms/int8-sage2.json --repeats 3
```

Every timed run holds all of the host's GPU lock files and records what
`nvidia-smi` showed; a run with a foreign compute process on the board is
marked contaminated rather than reported.

## What this workload turned out to be

Four things decided every arm in [measurements.md](measurements.md), and they
are worth knowing before reading any number here.

**The DiT does not fit anywhere.** 56.14 GiB in BF16 -- 60 visual blocks of
0.899 GiB, four text blocks, embeddings and two output heads -- against a 32 GB
board and 60 GB of host RAM. It runs from the mmapped checkpoint and every one
of W1's 10 steps re-reads all of it, at a measured 1.67 GB/s off NVMe with a
page-cache hit rate near 0.6.

**No sub-BF16 weight format on this GPU can pass a tight perceptual gate.** FP8
E4M3's weight error is 2.6% *at any scale granularity* -- per-tensor 0.02645,
per-output-row 0.02643 -- because the format carries its own 4-bit exponent, so
a finer scale slides the matrix along the exponent ladder while the step stays
at the 3-bit mantissa. INT8 at per-row scales would cost 0.00908, 2.9x less, and
sm_120 has no INT8 GEMM: CUTLASS's `dispatch_scaled_mm` refuses with "Int8 not
supported on SM120". So the weights stay BF16 and precision is not the lever.

**Attention is, and it is worth a third of the request.** The platform's cuDNN
attention runs 21.9 s/step against a stream worth about 15 s; SageAttention2
takes the arm to 165.9 s from 231.8 s. But Sage2 everywhere costs LPIPS 0.1858
against the BF16 reference, so the question becomes *which blocks* approximate,
which is what `AttentionSpec.layers` exists for.

**And the harness moves more than most of the arms do.** Running the same BF16
checkpoint compiled instead of eager changes LPIPS by 0.1455 on the set mean and
**0.4160 on the worst frame** -- Inductor picks kernels by measured latency, so
two compiled processes are not identical while two eager ones are. That is
larger than the gate those arms were being judged against, and it is why three
bars are reported instead of one.

## The quality gate

Three questions, reported side by side, because they disagree and the
disagreement is informative:

- **G1** is the user's bar: per prompt set, LPIPS mean <= 0.15 **and** max <=
  0.25 against the same checkpoint's BF16 output at the same seed.
- **G2** is floor-relative: within 1.25x the pipeline's own numerical floor.
  It exists because rerunning *the same configuration* in a fresh process
  already costs LPIPS 0.0272 mean / 0.0636 max on this pipeline, so an absolute
  bar below that movement measures noise rather than the arm. **Coordinator-chosen,
  pending the user.**
- **G3** is distributional: CLIP text-video agreement within 2% of the
  reference's, plus a contact sheet for a human to look at.

Two disjoint prompt sets of nine prompts each, every set covering faces,
rendered text, fast motion and a sharp sound event -- because an FP8 recipe in
the vault passed one 8-prompt set at LPIPS 0.034 and failed another at 0.162.
