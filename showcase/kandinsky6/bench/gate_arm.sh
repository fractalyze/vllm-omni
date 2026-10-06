#!/bin/bash
# Gate one arm on one prompt set: start its server under the memory cap, generate
# every prompt, score against the BF16 reference, stop the server.
#
#   gate_arm.sh <arm-name> <setA|setB> [extra vllm serve args...]
#
# The arm's environment switches (VLLM_OMNI_K6_EXACT_ATTN_STEPS, VLLM_OMNI_SAGE_KERNEL,
# ...) are taken from the caller's environment. Outputs: /data/jooman/k6/arms/<arm>/<set>/
# (MP4s, manifest.json with per-request wall times, gate.json, server.log).
set -euo pipefail
ARM=$1; SET=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/data/jooman/k6/arms/$ARM/$SET
REF=${K6_REF_ROOT:-/data/jooman/k6/ref}/$SET
PY=/data/jooman/k6/venv/bin/python
mkdir -p "$OUT"
env | grep -E '^(VLLM_OMNI_K6_|VLLM_OMNI_SAGE_|K6_CKPT)' > "$OUT/arm.env" || true
echo "$*" > "$OUT/arm.args"
K6_MEMMAX=${K6_MEMMAX:-44G} K6_CKPT=${K6_CKPT:-/data/jooman/k6/ckpt/pro-distill-fp8-min} \
  setsid "$HERE/../serve/run_capped.sh" "$HERE/../serve/serve_pro_fp8.sh" "$@" > "$OUT/server.log" 2>&1 < /dev/null &
LAUNCHER=$!
# Signal the launcher's whole session: the server process only takes the name
# "vllm" after exec, so a name check right after launch reports a live server as
# dead, and killing by name would also hit servers this script did not start.
cleanup() {
  kill -TERM -- "-$LAUNCHER" 2>/dev/null || true
  for _ in $(seq 90); do kill -0 "$LAUNCHER" 2>/dev/null || break; sleep 2; done
  for _ in $(seq 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt 1024 ] && break; sleep 2
  done
}
trap cleanup EXIT
until curl -sf localhost:8091/health >/dev/null 2>&1; do
  kill -0 "$LAUNCHER" 2>/dev/null || { echo "server died; see $OUT/server.log" >&2; exit 1; }
  sleep 3
done
"$PY" "$HERE/make_refs.py" --base-url http://127.0.0.1:8091 --prompts "$HERE/prompts/$SET.json" --out "$OUT" --label "$ARM"
"$PY" "$HERE/quality.py" "$REF" "$OUT" --prompts "$HERE/prompts/$SET.json" --out "$OUT/gate.json" 2>/dev/null \
  | "$PY" -c "import json,sys; r=json.load(sys.stdin); print(json.dumps({k: r[k] for k in ('lpips_mean','lpips_max','worst_prompt','gate_pass','tier','psnr_mean','logmel_l1_mean','si_sdr_db_min')}), json.dumps(r['per_prompt']))"
"$PY" -c "import json,statistics; m=json.load(open('$OUT/manifest.json'))['items']; w=[v['request_wall_s'] for v in m.values()]; print('wall_s median', round(statistics.median(w),2), 'min', round(min(w),2), 'max', round(max(w),2), 'n', len(w))"
