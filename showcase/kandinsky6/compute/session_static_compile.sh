#!/usr/bin/env bash
# Lever (c): does specialising the compiler on W1's shapes buy anything?
#
# The served pipeline is already compiled -- the log says "Regional compilation
# applied to 188 module(s) for repeated blocks" -- so (c) is not "turn compile
# on". It is compiled with `dynamic=True`, which is the right default for a
# server that sees many geometries and the wrong one for these arms, which only
# ever run W1: 31 x 30 x 54 = 50,220 visual tokens, every request, forever.
# `--diffusion-compile-dynamic false` lets Inductor specialise on that instead
# of emitting shape-generic kernels.
#
# Two arms, two prompts each, so the comparison is against the same attention
# configuration and the only difference is the compiler's shape policy:
#
#   static-sage2-mid   the candidate arm with dynamic=False
#   static-control     the reference configuration with dynamic=False
#
# The second one is not optional. Changing compilation changes which kernels
# run, and on this pipeline that is worth LPIPS 0.1455 against a
# differently-compiled reference -- so a static arm scored against a dynamic
# reference would read as a quality regression that is really a kernel-choice
# difference. The static control is the reference a static arm has to be scored
# against.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@"; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

run static-sage2-mid "$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/sage2-mid.json" --name static-sage2-mid --offload dlo-mmap \
    --compile-dynamic false --limit 2 --no-locks --out-dir "$OUT/static-sage2-mid"

run static-control "$PY" "$HERE/gate_pro.py" \
    --arm shipped --name static-control --offload dlo-mmap \
    --compile-dynamic false --limit 2 --no-locks --out-dir "$OUT/static-control"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
