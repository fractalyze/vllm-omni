#!/usr/bin/env bash
# The definitive block measurements: correct RoPE tables AND initialized
# weights.
#
# vLLM's ColumnParallelLinear/RowParallelLinear allocate with torch.empty and
# expect a checkpoint loader, so every projection in the synthetic block was
# returning zero. Dense GEMM and attention take the same time on zeros, which
# is why the timings looked stable -- but SageAttention's per-block scale is
# the block maximum, so a zero query gave a zero scale and NaN out. Every
# earlier block number is superseded by this run.
#
# The numerics check runs first and its control is a `tuned` arm, so all four
# tuned arms share one module and are actually comparable.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_RESULTS:-/data/jooman/k6/results}"
TUNED="$HERE/arms/tuned.json"
failures=0
run() {
    local name="$1"; shift
    echo "=== $(date +%H:%M:%S) $name"
    if "$@" --no-locks; then echo "=== $(date +%H:%M:%S) $name ok"
    else echo "=== $(date +%H:%M:%S) $name FAILED (exit $?)"; failures=$((failures + 1)); fi
}

# Do the CUDA-graph arms compute the right answer? All arms share one module.
run k6c-g02-tuned-numerics "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused \
    --compare-arm "tuned-eager=$TUNED@eager" \
    --compare-arm "tuned-default=$TUNED@default" \
    --compare-arm "tuned-reduce-overhead=$TUNED@reduce-overhead" \
    --compare-arm "tuned-max-autotune=$TUNED@max-autotune" \
    --check-numerics --repeats 3 --rounds 2 --warmups 3 --profile-iters 0 \
    --json "$OUT/k6c-g02-tuned-numerics-w1.json"

# The same question for the shipped attention arm, so the compiled-vs-eager
# default is checked too.
run k6c-b03-shipped-numerics "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused \
    --compare-arm "shipped-eager=default@eager" \
    --compare-arm "shipped=default@default" \
    --check-numerics --repeats 3 --rounds 2 --warmups 3 --profile-iters 0 \
    --json "$OUT/k6c-b03-shipped-numerics-w1.json"

# The baseline table, cross-config, timing only (numerics cannot cross
# attention configs: the modules differ).
run k6c-b04-baseline "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused \
    --compare-arm "shipped=default@default" \
    --compare-arm "shipped-eager=default@eager" \
    --compare-arm "tuned=$TUNED@default" \
    --compare-arm "tuned-max-autotune=$TUNED@max-autotune" \
    --repeats 3 --rounds 2 --warmups 3 --profile-iters 0 \
    --json "$OUT/k6c-b04-baseline-w1.json"

for arm in "shipped:default" "tuned:$TUNED"; do
    label="${arm%%:*}"; cfgfile="${arm##*:}"
    extra=()
    [ "$cfgfile" != "default" ] && extra=(--attention-config "$cfgfile")
    run "k6c-p05-split-$label" "$PY" "$HERE/block_profile.py" \
        --config pro --geometry w1 --target fused "${extra[@]}" --compile default \
        --repeats 5 --warmups 3 --profile-iters 3 \
        --json "$OUT/k6c-p05-split-w1-$label.json"
done

run k6c-p05-split-tuned-max "$PY" "$HERE/block_profile.py" \
    --config pro --geometry w1 --target fused --attention-config "$TUNED" --compile max-autotune \
    --repeats 5 --warmups 3 --profile-iters 3 \
    --json "$OUT/k6c-p05-split-w1-tuned-max-autotune.json"

echo "=== $(date +%H:%M:%S) session done, $failures failure(s)"
exit "$failures"
