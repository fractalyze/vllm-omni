#!/usr/bin/env bash
# Can a wider exact band pull set B's two failing prompts under the limit?
#
# `sage2-mid` (12 exact blocks) passes the floor-relative gate on set A at 1.06x
# and fails it on set B at 1.44x of that set's max. The failure is one prompt,
# b6-train-platform at 0.4989 against a 0.4345 limit, with b3-storefront-sign
# next; both are text-plus-motion, the category the schedule helps least.
#
# The band is a measured dial -- 12 exact blocks buy 24% of the error, 6 buy
# nothing -- so the lever the data points at is more exact blocks, not a
# different kernel. `sage2-narrow` is 24 exact blocks, which should cost about
# 12 s a request on top of 177.7 s.
#
# Screened on exactly the two prompts that fail, because that is the question.
# If b6 does not come under 0.4345 here, no band will, and the honest answer is
# that nothing passes both sets.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
OUT="${K6_GATE_PRO:-/data/jooman/k6/results/gate-pro}"
echo "=== $(date +%H:%M:%S) sage2-narrow on set B's two failing prompts"
"$PY" "$HERE/gate_pro.py" \
    --arm "$HERE/arms/sage2-narrow.json" --name sage2-narrow-hard2 --offload dlo-mmap \
    --prompts /data/jooman/k6/results/setB-hard2.json --reference-manifest /nonexistent \
    --no-locks --out-dir "$OUT/sage2-narrow-hard2"
status=$?
echo "=== $(date +%H:%M:%S) exit $status"
exit "$status"
