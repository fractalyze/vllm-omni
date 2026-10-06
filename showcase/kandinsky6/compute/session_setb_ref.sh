#!/usr/bin/env bash
# Set B's BF16 reference. Nobody has one, and the user's gate is per prompt set.
#
# A reference is a gate run on exact weights with the platform's own attention,
# which is what `--arm shipped --offload dlo-mmap` is -- the same server
# configuration that produced Track M's set A references, at the same seed 42
# and the same W1 geometry. It writes its manifest in the reference runner's
# `items` shape, so a candidate run can read its seeds and geometry back and the
# comparison is same-sample by construction rather than by assumption.
#
# About 40 minutes: nine prompts at the reference configuration's 232 s. It is
# on the critical path for every arm's set B number, so it runs whether or not a
# candidate has been chosen yet.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${K6_PYTHON:-/data/jooman/k6/venv/bin/python}"
REF="${K6_SETB_REF:-/data/jooman/k6/ref/setB-bs3}"
echo "=== $(date +%H:%M:%S) setb-bf16-reference"
"$PY" "$HERE/gate_pro.py" \
    --arm shipped --name setb-bf16-reference --offload dlo-mmap \
    --prompts "$HERE/../bench/prompts/setB.json" --reference-manifest /nonexistent \
    --no-locks --out-dir "$REF"
status=$?
echo "=== $(date +%H:%M:%S) setb-bf16-reference exit $status"
exit "$status"
