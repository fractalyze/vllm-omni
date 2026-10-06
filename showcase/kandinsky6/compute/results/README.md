# Race and profile records

The JSON each run in `../` writes: every arm's samples, the accuracy numbers,
the shapes, the library versions, which GPU locks were held and what
`nvidia-smi` showed. Checked in so a table in
[`../../measurements.md`](../../measurements.md) can be audited from the tree
alone.

`complete: false` (or an absent `complete` key, from a run before that field
existed) means the run was cut off partway through its roles — the GPU here is
shared. The roles present are still valid; the ones missing were never run.

Only post-fix runs are kept. The block results taken before the RoPE and
weight-initialization fixes (`k6c-p01`, `k6c-p01b`, `k6c-f01`, `k6c-f01b`) are
deliberately **not** here: keeping a superseded number next to a current one
invites it to be quoted. They are in the world-model vault's `raw/k6c-bs3`,
which is append-only, so the record of what was measured and corrected
survives without putting the wrong numbers in this tree.

| file | what it is |
|---|---|
| `k6c-a01-attn-race-w1.json` | the attention backend race, all five call sites |
| `k6c-a01b-attn-race-w1-fa4.json` | the headline role again, a second session |
| `k6c-b04-baseline-w1.json` | the four-arm baseline table |
| `k6c-g02-tuned-numerics-w1.json` | eager vs the three compile modes, with the output check |
| `k6c-b03-shipped-numerics-w1.json` | the same for the shipped attention arm |
| `k6c-p05-split-w1-*.json` | the kernel-category split, one per stack |
