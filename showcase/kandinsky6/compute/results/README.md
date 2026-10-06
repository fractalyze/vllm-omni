# Race and profile records

The JSON each run in `../` writes: every arm's samples, the accuracy numbers,
the shapes, the library versions, which GPU locks were held and what
`nvidia-smi` showed. Checked in so a table in
[`../../measurements.md`](../../measurements.md) can be audited from the tree
alone.

`complete: false` (or an absent `complete` key, from a run before that field
existed) means the run was cut off partway through its roles — the GPU here is
shared. The roles present are still valid; the ones missing were never run.
