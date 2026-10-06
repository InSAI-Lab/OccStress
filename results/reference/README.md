# Reference Results

Optional location for compact, audited JSON/CSV summaries and their provenance index.
No model scores are populated by the directory migration. Data, weights, raw
voxel predictions, token caches and scene-level research intermediates do not
belong here.

Populating this directory or rerunning the paper is not a code-release gate.
Small private GPU validation receipts are maintained separately and are not
paper reference scores. The paper remains the reference for reported results.

Every future entry must identify its dataset, model/checkpoint, protocol coverage,
anchor count, horizon convention and metric policy. Keep historical/native and
recomputed present-class results separate. Synthetic demonstrations are under
`examples/`, not reference results.
