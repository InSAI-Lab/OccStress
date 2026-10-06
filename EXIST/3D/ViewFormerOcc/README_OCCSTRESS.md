# OccStress Adapter Notes

This directory keeps the original ViewFormerOcc method layout.

Use the shared OccStress paths:

```bash
export OCCSTRESS_CODE_ROOT=$(pwd)/../../..
export OCCSTRESS_DATA_ROOT=$OCCSTRESS_CODE_ROOT/data/OccStress
```

Method-specific evaluation scripts should read protocols from `data/OccStress/protocols/...` and write occupancy predictions as `labels.npz` under `data/OccStress/occ/...`.

See `../../../docs/3d-method-adapters.md` for the common 3D adapter contract.
