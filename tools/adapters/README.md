# OccStress Adapters

OccStress adapters should be thin path/protocol layers around the original third-party method code.

Recommended contract:

- `OCCSTRESS_CODE_ROOT` points to this repository.
- `OCCSTRESS_DATA_ROOT` points to the dataset root, defaulting to `data/OccStress`.
- Protocol records store relative paths whenever possible.
- Dataloaders resolve `data/OccStress/...` paths through `OCCSTRESS_CODE_ROOT` or `OCCSTRESS_DATA_ROOT`.
- Targets must remain clean unless the protocol explicitly defines a target-side transformation.

Use `occstress_paths.py` for shared path construction in new adapters.
