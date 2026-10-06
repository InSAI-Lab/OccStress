# CVT-Occ Waymo Integration

Formal config: `projects/configs/occstress/cvtocc_waymo_local_2hz.py`.
Paths use `OCCSTRESS_CODE_ROOT`, `OCCSTRESS_DATA_ROOT`,
`OCCSTRESS_EXTERNAL_ROOT` and `CVTOCC_RUNTIME_ROOT`.
Optional `CVTOCC_SENSOR_ROOT`, `CVTOCC_FRAME_INDEX_ROOT` and `CVTOCC_GT_ROOT`
select existing native input views without moving raw data. The compact frame
index and model annotation are separate prerequisites, not occupancy payloads.

The camera source has a native seven-state perception queue; downstream
forecasters still use their declared H4/current windows. Source queue length
does not redefine the OccStress position-sweep axis.

See `docs/UPSTREAM_EXPORTS.md` and `docs/ENVIRONMENTS.md` at the release root.
Use only the dedicated CVT environment plus its declared vendored mmdet3d base.
The inherited runtime base is `cvtocc_waymo_runtime.py`; defaults are portable.
