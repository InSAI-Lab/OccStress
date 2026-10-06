# Dataset Mounts

Dataset payloads are hosted separately on
[Hugging Face](https://huggingface.co/datasets/insailab/OccStress).
Set `OCCSTRESS_DATA_ROOT` to one shared directory; the default is `data/OccStress`.

```text
OccStress/
  protocols/
    manual/OccStress-nuScenes/       # Also OccStress-Waymo and OccStress-CARLA
    upstream/OccStress-nuScenes/
    position_sweep/OccStress-nuScenes/
  occ/
    manual/OccStress-nuScenes/
    upstream/OccStress-nuScenes/
  events/manual/OccStress-nuScenes/
  meta/OccStress-nuScenes/
  external/OccStress-nuScenes/       # User-supplied dependencies, not payloads
```

The same dataset namespaces apply to every track. Protocol references are
relative to this shared root, never to the author's filesystem.
`OCCSTRESS_EXTERNAL_ROOT` can place external dependencies on another disk.

See [download selection](../docs/RESOURCES.md),
[the full layout](../docs/DATASETS.md) and
[external preparation](../docs/EXTERNAL_DATA_PREPARATION.md).
