# OccStress Benchmark Design

## Goal

The benchmark should evaluate **4D model robustness** under corrupted occupancy-state inputs while keeping target supervision clean.

The benchmark is split into two top-level tracks:

- `upstream`
- `manual`

This separation is intentional:

- `upstream` measures how 4D methods behave when fed occupancy predictions produced by a real upstream 3D model
- `manual` measures how 4D methods behave under controlled synthetic occupancy-state corruption

These two tracks should not be flattened into one mixed corruption namespace.

## Track Definition

### `upstream`

`upstream` uses **predicted occupancy** as the input state source.

Recommended subtracks:

- `camera_only`
- `pointcloud_fusion`

Recommended initial mapping:

- `upstream/camera_only/stcocc`
- `upstream/pointcloud_fusion/occfusion`

Interpretation:

- `camera_only` means the upstream 3D occupancy model uses only camera input
- `pointcloud_fusion` means the upstream 3D occupancy model uses point-cloud-aligned fusion input, typically camera and lidar jointly

If the corruption source is currently `nuScenes-C` camera corruption, both subtracks may still use the same 8 camera corruption families. The difference lies in the **upstream model modality**, not in the corruption family name.

### `manual`

`manual` uses clean occupancy plus hand-designed occupancy-state corruption.

Current corruption families:

- `semantic`
- `hole`
- `dropout`
- `misalignment`
- `traffic`

This is the existing `OccStress` direction and should remain as a separate track.

## Clean Reference Policy

`clean` should be kept in both tracks, but it should be treated as a **reference setting**, not as a corruption class.

Rules:

- keep `clean` for every upstream source
- keep `clean` for manual track
- do not include `clean` when computing corruption averages

Reason:

- without a clean reference, it becomes impossible to separate upstream model quality from 4D robustness
- `upstream` and `manual` use different references and should not share one global clean baseline

Recommended references:

- `upstream`: compare corrupted upstream inputs against the **same upstream model's clean prediction**
- `manual`: compare corrupted manual inputs against **clean GT occupancy**

## What Counts as One Setting

For `nuScenes-C`-style upstream evaluation, the practical unit is:

- `1 clean`
- `8 corruption families`
- `3 severities per family`

So the full upstream bundle is not just `1 + 8`, but:

- `1 + 8 x 3 = 25 settings`

At the reporting level, it is still fine to say:

- one clean reference
- eight corruption families

But protocol and result storage should keep the `easy/mid/hard` split explicit.

## Recommended Directory Layout

```text
data/OccStress/
  occ/
    upstream/
      camera_only/
        stcocc/
          clean/<scene>/<token>/labels.npz
          Brightness/easy/<scene>/<token>/labels.npz
          Brightness/mid/<scene>/<token>/labels.npz
          Brightness/hard/<scene>/<token>/labels.npz
          ...
      pointcloud_fusion/
        occfusion/
          clean/<scene>/<token>/labels.npz
          Brightness/easy/<scene>/<token>/labels.npz
          ...

    manual/
      semantic/easy/<scene>/<token>/labels.npz
      semantic/mid/<scene>/<token>/labels.npz
      semantic/hard/<scene>/<token>/labels.npz
      hole/easy/<scene>/<token>/labels.npz
      hole/mid/<scene>/<token>/labels.npz
      hole/hard/<scene>/<token>/labels.npz
      dropout/easy/<scene>/<token>/labels.npz
      dropout/mid/<scene>/<token>/labels.npz
      dropout/hard/<scene>/<token>/labels.npz
      traffic/<scene>/<token>/labels.npz

  protocols/
    upstream/
      camera_only/
        stcocc/
          clean/
            H4_F6_val_backbone.pkl
          Brightness/
            easy/
              current_H4_F6_val_backbone.pkl
              history_k1_H4_F6_val_backbone.pkl
              all_frame_H4_F6_val_backbone.pkl
            mid/
              ...
            hard/
              ...
          CameraCrash/
            easy/
              ...
          ...
      pointcloud_fusion/
        occfusion/
          clean/
            H4_F6_val_backbone.pkl
          Brightness/
            easy/
              current_H4_F6_val_backbone.pkl
              history_k1_H4_F6_val_backbone.pkl
              all_frame_H4_F6_val_backbone.pkl
            mid/
              ...
            hard/
              ...
          ...
    manual/
      clean/
        H4_F6_val_backbone.pkl
      semantic/
        easy/
          current_H4_F6_val_backbone.pkl
          history_k1_H4_F6_val_backbone.pkl
          all_frame_H4_F6_val_backbone.pkl
        mid/
          ...
        hard/
          ...
      hole/
        easy/
          ...
      dropout/
        easy/
          ...
      misalignment/
        easy/
          ...
      traffic/
        all_frame_H4_F6_val_backbone.pkl

  events/
    manual/
      semantic/
      hole/
      dropout/
      misalignment/
      traffic/

  meta/
    upstream/
      stcocc_camera_only_manifest.yaml
      occfusion_pointcloud_fusion_manifest.yaml
    manual/
      corruption_schema.yaml
      subset_info.yaml
```

## Protocol Naming

Recommended naming:

- upstream clean:
  - `protocols/upstream/camera_only/stcocc/clean/H4_F6_val_backbone.pkl`
- upstream corrupted:
  - `protocols/upstream/camera_only/stcocc/Brightness/easy/current_H4_F6_val_backbone.pkl`
  - `protocols/upstream/camera_only/stcocc/Brightness/easy/history_k1_H4_F6_val_backbone.pkl`
- manual corrupted:
  - `protocols/manual/semantic/easy/current_H4_F6_val_backbone.pkl`
  - `protocols/manual/dropout/hard/all_frame_H4_F6_val_backbone.pkl`

This keeps path-based discovery simple and avoids one large flat protocol directory.

## Implementation Notes

Current local script plan:

- manual migration and helper:
  - `scripts/migrate_occstress_manual_layout.py`
  - `scripts/occstress_layout.py`
- upstream single-protocol generator:
  - `scripts/build_upstream_occstress_protocol.py`
- STCOcc dump helpers:
  - `scripts/stcocc/run_dump_upstream_1gpu.slurm`
  - `scripts/stcocc/submit_dump_upstream_clean.sh`
  - `scripts/stcocc/submit_dump_upstream_nusc_c_all.sh`
  - `scripts/stcocc/create_info_subset.py`
- STCOcc camera-only upstream wrapper:
  - `scripts/stcocc/build_occstress_upstream_protocols_from_nuscc.sh`

Recommended upstream generation flow:

1. materialize per-frame upstream predictions under:
   - `data/OccStress/occ/upstream/camera_only/stcocc/clean/<scene>/<token>/labels.npz`
   - `data/OccStress/occ/upstream/camera_only/stcocc/<corruption>/<severity>/<scene>/<token>/labels.npz`
   - clean batch submit:
     - `bash scripts/stcocc/submit_dump_upstream_clean.sh`
   - nuScenes-C batch submit:
     - `bash scripts/stcocc/submit_dump_upstream_nusc_c_all.sh`
2. run `scripts/stcocc/build_occstress_upstream_protocols_from_nuscc.sh`
3. adapt 4D method configs to point at:
   - `protocols/upstream/camera_only/stcocc/...`

## Protocol Record Metadata

The current protocol record format can be reused. The following fields should be added or standardized:

```python
{
    "track": "upstream" | "manual",
    "subtrack": "camera_only" | "pointcloud_fusion" | None,
    "source_model": "stcocc" | "occfusion" | None,
    "reference_protocol": "clean/H4_F6_val_backbone",
    "corruption": {
        "type": "clean" | "Brightness" | "semantic" | "traffic" | ...,
        "severity": "clean" | "easy" | "mid" | "hard",
        "frame_protocol": "current" | "history_k1" | "all_frame"
    }
}
```

Practical interpretation:

- `manual` records usually point to clean GT target paths and manual-corrupted history/current input paths
- `upstream` records point to upstream prediction paths for history/current inputs, while target and future target paths remain clean

## Storage Contract

For compatibility with current `OccWorld` and `II-World` loaders:

- every frame should still be stored as `labels.npz`
- the file must contain at least `semantics`
- shape and class ids must match the clean occupancy convention exactly

Minimum requirement:

```python
np.savez(path, semantics=pred_semantics.astype(np.uint8))
```

This is enough for:

- `OccWorld`, which loads `label['semantics']`
- `II-World`, which loads `label['semantics']` and then builds OccStress tokenizer/world inputs from protocol-selected `occ_path`

## Evaluation Rules

Do not publish one single mixed robustness score across all tracks by default.

Recommended reporting:

### `upstream`

Report per source model:

- clean reference
- corruption average excluding clean
- delta vs that source model's clean reference
- retention ratio

Examples:

- `upstream/camera_only/stcocc`
- `upstream/pointcloud_fusion/occfusion`

### `manual`

Report:

- clean GT reference
- corruption average excluding clean
- delta vs clean GT reference

This keeps the benchmark interpretable:

- `upstream` answers: how well does the 4D model tolerate realistic upstream 3D prediction error?
- `manual` answers: how well does the 4D model tolerate controlled occupancy-state corruption?

## Recommended First Release

The first clean version of the benchmark should be:

- `upstream/camera_only/stcocc`
- `upstream/pointcloud_fusion/occfusion`
- `manual`

This avoids over-expanding the first release while still covering:

- one representative camera-only upstream source
- one representative pointcloud-fusion upstream source
- one explicit synthetic corruption track

## Final Recommendation

Use the following benchmark taxonomy:

- `upstream`
  - `camera_only`
  - `pointcloud_fusion`
- `manual`

Keep `clean` in every track as the reference setting.

Do not count `clean` in corruption averages.

Do not flatten `upstream` and `manual` into one unified corruption table.

This is the most defensible design if the benchmark is supposed to measure **4D robustness**, not just raw 3D occupancy corruption severity.
