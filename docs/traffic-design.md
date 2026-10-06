# Traffic Design

## 1. Goal

`traffic` is intended to simulate opposite-side driving by mirroring the full scene left-right in ego coordinates.

The target effect is:

- swap the left-right layout of the scene
- preserve semantic identities
- preserve local occupancy geometry
- mirror the spatial arrangement and every motion field that is expressed in that arrangement

This corruption is different from `semantic`, `hole`, and `dropout`:

- it does not change semantic labels
- it does not delete occupancy by itself
- it applies a deterministic geometric remap

## 2. Core Rule

The default rule is a left-right reflection in ego coordinates. This is not occupancy-only.

Given occupancy indexed as `[x, y, z]`, mirror along the lateral `y` axis:

```text
y' = (Ny - 1) - y
```

In tensor form this is a flip of the second dimension:

```python
mirrored = semantics[:, ::-1, :]
```

The same mirror should be applied to occupancy tensors:

- `semantics`
- `mask_lidar`
- `mask_camera`

The same reflection must also be applied to metadata consumed by temporal/world models:

- history/current/future ego-to-ego transforms
- ego-to-global pose matrices, translations, and rotations
- `can_bus`
- ego future trajectories, ego velocity/yaw-rate features, and left/right command labels
- agent boxes, velocities, future trajectories, future yaw deltas, and local agent features

## 3. Severity

`traffic` should **not** be split into `easy / mid / hard`.

This corruption is a deterministic scene-level remap rather than a random corruption with natural strength levels.

Therefore the benchmark should use exactly one traffic setting:

- mirror the full occupancy frame
- use one compatibility protocol alias:
  - `traffic_all_frame_H4_F6_val_backbone`

This alias should be interpreted as:

- history mirrored
- current mirrored
- target mirrored
- future targets mirrored
- ego motion and agent motion mirrored

It is not a "history-only" or "occupancy-only" corruption.

## 4. Storage

`traffic` may be materialized per frame because the mirror transform itself is deterministic and frame-independent.

```text
data/OccStress/
  occ/
    traffic/
      <scene>/<sample_token>/labels.npz

  events/
    traffic/
      <scene>/<sample_token>.json

  meta/
    traffic_summary.json
```

## 5. Event Payload

Each event file should store:

- `type`
- `scene_name`
- `sample_token`
- `mirror_axis`
- `operation`
- `changed_voxels_total`
- `class_histogram`
- `occ_path_clean`
- `occ_path_corrupt`
