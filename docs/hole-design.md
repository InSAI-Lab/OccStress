# Hole Design

## 1. Goal

`hole` is intended to simulate internal hollowness inside dynamic object occupancy.

The target effect is:

- keep the outer shell of the object
- remove part or all of the interior voxels
- produce a visible cavity inside the object support

This corruption is geometry-oriented and should not change semantic labels.

## 2. Scope

This version of `hole` should only operate on dynamic object supports.

Recommended current scope:

- `vehicle`
  - `car`
  - `truck`
  - `bus`
  - `trailer`
  - `construction_vehicle`

Static scene classes should not be processed by `hole`.

## 3. Object Support

The support should follow the same object-aware support rule already validated in `semantic`:

- build an instance support from annotation-aligned occupancy
- use `completed_instance + bbox_expand_voxels = 1`

This gives a reasonably complete voxel support for the object.

## 4. Inner-Void Rule

The core rule for `hole` is:

1. build the full object support
2. erode the support inward by one voxel using 6-neighborhood connectivity
3. treat the eroded result as the `interior`
4. carve the hole only inside that interior

This guarantees:

- the outermost voxel shell is preserved
- the cavity remains internal
- the corruption looks like a hollow object rather than a broken silhouette

## 5. Carve Strategy

The recommended carve strategy is a `central cavity`:

- compute the object support centroid
- rank interior voxels by distance to that centroid
- remove the most central voxels first

This yields a compact inner cavity and avoids deleting the outer shell.

## 6. Severity

The primary difficulty axis for `hole` is the number of affected objects in one frame.

### 6.1 Easy

- 1 vehicle object
- preserve shell
- remove a moderate fraction of interior voxels
- produce one clear inner cavity

### 6.2 Mid

- 2 vehicle objects
- preserve shell
- remove a larger fraction of interior voxels
- cavity should be more visually obvious than `easy`

### 6.3 Hard

- 3 or 4 vehicle objects
- preserve shell
- remove most or all interior voxels
- object should approach a nearly hollow shell

## 7. Storage

Recommended materialization:

```text
data/OccStress/
  occ/
    hole/
      easy/<scene>/<sample_token>/labels.npz
      mid/<scene>/<sample_token>/labels.npz
      hard/<scene>/<sample_token>/labels.npz

  events/
    hole/
      easy/<scene>/<sample_token>.json
      mid/<scene>/<sample_token>.json
      hard/<scene>/<sample_token>.json

  meta/
    hole_easy_summary.json
    hole_mid_summary.json
    hole_hard_summary.json
```

## 8. Event Payload

Each event should store:

- `severity`
- `sample_token`
- `scene_name`
- `events`
- `changed_voxels_total`
- `occ_path_clean`
- `occ_path_corrupt`

Each object-hole event should store:

- `source_class`
- `instance_token`
- `support_voxels`
- `interior_voxels`
- `removed_voxels`
- `removed_ratio`
- `shell_thickness_voxels`
