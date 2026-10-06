# Dropout Design

## 1. Goal

`dropout` is intended to simulate a contiguous chunk of occupancy being entirely missing.

The target effect is:

- choose one or more local 3D regions
- remove all occupied voxels inside those regions
- create a clear block-level missing-state artifact

This corruption is different from `hole`:

- `hole` preserves the outer shell of an object and hollows its interior
- `dropout` removes an entire local occupancy block, regardless of whether it belongs to dynamic objects, static structures, or mixed content

## 2. Scope

`dropout` may affect:

- dynamic object voxels
- static scene voxels
- mixed regions that contain both

It should not be restricted to a single semantic family.

## 3. Region Rule

The corruption unit is a local 3D region anchored by an occupied seed.

For each selected region:

1. choose a seed from occupied voxels
2. sample a local envelope size based on severity
3. build a removal mask inside that envelope
4. collect occupied voxels covered by the removal mask
5. set those voxels to `free`

For `easy` and `mid`, the removal mask can remain cuboid-like.

For `hard`, the removal mask should be allowed to be irregular rather than strictly box-shaped. A practical implementation is:

- sample a large envelope
- place multiple local blob centers inside that envelope
- keep a high-score irregular union of those blobs
- remove occupied voxels covered by that irregular mask

To avoid degenerate empty regions, each cuboid should satisfy:

- minimum removed occupied voxel count
- minimum occupied density inside the cuboid

## 4. Severity

The main difficulty axes are:

- number of dropped regions in one frame
- region size
- amount of removed occupied voxels
- regular vs irregular region shape

### 4.1 Easy

- 1 dropped region
- moderate cuboid size
- clear local missing patch

### 4.2 Mid

- 2 dropped regions
- larger cuboids than `easy`
- noticeably more removed voxels than `easy`
- missing regions should be visibly separated when possible

### 4.3 Hard

- 3 or 4 dropped regions
- larger regions than `mid`
- more removed voxels than `mid`
- irregular regions are allowed and preferred
- substantial occupancy removal across multiple parts of the frame

## 5. Storage

```text
data/OccStress/
  occ/
    dropout/
      easy/<scene>/<sample_token>/labels.npz
      mid/<scene>/<sample_token>/labels.npz
      hard/<scene>/<sample_token>/labels.npz

  events/
    dropout/
      easy/<scene>/<sample_token>.json
      mid/<scene>/<sample_token>.json
      hard/<scene>/<sample_token>.json

  meta/
    dropout_easy_summary.json
    dropout_mid_summary.json
    dropout_hard_summary.json
```

## 6. Event Payload

Each event file should store:

- `severity`
- `sample_token`
- `scene_name`
- `num_regions_requested`
- `num_regions_applied`
- `changed_voxels_total`
- `events`
- `occ_path_clean`
- `occ_path_corrupt`

Each dropped-region event should store:

- `center_voxel`
- `half_size_voxels`
- `bounds`
- `removed_voxels`
- `occupied_density`
- `class_histogram`
- `shape_mode`
- `removed_fraction`
