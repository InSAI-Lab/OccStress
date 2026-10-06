# Misalignment Design

## 1. Goal

`misalignment` is intended to simulate temporal alignment errors between history frames and the anchor frame.

The target effect is:

- history occupancy content itself remains clean
- the relative transform used to align history into the anchor frame is wrong
- temporal fusion therefore sees shifted or rotated historical occupancy

This corruption is different from `semantic`, `hole`, and `dropout`:

- those modify voxel content
- `misalignment` modifies only temporal alignment metadata

## 2. Core Storage Rule

`misalignment` should **not** be materialized as a new occupancy subset under `occ/`.

The preferred storage is:

- clip-scoped event metadata in `events/misalignment/`
- clip-scoped matrix cache in `cache/misalignment/`

This is necessary because the corruption is anchor-relative:

- the same history frame can have different relative transforms under different anchors

Therefore, `misalignment` should be stored per temporal sample rather than per frame.

## 3. Transform Definition

For an anchor frame `t` and a history frame `h`, the clean relative transform is:

```text
T_clean(t <- h)
```

The corrupted transform is:

```text
T_misaligned(t <- h) = Δ(h; t) · T_clean(t <- h)
```

where `Δ(h; t)` is a sampled perturbation in the anchor frame.

Recommended perturbation dimensions:

- `dx`
- `dy`
- `yaw`

Recommended to keep:

- `dz = 0`
- `roll = 0`
- `pitch = 0`

for the first benchmark version.

## 4. Temporal Modes

Recommended supported modes:

### 4.1 Constant Bias

All affected history frames share the same perturbation:

- same `dx`
- same `dy`
- same `yaw`

This simulates:

- fixed synchronization bias
- systematic localization offset

### 4.2 Drift

Older history frames have larger perturbations than recent history frames.

This simulates:

- accumulated odometry drift
- progressively worsening temporal alignment

Recommended form:

- sample one maximum perturbation
- scale it by normalized history distance to the anchor

## 5. Severity

Severity should control both:

- perturbation magnitude
- number of affected history frames

### 5.1 Easy

- affect a small subset of history frames
- prefer `constant_bias`
- small translation and yaw

Suggested range:

- `|dx|, |dy| <= 0.4 m`
- `|yaw| <= 2 deg`

### 5.2 Mid

- affect more history frames
- allow `constant_bias` or mild `drift`
- medium translation and yaw

Suggested range:

- `|dx|, |dy| <= 0.8 m`
- `|yaw| <= 4 deg`

### 5.3 Hard

- affect most or all history frames
- prefer `drift`
- larger translation and yaw

Suggested range:

- `|dx|, |dy| <= 1.5 m`
- `|yaw| <= 8 deg`

## 6. Storage

```text
data/OccStress/
  events/
    misalignment/
      easy/<scene>/<sample_id>.json
      mid/<scene>/<sample_id>.json
      hard/<scene>/<sample_id>.json

  cache/
    misalignment/
      easy/<scene>/<sample_id>.npz
      mid/<scene>/<sample_id>.npz
      hard/<scene>/<sample_id>.npz

  meta/
    misalignment_easy_summary.json
    misalignment_mid_summary.json
    misalignment_hard_summary.json
```

Recommended `sample_id`:

```text
<anchor_token>__H<history_length>__F<future_length>__misalignment_<severity>__<mode>
```

## 7. Event Payload

Each event file should store:

- `type`
- `severity`
- `mode`
- `sample_id`
- `scene_name`
- `anchor_token`
- `history_length`
- `history_tokens`
- `affected_mask`
- `dx_m`
- `dy_m`
- `yaw_deg`
- `cache_path`

## 8. Matrix Cache Payload

Each `.npz` cache file should store:

- `rt_clean`: `[H, 4, 4]`
- `delta_rt`: `[H, 4, 4]`
- `rt_misaligned`: `[H, 4, 4]`
- `affected_mask`: `[H]`
- `dx_m`: `[H]`
- `dy_m`: `[H]`
- `yaw_deg`: `[H]`

This gives a direct interface for later protocol integration or loader-side consumption.
