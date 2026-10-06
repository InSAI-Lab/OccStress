# Occupancy-State Corruption Subset Design

## 1. Scope

This document specifies how to construct and store the `occupancy-state corruption` subset for the OccStress benchmark based on nuScenes occupancy data.

The current target corruption families are:

- `semantic`: local semantic mislabeling
- `hole`: internal holes or fractures inside occupied voxels
- `dropout`: large missing occupancy regions
- `misalignment`: temporal alignment errors between history frames
- `traffic`: left-right mirroring to simulate opposite-side driving

This document focuses on:

- data organization
- storage layout
- temporal protocol definition
- compatibility with the current codebase
- reproducibility and future extensibility

For `semantic`, the current direction is a hybrid design:

- `instance-aware` semantic corruption for thing classes
- `region-aware` semantic corruption for static or stuff classes

## 2. Design Goal

The benchmark should evaluate robustness to corrupted occupancy-state inputs under a temporal setting, while keeping the supervision target clean.

The design must satisfy the following:

- avoid storing duplicated temporal clips
- separate clean ground truth from corrupted input state
- support variable temporal history lengths
- support frame-wise and sequence-wise corruption protocols
- support deterministic reconstruction from metadata
- remain easy to load using the existing per-frame occupancy folder interface

## 3. Core Decision

The subset will **not** use a materialized `anchor-clip` file as the primary storage unit.

Instead, it will use:

- one `clean occupancy subset`
- multiple `corrupted occupancy subsets`
- one set of `temporal protocol files`

In other words:

- occupancy tensors are stored **per frame**
- temporal samples are defined **by protocol metadata**

This avoids repeated storage of highly overlapping temporal windows such as:

- `[t-7, ..., t]`
- `[t-6, ..., t+1]`
- `[t-5, ..., t+2]`

which would otherwise duplicate most history frames many times.

## 4. Storage Principle

The benchmark is split into two logical layers.

### 4.1 Data Layer

This layer stores actual occupancy tensors.

- `clean_occ`: clean occupancy states
- `occ/semantic`
- `occ/hole`
- `occ/dropout`
- `occ/traffic`

For `misalignment`, the preferred representation is **not** a separate full occupancy tensor subset. It should primarily be represented by corrupted temporal transforms in protocol metadata, because the corruption lies in alignment rather than in voxel semantics themselves.

### 4.2 Protocol Layer

This layer stores temporal benchmark samples.

Each protocol record defines:

- which frame is the anchor frame
- which history frames are used
- for each history frame, whether the source is clean or corrupted
- which corruption family and severity are active
- clean temporal transforms
- corrupted temporal transforms when needed

Therefore, the actual benchmark sample is assembled at read time from:

- frame-level occupancy files
- protocol-level indexing metadata

## 5. Recommended Directory Layout

```text
data/OccStress/
  clean_occ/
    <scene_name>/<sample_token>/labels.npz
    <scene_name>/<sample_token>/labels_1_2.npz
    <scene_name>/<sample_token>/labels_1_4.npz
    <scene_name>/<sample_token>/labels_1_8.npz

  occ/
    semantic/
      <severity>/<scene_name>/<sample_token>/labels.npz
    hole/
      <severity>/<scene_name>/<sample_token>/labels.npz
    dropout/
      <severity>/<scene_name>/<sample_token>/labels.npz
    traffic/
      <scene_name>/<sample_token>/labels.npz

  protocols/
    semantic_easy.pkl
    semantic_mid.pkl
    semantic_hard.pkl
    hole.pkl
    dropout.pkl
    misalignment.pkl
    traffic.pkl

  events/
    semantic/
      <severity>/<scene_name>/<sample_token>.json
    hole/
      <severity>/<scene_name>/<sample_token>.json
    dropout/
      <severity>/<scene_name>/<sample_token>.json
    misalignment/
      <severity>/<scene_name>/<anchor_token>.json
    traffic/
      <scene_name>/<sample_token>.json

  meta/
    corruption_schema.yaml
    subset_info.yaml
```

## 6. Naming and Roles

### 6.1 Clean Occupancy

`clean_occ` stores the clean occupancy tensors that act as:

- clean input source when the protocol requests clean history
- clean supervision target for anchor frames

### 6.2 Corrupted Occupancy

`occ` stores materialized corrupted occupancy tensors for corruption families that directly alter occupancy content.

This currently includes:

- `semantic`
- `hole`
- `dropout`
- `traffic`

### 6.3 Event Files

`events` stores compact metadata for deterministic reconstruction or inspection.

Examples:

- local semantic remap region
- static semantic cuboid region and source-target remap
- carved hole mask parameters
- dropout cuboid definition
- mirrored axis and scene-level remap info
- corrupted temporal pose offsets

Event files are useful for:

- reproducing corruption without storing full tensors
- debugging corruption placement
- auditing severity settings

### 6.4 Protocol Files

`protocols/*.pkl` defines temporal sampling rules.

The protocol is the canonical definition of benchmark episodes. It does not store dense voxel arrays.

## 7. Protocol Record Definition

Each record in a protocol file should define one temporal sample.

Recommended structure:

```python
{
    "sample_id": "scene-0001__anchor_xxx__H4__F6__semantic_easy__history_k1",
    "scene_name": "scene-0001",
    "scene_token": "...",
    "anchor_token": "...",
    "anchor_timestamp": 1234567890,
    "history_length": 4,
    "history_tokens": ["...", "...", "..."],
    "future_length": 6,
    "future_tokens": ["...", "...", "..."],
    "current_input": {
        "token": "...",
        "source": "clean",
        "occ_path": "data/nuscenes/gts/scene-0001/<anchor_token>/labels.npz"
    },
    "target": {
        "token": "...",
        "source": "clean",
        "occ_path": "data/nuscenes/gts/scene-0001/<anchor_token>/labels.npz"
    },
    "future_targets": [
        {
            "index": 0,
            "token": "...",
            "source": "clean",
            "occ_path": "data/nuscenes/gts/scene-0001/<future_token>/labels.npz"
        }
    ],
    "history": [
        {
            "token": "...",
            "occ_source": "clean",
            "occ_path": "data/nuscenes/gts/scene-0001/<token>/labels.npz",
            "rt_source": "clean"
        },
        {
            "token": "...",
            "occ_source": "semantic",
            "occ_path": "data/OccStress/occ/semantic/easy/scene-0001/<token>/labels.npz",
            "event_path": "data/OccStress/events/semantic/easy/scene-0001/<token>.json",
            "rt_source": "clean"
        }
    ],
    "corruption": {
        "type": "semantic",
        "severity": "easy",
        "frame_protocol": "history_k1",
        "k": 1
    }
}
```

## 8. Temporal Semantics

The benchmark sample is centered around an **anchor frame**:

- the anchor frame is the frame evaluated or predicted
- the history frames provide temporal context
- the future frames provide clean supervision targets for forecasting tasks
- the anchor target supervision remains clean
- the current input can be either clean or corrupted, depending on the frame protocol

This implies the benchmark protocol always distinguishes:

- `target supervision`
- `history state source`

The benchmark should avoid mixing those two concepts into a single `occ_path`.

## 9. Clean vs Corrupted Separation

The following rule should be treated as mandatory:

- the anchor target occupancy should remain clean unless a future task explicitly defines otherwise

This means:

- benchmark input may be corrupted
- benchmark target remains clean

Practical consequence:

- do not overwrite the clean `occ_path` semantics used for evaluation
- store corrupted state using a separate source namespace

## 10. Recommended Protocol Granularity

One protocol file per corruption family is recommended at first:

- `semantic_easy.pkl`
- `semantic_mid.pkl`
- `semantic_hard.pkl`
- `hole_noise.pkl`
- `region_dropout.pkl`
- `misalignment.pkl`
- `traffic.pkl`

If needed later, the protocol can be split further by:

- severity
- history length
- persistence mode
- train/val/test

For example:

- `semantic_s1_H4.pkl`
- `semantic_s2_H8.pkl`
- `misalignment_drift_s3.pkl`

## 11. Severity and Reproducibility

Every protocol record should store:

- corruption type
- severity
- seed
- temporal mode

Recommended deterministic seed policy:

- derive seed from `(scene_token, anchor_token, corruption_type, severity)`

This guarantees:

- reproducibility across runs
- independence from data loading order
- easier regeneration if corrupted subsets need to be rebuilt

## 12. Corruption-Specific Storage Strategy

### 12.1 Semantic

Preferred storage:

- materialized corrupted occupancy in `occ/semantic`
- optional local remap metadata in `events/semantic`

### 12.2 Hole Noise

Preferred storage:

- materialized corrupted occupancy in `occ/hole_noise`
- optional hole geometry metadata in `events/hole_noise`

### 12.3 Region Dropout

Preferred storage:

- materialized corrupted occupancy in `occ/region_dropout`
- optional dropout region metadata in `events/region_dropout`

### 12.4 Misalignment

Preferred storage:

- protocol-level corrupted transforms
- event files storing pose perturbation parameters

This corruption should be treated primarily as a **temporal pose corruption**, not as a voxel-content corruption.

Therefore:

- clean occupancy files remain reusable
- only temporal alignment metadata changes

### 12.5 Traffic

Preferred storage:

- materialized mirrored occupancy in `occ/traffic`
- per-frame metadata in `events/traffic`

This corruption is closer to a **scene-level geometric remapping** than a local random noise process.

## 13. Temporal Modes

The protocol should support at least two temporal modes for content corruption:

- `persistent`: corruption persists across consecutive history frames
- `flicker`: corruption appears only on some history frames

This is important because a temporal benchmark should distinguish:

- robustness to short transient noise
- robustness to persistent corrupted state

For `misalignment`, the protocol should support:

- `constant_bias`
- `drift`

For the current benchmark release, the concrete frame protocols are fixed as:

- `all_frame`: all history frames are corrupted and current remains clean
- `history_k1`: the history frame closest to current is corrupted and current input is also corrupted
- `current`: only the current input frame is corrupted and target supervision remains clean

Current plan:

- `semantic`, `hole`, `dropout`, `misalignment`: generate all three frame protocols
- `traffic`: generate one compatibility alias `all_frame`, but interpret it as a full mirrored sequence
  - history mirrored
  - current mirrored
  - target mirrored
  - future targets mirrored

## 14. Compatibility with Existing Codebase

The current codebase expects occupancy data as a per-frame folder containing:

- `labels.npz`
- optional multi-scale occupancy files

Therefore, this design intentionally keeps corrupted occupancy materialized at the same per-frame folder granularity.

This makes it straightforward later to:

- swap `occ_path` with a protocol-selected source path
- preserve current loading behavior
- inject temporal state selection through additional protocol-aware fields

## 15. What Should Not Be Done

The following design is not recommended as the primary storage format:

- storing each temporal sample as a fully materialized `anchor-clip` tensor package

Reason:

- severe frame duplication
- large storage overhead
- poor flexibility when changing history length
- poor reuse across protocols

`Anchor-clip` remains a useful **conceptual unit for protocol definition**, but not the primary storage unit.

## 16. Final Recommended Version

The current recommended benchmark design is:

- one clean occupancy subset
- four materialized occupancy corruption subsets:
  - `semantic`
  - `hole_noise`
  - `region_dropout`
  - `traffic`
- one pose-based corruption protocol:
  - `misalignment`
- five protocol files describing temporal sampling and source selection

In short:

- store occupancy by frame
- define temporal episodes by protocol
- keep supervision clean
- use protocol metadata to choose clean vs corrupted history
- treat `misalignment` as transform corruption rather than dense occupancy duplication

## 17. Open Items for Later Discussion

The following will be decided corruption-by-corruption later:

- exact severity levels
- exact local region sampling rules
- class remapping policy for semantic noise
- whether corruption is defined in ego coordinates or world coordinates
- persistence window length
- whether some corruption families also affect the anchor frame
- whether traffic should remain occupancy-only or become a full scene remap

## 18. Immediate Next Step

The next step is to refine each corruption family separately and freeze:

- generation rule
- severity definition
- temporal consistency rule
- whether it is materialized as corrupted occupancy or stored as protocol-only metadata
