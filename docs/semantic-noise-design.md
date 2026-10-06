# Semantic Noise Design

## 1. Goal

`semantic` is intended to simulate semantic recognition errors in occupancy-state rather than geometry corruption.

The preferred benchmark interpretation is:

- keep occupied voxels occupied
- keep the corruption local
- modify semantic identity only
- preserve temporal structure
- support object-aware corruption for thing classes
- support region-aware corruption for static or stuff classes

This document refines the semantic corruption design under the storage architecture defined in:

- `docs/occupancy-state-corruption-design.md`

## 2. Main Design Choice

The default semantic corruption mode should be a **hybrid semantic noise** design:

- `instance-aware` for thing or dynamic classes
- `region-aware` for static or stuff classes

Instead of forcing all classes into one mechanism, the corruption should follow the semantic type.

This means the corruption can simulate:

- `car -> truck`
- `truck -> bus`
- `bicycle -> motorcycle`
- `sidewalk -> other_flat`
- `terrain -> vegetation`
- `manmade -> vegetation`

while still keeping the corruption local.

## 3. Two Semantic Submodes

### 3.1 Instance-Aware Semantic Noise

This is the preferred primary setting.

Applicable classes:

- `car`
- `truck`
- `bus`
- `trailer`
- `construction_vehicle`
- `bicycle`
- `motorcycle`
- `pedestrian`
- optionally `barrier`
- optionally `traffic_cone`

Generation principle:

- choose a nuScenes object instance
- map the instance box into the current occupancy frame
- find occupied voxels whose semantics agree with the source class
- relabel those voxels into another semantic class from the same confusion group

### 3.2 Region-Aware Semantic Noise

This is the preferred mode for static or stuff semantics.

Applicable classes:

- `driveable_surface`
- `other_flat`
- `sidewalk`
- `terrain`
- `manmade`
- `vegetation`

Generation principle:

- choose a local spatial region
- relabel only occupied voxels within that region
- keep label changes constrained by semantic groups
- keep the corruption local rather than scene-wide

Recommended interpretation:

- this simulates local semantic state errors in the static scene map
- it does not rely on object instance annotations
- it affects only a partial region, not the whole class mask

## 4. Temporal Mode

For the benchmark, the recommended default temporal mode is neither fully iid nor long persistent.

The preferred default is:

- `burst`: the same semantic corruption event persists across a short consecutive time span

Definition:

- one event lasts for `L` frames
- `L ~ Uniform{1, 2, 3}`
- the corrupted target is consistent during those frames

Interpretation:

- shorter than a full persistent corruption
- more structured than iid per-frame noise

## 5. Semantic Groups

Semantic target remapping should be constrained by confusion groups rather than allowing arbitrary label flips.

Recommended groups:

### 5.1 Vehicle Group

- `car`
- `truck`
- `bus`
- `trailer`
- `construction_vehicle`

### 5.2 Vulnerable Road User Group

- `bicycle`
- `motorcycle`
- `pedestrian`

### 5.3 Small Roadside Object Group

- `barrier`
- `traffic_cone`

### 5.4 Ground Group

- `driveable_surface`
- `other_flat`
- `sidewalk`
- `terrain`

### 5.5 Structure Group

- `manmade`
- `vegetation`
- `others`

Default rule:

- remap only within the same group
- do not remap to `free`
- do not perform arbitrary cross-group flips by default

This keeps the corruption plausible while still challenging.

## 6. Hybrid Generator Rule

The benchmark should treat semantic corruption as the union of two generators.

### 6.1 Thing Branch

Use `instance-aware semantic noise` for:

- `vehicle`
- `vru`
- `roadside` when instance support is reliable enough

This branch simulates object recognition mistakes.

### 6.2 Static Branch

Use `region-aware semantic noise` for:

- `ground`
- `structure`

This branch simulates local semantic map mistakes in the static environment.

### 6.3 Mixing Rule

At frame level, one semantic event may contain:

- only thing-branch edits
- only static-branch edits
- both thing and static edits

Recommended default:

- `easy`: include both one thing event and one static region
- `mid`: include both thing events and at least one static region
- `hard`: include both thing events and one or more static regions in the same frame or burst

## 7. Object Alignment Rule

For instance-aware semantic noise, alignment with nuScenes objects should follow:

1. choose a sample token
2. gather all annotations with that sample token
3. map nuScenes category names into benchmark occupancy class names
4. filter objects whose mapped class belongs to an allowed semantic group
5. transform the object box into the occupancy frame coordinates
6. modify only:
   - voxels inside the box
   - voxels currently labeled as the source class

This prevents semantic noise from leaking into unrelated occupied regions.

## 8. Why Box-Filtered Voxels Are Preferred

Occupancy labels are semantic, not instance-level.

Therefore, the safest approximation is:

- use the object box as a spatial support
- use current semantic class as a second filter

In other words, a voxel is eligible for corruption only if:

- it lies inside the instance box
- and its current occupancy label matches the source semantic class

This substantially reduces incorrect edits to background or nearby objects.

## 9. Final Instance Support Rule

After prototype validation, the recommended instance support is no longer just:

- `bbox ∩ source-class voxels`

The preferred support is:

- `completed_instance + bbox_expand_voxels = 1`

This means:

1. start from voxels inside the annotation box whose semantic label matches the source class
2. collect box-contained non-free voxels connected to those source voxels while excluding obvious static background classes
3. expand the annotation box by one voxel along each axis
4. add newly covered voxels only if:
   - they are non-free
   - and their semantic label is still the source class

This rule is preferred because it captures:

- the core instance body
- slightly incomplete object boundaries
- occupancy voxels just outside the nominal box boundary

while still avoiding aggressive leakage into unrelated nearby objects.

## 10. Static Region Support Rule

For static semantic noise, support is defined by a local region rather than an instance box.

Recommended support definition:

1. choose a seed voxel from one allowed static source class
2. sample a local 3D cuboid around the seed in the current occupancy frame
3. keep only voxels inside the cuboid whose current semantic label matches the source class
4. optionally keep only the connected component attached to the seed voxel
5. relabel the resulting local support to a target class from the same static confusion group

Recommended defaults:

- coordinate frame: current occupancy frame
- region shape: axis-aligned cuboid
- support type: local same-class support
- keep free voxels unchanged

## 11. Static Region Parameterization

Static semantic noise should be controlled by:

- region size
- number of static regions in a frame
- source class group
- target remap class
- connectivity rule

Recommended initial ranges:

- `easy`
  - one region
  - approximately `2m-4m x 2m-4m x 1m-2m`
  - one static group only
- `mid`
  - one or two regions
  - approximately `4m-8m x 4m-8m x 1m-3m`
  - optionally mix `ground` and `structure`
- `hard`
  - two or three regions
  - approximately `6m-12m x 6m-12m x 1m-4m`
  - may coexist with object-aware semantic corruption

## 12. Severity Definition

Severity should not be controlled by a single parameter only.

Recommended severity factors:

- number of events per clip
- voxel relabel ratio inside the object support
- number of affected objects
- temporal duration

Recommended initial setting:

### 12.1 Easy

- 1 affected object per temporal episode
- use `completed_instance + bbox_expand_voxels=1`
- relabel `100%` of the final support
- duration fixed to `1` frame
- restrict to large, easy-to-inspect `vehicle` instances
- recommend a minimum object support threshold such as `> 500` voxels
- remap only within close semantic neighbors:
  - `car <-> truck`
  - `truck <-> bus`
  - avoid long-distance group jumps

Interpretation:

- one clearly visible vehicle
- one frame only
- one simple object identity mistake
- group sampling probability:
  - `vehicle = 1.0`
  - `vru = 0.0`
  - `roadside = 0.0`
- optional static-only variant:
  - 1 local static region
  - remap inside `ground` or inside `structure`
  - no thing corruption in the same frame

### 12.2 Mid

- 1 or 2 affected objects
- use `completed_instance + bbox_expand_voxels=1`
- relabel `100%` of the final support
- duration `1 - 3` frames using the `burst` mode
- allow all three instance-aware groups:
  - `vehicle`
  - `vru`
  - `roadside`
- use group-level sampling probability:
  - `vehicle = 0.75`
  - `vru = 0.15`
  - `roadside = 0.10`
- keep vehicle as the dominant group, but do not force the other groups to zero
- allow full within-group remapping:
  - `car`
  - `truck`
  - `bus`
  - `trailer`
  - `construction_vehicle`
- `bicycle`
- `motorcycle`
- `pedestrian`
- `barrier`
- `traffic_cone`
- allow medium and moderately small objects
- recommend a minimum object support threshold such as `> 100` voxels

Interpretation:

- one short burst
- one or two objects
- still vehicle-dominant, but no longer vehicle-exclusive
- non-vehicle groups are allowed and should have non-zero sampling probability
- mid should not collapse into an all-vehicle-only regime
- optionally add 1 static region in the same frame or burst
- recommended static region count:
  - `0 or 1`

### 12.3 Hard

- 2 to 4 affected objects
- use `completed_instance + bbox_expand_voxels=1`
- relabel `100%` of the final support
- duration `2 - 3` consecutive frames
- allow multiple semantic groups:
  - `vehicle`
  - `vru`
  - `roadside`
- allow small, medium, and large instances
- recommend a lower minimum object support threshold such as `> 50` voxels
- allow multiple simultaneous source-target mappings inside the same temporal episode
- group sampling probability:
  - `vehicle = 0.55`
  - `vru = 0.30`
  - `roadside = 0.15`
- recommend a soft cap such as `vehicle <= 2` objects per episode when enough non-vehicle candidates exist
- recommend forcing at least one non-vehicle object when the candidate pool permits
- allow `1 to 3` static semantic regions in the same frame or burst
- allow thing and static semantic edits to coexist

Interpretation:

- several objects are corrupted together
- the corruption persists across multiple frames
- difficulty comes from instance count, static region count, size diversity, and temporal continuity
- hard should visibly include non-vehicle corruption when such candidates are available
- hard should also visibly include static-scene corruption

The benchmark should treat semantic corruption as an **object identity error**, not as a partial masking effect.
Therefore, the default official setting should use full-object relabeling rather than random partial relabeling.

Partial relabeling can still be retained as an auxiliary ablation, but it should not be the main benchmark setting.

## 13. Why the Three Levels Are Now Clearly Separated

The severity gap should not be created by random voxel ratio differences alone.

The main axes of separation are:

- number of corrupted instances
- duration in frames
- semantic group scope
- object size difficulty
- complexity of simultaneous confusion events

In short:

- `easy`: single large vehicle or one local static region
- `mid`: one or two objects, short burst, optionally plus one static region
- `hard`: multiple objects, multiple frames, multiple semantic groups, plus static regions

## 14. Temporal Episode Construction

Although semantic corruption is materialized per frame, its temporal meaning should be defined at the episode level.

The recommended semantic episode definition is:

```python
{
    "anchor_token": "...",
    "history_tokens": ["...", "...", "..."],
    "severity": "mid",
    "temporal_mode": "burst",
    "instances": [
        {
            "instance_token": "...",
            "source_class": "truck",
            "target_class": "car",
            "burst_tokens": ["...", "..."]
        }
    ]
}
```

Recommended rules:

1. choose the anchor frame first
2. define the history window around that anchor
3. sample one or more target instances inside the history window
4. for each target instance, sample a burst length:
   - `easy`: fixed to `1`
   - `mid`: `1 - 3`
   - `hard`: `2 - 3`
5. keep the instance fixed during the burst
6. keep the source-target semantic mapping fixed during the burst
7. materialize each affected history frame independently
8. let the temporal protocol reference those materialized corrupted frames later

This gives the benchmark both:

- frame-level storage simplicity
- temporally coherent semantic corruption

## 15. Anchor Rule

Unless explicitly needed otherwise, the anchor target should remain clean.

The corruption is primarily applied to history state.

That means:

- protocol chooses whether a history frame comes from clean or semantic-noise subset
- benchmark supervision remains the clean anchor occupancy

## 16. Frame-Level vs Episode-Level Responsibilities

The semantic pipeline is split into two responsibilities.

### 13.1 Frame-Level Generator

The frame-level generator is responsible for:

- selecting eligible object instances
- selecting eligible static seed regions
- constructing the support voxels
- applying semantic relabeling
- saving corrupted occupancy files
- saving per-frame event metadata

### 13.2 Episode-Level Protocol Builder

The episode-level protocol builder is responsible for:

- choosing the anchor frame
- choosing history tokens
- selecting which frames use clean vs semantic-noise occupancy
- enforcing burst duration and temporal consistency
- grouping one or more object-level semantic events into one benchmark episode

Therefore, the subset generator should produce deterministic per-frame semantic corruption artifacts, while the protocol layer should impose the final temporal benchmark semantics.

## 17. Generation Workflow

The recommended semantic-noise generation workflow is:

1. choose a sample token
2. load occupancy `labels.npz`
3. load the corresponding nuScenes sample annotations
4. map each annotation category into the occupancy semantic label space
5. choose one or more eligible semantic events:
   - object instances for thing classes
   - local regions for static classes
6. if the event is object-aware:
   - transform the annotation box from global coordinates into the occupancy frame
   - build the instance support using `completed_instance + bbox_expand_voxels=1`
7. if the event is static-region-aware:
   - choose a static seed voxel
   - sample a local cuboid
   - keep same-class occupied voxels within that region
8. choose a target semantic label from the same confusion group
9. relabel the final support from source class to target class
10. save:
    - corrupted occupancy
    - event metadata
    - inspection visualizations

This gives a deterministic semantic corruption pipeline that is compatible with the later temporal protocol layer.

## 18. Storage Under Current Architecture

The semantic corruption design should fit the per-frame storage architecture.

Recommended materialization:

```text
data/OccStress/
  occ/
    semantic/
      easy/<scene>/<sample_token>/labels.npz
      mid/<scene>/<sample_token>/labels.npz
      hard/<scene>/<sample_token>/labels.npz

  events/
    semantic/
      easy/<scene>/<sample_token>.json
      mid/<scene>/<sample_token>.json
      hard/<scene>/<sample_token>.json
```

Each event file should store:

- source sample token
- source annotation token or instance token
- source semantic class
- target semantic class
- temporal duration
- affected voxel count
- random seed

For semantic corruption specifically, the event payload should also store:

- `severity`
- `support_mode`
- `bbox_expand_voxels`
- `group`
- `branch`
- `eligible_voxels`
- `changed_voxels`
- `regions` when static-region corruption is present

## 19. Recommended Initial Implementation

The initial implementation should prioritize:

- object-aware corruption
- one sample at a time
- one affected instance at a time
- deterministic output

This is sufficient for:

- verifying voxel-object alignment quality
- checking visual plausibility
- validating semantic group mapping

Only after that should batch generation be expanded to the full dataset.

Current implementation note:

- the existing prototype script only covers the object-aware branch
- static-region semantic corruption still needs to be added
- a dedicated review-sample prototype can combine one object-aware event with one static region for qualitative inspection

## 20. Immediate Implementation Plan

The first executable prototype should:

1. load one occupancy sample
2. load raw nuScenes JSON metadata
3. select one eligible object instance
4. map the object box into occupancy voxel space
5. relabel matched voxels into a target class from the same semantic group
6. save:
   - corrupted `labels.npz`
   - event metadata
   - a simple BEV visualization for inspection

The next prototype should extend this with static semantic regions:

1. choose one static source class from `ground` or `structure`
2. sample one local cuboid around a same-class seed voxel
3. relabel only the local same-class occupied support
4. record the region metadata in the event file
