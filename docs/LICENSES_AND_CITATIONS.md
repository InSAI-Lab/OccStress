# Licenses And Citations

This inventory lists source provenance, retained notices and component-specific
terms. Some components have non-commercial or copyleft conditions.

The root [MIT license](../LICENSE) covers original OccStress code authored by
the OccStress authors, not third-party code or derivatives subject to other
terms. Retain upstream notices, including notices inside nested dependencies.
See [third-party notices](../THIRD_PARTY_NOTICES.md) for the scope of exceptions.

## Audit Scope

We checked the complete GitHub source-tree notice inventory and root README
of 19 model repositories plus RoboBEV and Robo3D. We also checked the specific
MIT-licensed II-World base used by the experimental checkout. The pinned
revisions, source links and findings are in
[third-party-audit.json](third-party-audit.json). Current upstream revisions
there identify the audit target, not a claim that every imported file came
from that revision. Imported source provenance remains in
[source-imports.json](source-imports.json).

The inventory describes repository-level sources. More specific notices inside
individual files and nested dependencies remain applicable.

## Restored Notices

| Component | Finding and action |
| --- | --- |
| COTR | The local experimental base and [official commit `4328e0a`](https://github.com/NotACracker/COTR/blob/4328e0abcd1360d950bef4d1c8cbe950a01aaced/LICENSE) contain Apache-2.0. Restored the exact omitted `EXIST/3D/COTR/LICENSE`. |
| II-World | The experimental checkout is based on [MIT-licensed `2b487f8`](https://github.com/lzzzzzm/II-World/blob/2b487f894b46ed82383dd9d5d8f8a85feba30f42/LICENSE). Restored that exact notice. The later upstream commit [`661d830`](https://github.com/lzzzzzm/II-World/commit/661d830f9b34ee03ce368db164a72753ab8764a3) deletes it; we do not infer a grant for later or unrelated code from this restoration. |

Both restored files match the pinned public files and the local experimental
Git objects byte-for-byte. Their origins are recorded in the import inventory.
OccStress adaptation edits remain distinct from the upstream base.

## Additional Model Sources

| Component | Directory | Checked official source |
| --- | --- | --- |
| SparseWorld-TC | `EXIST/4D/SparseWorld` | [SparseWorld `14f5fe3`](https://github.com/MSunDYY/SparseWorld/tree/14f5fe3bdd3428002d913988fbfd9b653a1a207e) |
| CVT-Occ | `EXIST/3D/CVT-Occ` | [CVT-Occ `42c1d4e`](https://github.com/Tsinghua-MARS-Lab/CVT-Occ/tree/42c1d4e49d5f170840e991db1c3d73f56791d53f) |
| FusionOcc | `EXIST/3D/FusionOcc` | [FusionOcc `83ded38`](https://github.com/ShuoZhang-code/FusionOcc/tree/83ded3884b98b299d35d636a91e9aa2a92d89221) |
| SDGOcc | `EXIST/3D/SDGOCC` | [SDGOCC `bd63c00`](https://github.com/DzpLab/SDGOCC/tree/bd63c00b2ba4cbd120077a8a468b8e36a7749154) |
| OccFusion | `EXIST/3D/OccFusion` | [OccFusion `193a3c3`](https://github.com/DanielMing123/OccFusion/tree/193a3c3cb38a8ca8978be54c98e436eb96b5e8c3) |

Package profiles and their source selections are described in
[source packaging](CODE_LAYOUT.md#source-packaging).

## Other Model Sources

The following root notices are present and match the versions inspected in
this audit. This does not supersede more specific notices inside each tree.

| Component | Source | Root notice |
| --- | --- | --- |
| ALOcc | [cdb342/ALOcc](https://github.com/cdb342/ALOcc) | Apache-2.0 |
| BEVFormer | [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer) | Apache-2.0 |
| COTR | [NotACracker/COTR](https://github.com/NotACracker/COTR) | Apache-2.0; restored |
| FBBEV / FB-OCC | [NVlabs/FB-BEV](https://github.com/NVlabs/FB-BEV) | NVIDIA Source Code License-NC; research/evaluation use restriction |
| PanoOcc | [Robertwyq/PanoOcc](https://github.com/Robertwyq/PanoOcc) | GPL-3.0; not MIT |
| SparseOcc | [MCG-NJU/SparseOcc](https://github.com/MCG-NJU/SparseOcc) | Apache-2.0 |
| STCOcc | [lzzzzzm/STCOcc](https://github.com/lzzzzzm/STCOcc) | MIT |
| ViewFormerOcc | [ViewFormerOcc/ViewFormer-Occ](https://github.com/ViewFormerOcc/ViewFormer-Occ) | Apache-2.0 |
| COME | [synsin0/COME](https://github.com/synsin0/COME) | Apache-2.0 |
| II-World | [licensed base](https://github.com/lzzzzzm/II-World/tree/2b487f894b46ed82383dd9d5d8f8a85feba30f42) | MIT at the imported base; restored |
| OccWorld | [wzzheng/OccWorld](https://github.com/wzzheng/OccWorld) | Apache-2.0 |
| GenieDrive | [Huster-YZY/GenieDrive](https://github.com/Huster-YZY/GenieDrive) | MIT; occupancy subset only, not unbundled rasterizer/render licenses |
| DOME | [gusongen/DOME](https://github.com/gusongen/DOME) | Apache-2.0 |
| EFFOcc / CARLA FlashOcc | [synsin0/EFFOcc](https://github.com/synsin0/EFFOcc) | Apache-2.0; distinct model configs/checkpoints |
| CVT runtime MMDetection3D | [open-mmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc4) | Apache-2.0; retained under `CVT-Occ/dependencies/mmdetection3d` |

## Corruption Implementations

[RoboBEV's README](https://github.com/Daniel-xsy/RoboBEV/blob/3a32edaba9434dc27791bd25a1168951d091bd89/README.md#license)
and [Robo3D's README](https://github.com/worldbench/Robo3D/blob/481a3b8634b2d291b84736cab1b546ed266efafa/README.md#license)
declare CC-BY-NC-SA-4.0, with operator-specific exceptions. The absence of a
root `LICENSE` file does not mean the READMEs contain no grant.
Robo3D's [nuScenes fog subtree](https://github.com/worldbench/Robo3D/blob/481a3b8634b2d291b84736cab1b546ed266efafa/create/nuscenes_c/fog/LICENSE)
instead carries CC-BY-NC-4.0. Its lookup tables remain external.

The Waymo/CARLA camera and point-corruption adaptations identify those sources
and revisions. Review copied/adapted portions and operator exceptions before
distribution; do not label them as unrestricted MIT merely because they are
outside `EXIST/`. Preserve source attribution, modification notices and all
applicable non-commercial/share-alike conditions. Consult the linked upstream
READMEs for operator-specific exceptions.

## Checkpoints And Data

The 2026-10-06 [asset distribution review](ASSET_DISTRIBUTION.md) distinguishes
official weight references, project experiment weights and dataset-specific
sharing conditions. It also records the designated Hugging Face dataset repository.

Code permission is separate from weight/data permission. In particular,
[EFFOcc's README](https://github.com/synsin0/EFFOcc/blob/13aeb78c2774f2f3dd447166d1f8f5168fd8059d/README.md)
states that its Occ3D-Waymo checkpoints are not shared due to Waymo rules.
This records the upstream restriction, not an independent interpretation of
every Waymo-trained checkpoint. Weight and dataset terms apply independently
of hosting on Google Drive or Hugging Face.

The current Waymo agreement permits conditional non-commercial model/weight
distribution. The upstream EFFOcc statement above is not a blanket prohibition
on publishing every independently trained Waymo checkpoint; use the separate
asset review before making that decision.

## Citations

Use [CITATION.bib](../CITATION.bib) for OccStress and
[references.bib](references.bib) for the actual methods, datasets and corruption
frameworks used. [Citation guidance](CITATIONS.md) maps components to keys and
records source provenance. Preserved upstream READMEs remain the primary source
for their own recommended citations. Citation is scientific attribution, not a
replacement for a license.
