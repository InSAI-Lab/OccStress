# Data And Weight Distribution

Evidence reviewed on 2026-10-06. This is a release-scope decision record, not
legal certification. The machine-readable plan is
[asset_distribution.json](../configs/asset_distribution.json).
No upload or third-party mirror is authorized merely by this document.

## Dataset Repository

The designated repository is
[insailab/OccStress](https://huggingface.co/datasets/insailab/OccStress).
Its unauthenticated package index and sampled archive links returned HTTP 200
on 2026-10-06. Read the [live package index](https://huggingface.co/datasets/insailab/OccStress/resolve/main/packages.json)
for per-shard availability. The release owner authorized the planned derived
packages through public links with source-term notices; this is not a claim of
new permission from the source providers.

| Component | Planned delivery | Remaining condition |
| --- | --- | --- |
| Original sensors and official clean GT | User obtains from original providers | Source access terms; no copies in OccStress packages |
| nuScenes-derived manual/upstream assets and protocols | Conditional non-commercial release | Current dataset terms, attribution, inherited notices and prediction-output terms |
| Waymo dataset modifications, including manual occupancy | Registered-recipient distribution | Verify Waymo registration and acceptance; include applicable notices |
| Waymo model-generated assets | Review separately from dataset modifications | Applicable output/model terms; not blanket MIT |
| CARLA-derived manual/upstream assets | Conditional source-license release | Retain UniOcc notices; check additional component terms |
| Private logs, original paths, credentials and caches | Excluded | Never copy the staging parent or private provenance directory |

[Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D#license) declares its code and
generated data MIT-licensed, but also requires acceptance of nuScenes/Waymo
source terms. The original [nuScenes paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf)
identifies CC BY-NC-SA 4.0; its current
[dataset terms](https://www.nuscenes.org/terms-of-use) must also be checked before
publication. Consult the provider's current terms for additional conditions.

The [March 2025 Waymo agreement](https://waymo.com/open/terms/) distinguishes
dataset modifications, shared with registered users who accepted the terms,
from models and related outputs. It permits conditional non-commercial model
distribution, including weights; it is not a blanket weight-sharing ban.
Review the actual artifact and include the required agreement and attribution.
An upstream repository's decision not to share its weights does not grant us
permission to mirror them or establish that our own weights can never be shared.

This is an artifact-level review, not a rule that every Waymo-named file needs
manual approval. GT-derived manual labels are conservatively treated as dataset
modifications; model predictions need a separate classification. Protocol code
is distinct from protocol files containing copied poses or control metadata.
Being a label rather than an image does not by itself remove source conditions.
The terms specify eligible recipients, not a mandatory manual-approval workflow.

The selected publication approach follows Occ3D's public-link and source-terms
notice model. Manual approval is not an additional OccStress requirement.
Users must still obtain the original access and accept applicable terms; a
public URL does not waive recipient eligibility or redistribution conditions.
No repository visibility or gating settings are changed by the upload tool.

The pinned [UniOcc dataset revision](https://huggingface.co/datasets/tasl-lab/uniocc/tree/e81775b36e376f591a1145ca054b72fd541a67eb)
has an MIT dataset-card declaration. Its actual CARLA validation file inventory
contains the three 120-frame scenes used here, even though the README summary
table lists different counts. Retain the source notice and record our class and
coordinate transformations. This finding does not relicense other UniOcc
subsets or automatically clear every simulator asset.

Corruption implementation licenses and exceptions are recorded in
[the code license audit](LICENSES_AND_CITATIONS.md#corruption-implementations).
A source-code license is not, by itself, a determination of every generated
output's license. Keep data, code, and checkpoint reviews separate.

## Are Checkpoints Required?

Running a forecaster requires its selected checkpoint and associated tokenizer
or VAE/ControlNet components. They need not be hosted by OccStress: the preferred
route is an author-maintained link with the exact variant and file identity.
To evaluate already exported occupancy, upstream 3D model weights are **not**
required. They are needed only to regenerate those predictions.

| Group | Delivery decision |
| --- | --- |
| OccWorld/tokenizer, II-World/tokenizer, GenieDrive, DOME/VAE, COME components | Official links recorded; no mirrors. File-level benchmark identity still needs matching. |
| ALOcc | Official model hub; select the experiment's exact config/variant. |
| STCOcc | Keep experiment `iter_63288.pth` identity distinct from the official model-zoo alternative. Do not silently substitute. |
| EFFOcc-Waymo, EFFOcc-CARLA, FlashOcc-CARLA | Experiment checkpoints. Confirm training provenance and terms before publishing a project-owned mirror. |
| SparseWorld-TC, CVT-Occ, FusionOcc, SDGOcc | Source is included; obtain the exact weights separately. Source inclusion does not grant checkpoint redistribution rights. |

Official source pages: [OccWorld](https://github.com/wzzheng/OccWorld),
[II-World](https://github.com/lzzzzzm/II-World),
[COME](https://github.com/synsin0/COME), [DOME](https://github.com/gusongen/DOME),
[GenieDrive](https://github.com/Huster-YZY/GenieDrive),
[ALOcc](https://github.com/cdb342/ALOcc), [STCOcc](https://github.com/lzzzzzm/STCOcc).
COME now directs users to its model hub instead of older cloud-disk links.

Google Drive is reserved for any separately approved project-owned mirrors.
Do not publish third-party weights simply because their source code is licensed.
See [resources](RESOURCES.md) for the distinction between a reference link and
a verified downloadable benchmark artifact.
