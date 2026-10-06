# Citation Guide

## OccStress

Use [CITATION.bib](../CITATION.bib), also shown in the root README.
[CITATION.cff](../CITATION.cff) supplies the same preferred paper citation for
GitHub's citation interface. The seven-author order follows the current final
manuscript: Yu Zheng, Jie Hu, Jiaqi Xiong, Ruiping Liu, Junwei Zheng, Kailun Yang,
Jiaming Zhang. The venue is NeurIPS 2026, Evaluations and Datasets Track.

Paper: [OccStress on arXiv](https://arxiv.org/abs/2512.15621).
The README, BibTeX and CFF use this author-designated paper URL consistently.

## Components Used

[references.bib](references.bib) provides 28 entries. Cite the components
actually used in an experiment, not every method in the repository by default.
The recorded primary sources and extraction basis are in
[citation-sources.json](citation-sources.json).

| Component | BibTeX key |
| --- | --- |
| Occ3D annotations | `tian2023occ3d` |
| nuScenes | `caesar2020nuscenes` |
| Waymo Open Dataset | `sun2020scalability` |
| UniOcc benchmark / CARLA subset | `wang2025uniocc` |
| CARLA simulator | `dosovitskiy2017carla` |
| RoboBEV | `xie2025benchmarking` |
| Robo3D | `kong2023robo3d` |
| OccWorld | `zheng2023occworld` |
| II-World | `liao2025i2` |
| COME | `shicome` |
| GenieDrive | `yang2025geniedrive` |
| DOME | `gu2024dome` |
| SparseWorld-TC | `dang2025sparseworld` |
| STCOcc | `liao2025stcocc` |
| SDGOcc | `duan2025sdgocc` |
| ALOcc | `chen2025alocc` |
| FusionOcc | `zhang2024fusionocc` |
| CVT-Occ | `ye2024cvt` |
| EFFOcc | `effocc2025shi` |
| FlashOcc | `yu2023flashocc` |
| BEVFormer | `li2022bevformer` |
| COTR | `ma2023cotr` |
| FBBEV / FB-OCC | `li2023fbbev`, `li2023fbocc` |
| OccFusion | `ming2024occfusion` |
| PanoOcc | `wang2024panoocc` |
| SparseOcc (MCG-NJU) | `liu2023fully` |
| ViewFormerOcc | `li2024viewformer` |

Upstream recommended BibTeX is used where supplied, even when it points to
an earlier preprint rather than later proceedings. Do not equate citation
year with the date a checkpoint was downloaded. Where a README has no BibTeX,
entries come from the final manuscript bibliography with primary-source links.
The missing author-field comma in BEVFormer's upstream BibTeX was repaired;
COME's full author list was checked against its paper metadata.

Similar names can refer to different projects. `EXIST/3D/OccFusion` is
DanielMing123/OccFusion, not FusionOcc or another paper called OccFusion.
`EXIST/3D/SparseOcc` is MCG-NJU/SparseOcc, not a different sparse occupancy
method. UniOcc here is the 2025 unified dataset/forecasting benchmark.

Frameworks and nested dependencies retain their original citation guidance in
their own READMEs. The software's MIT notice and scientific BibTeX do not grant
redistribution rights for other authors' source, data or checkpoints; see
[the license audit](LICENSES_AND_CITATIONS.md).
