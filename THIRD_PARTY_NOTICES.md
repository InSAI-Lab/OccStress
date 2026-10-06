# Third-Party Notices

The root MIT license applies to original OccStress-authored code only. It does
not relicense code in `EXIST/`, copied or adapted third-party operators, model
weights, datasets, or other externally sourced assets. Keep upstream notices
and copyright headers with their components.

- Model code retains the terms in its own directory and nested dependencies.
  Missing notices restored in this preparation are COTR (Apache-2.0) and the
  pinned II-World base (MIT). Their exact sources are recorded in
  [the audit](docs/LICENSES_AND_CITATIONS.md).
- FBBEV/FB-OCC has NVIDIA Source Code License-NC restrictions. PanoOcc retains
  GPL-3.0. Neither component is relicensed under the root MIT license.
- RoboBEV/Robo3D-derived operators are not covered by a blanket MIT grant.
  The upstream READMEs declare CC-BY-NC-SA-4.0 with operator exceptions;
  Robo3D's fog subtree declares CC-BY-NC-4.0. Relevant OccStress adaptation
  entrypoints are `scripts/waymo/waymo_corruptions.py`,
  `scripts/waymo/effocc_waymo_camera_corruptions.py`,
  `scripts/waymo/waymo_sdgocc_corruptions.py`, and
  `scripts/carla/effocc_carla_upstream_adapter.py`. These adapt sensor layouts,
  deterministic seeds and benchmark interfaces; the upstream corruption
  definitions are attributed to their original authors. See the pinned source
  and license links in the audit before redistributing derived portions.
- External imagecorruptions, OpenMMLab, spconv, NATTEN and other runtime
  dependencies retain their own licenses. License availability at a package
  root does not settle every nested component.

Original model attribution and citation keys are listed in
[CITATIONS.md](docs/CITATIONS.md). Dataset and checkpoint publication has
separate resource-specific status; no access credentials or private runtime assets
belong in a public distribution.
