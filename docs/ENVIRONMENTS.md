# Isolated Model Environments

The full-source candidate includes SparseWorld-TC, CVT-Occ, FusionOcc and
SDGOcc with their installation profiles enabled. Hardware compatibility and
tested configurations are documented below.

Use **one environment per model**, even when recipes share versions. Their
`mmdet3d`, `model`, `dataset` and registry names collide. Never copy another
machine's `.so`, editable-install paths or `PYTHONPATH`.

## Installation Recipes

The release includes model-specific installation recipes, not machine-specific
package dumps. Historical package inventories are kept outside the public
source tree. Hardware-dependent kernels must be rebuilt in your own environment.

`environments/profiles.json` selects each model's pinned direct requirements,
source package and extension builds:

| Model | Python / PyTorch / CUDA | OpenMMLab | Additional Build |
| --- | --- | --- | --- |
| II-World | 3.9 / 2.8.0 / 12.8 | mmcv-full 1.7.2, mmdet 2.28.2, mmseg 0.30.0 | MMCV ops; local II-World on PYTHONPATH |
| GenieDrive | same, separate prefix | same | MMCV ops; local occ_gen on PYTHONPATH |
| SparseWorld-TC | same, separate prefix | same | BEV pool v2 and sparsedetectors/csrc |
| ALOcc | same, separate prefix | same | Its setup.py extensions |
| STCOcc | same, separate prefix | same | BEV pool extension and eager DVR ray-metric build |
| EFFOcc | same, separate prefix | same | BEV pools v1/v2; spconv-cu120 2.3.6, cumm-cu120 0.4.11 |
| FlashOcc (CARLA) | same, separate prefix | same | EFFOcc source, camera config; shared imports need spconv |
| CVT-Occ (Waymo) | same, separate prefix | same | Its own v1.0.0rc4 mmdet3d snapshot under dependencies/ plus CVT plugin; CVTOCC_WAYMO_ONLY=1 |
| OccWorld | 3.10 / 2.0.1 / 11.8 | mmcv 2.0.1, mmengine 0.8.4 | MMCV ops |
| COME | same, separate prefix | same | diffusers 0.24.0, timm 0.9.7, huggingface-hub 0.25.2 |
| DOME | same, separate prefix | same | Own diffusion/model registry |
| FusionOcc | 3.8 / 1.10.0 / 11.3 | mmcv-full 1.5.3, mmdet 2.25.1, mmseg 0.25.0 | BEV pool v2 and torch-scatter |
| SDGOcc | legacy native recipe | See its doc/install.md | Nested mmdetection3d, projects and bundled NATTEN; automatic refreshed install disabled |

CUDA 11.x profiles are **not RTX 5090 recipes**. FusionOcc's legacy profile is
for A100-class reproduction, not H100. Modern 12.8 profiles still require each
model's kernel tests on the target GPU. A spconv wheel's CUDA suffix alone does
not certify support for a new architecture.

The diffusion recipe uses Python 3.10 for the shared helpers. OpenCV/YAPF pins
avoid dependency conflicts. These are **pinned direct requirements**, not transitive lockfiles
or full-paper numerical reproduction certificates. The six forecasters and
ALOcc/STCOcc/CVT-Occ/EFFOcc/FlashOcc have passed small A100 real-checkpoint gates
in new prefixes. STCOcc used the existing experiment checkpoint `iter_63288.pth`,
not an asserted official model-zoo download.
FusionOcc/SDGOcc gates used isolated clones of a working legacy environment
with extensions rebuilt against the release sources, not fresh recipe solves.
See [the validation scope](CODE_RELEASE_STATUS.md#validation-scope).

## Install One Model

Prerequisites: Conda, a compatible NVIDIA driver, the matching CUDA toolkit
with nvcc, a compatible C++ compiler and temporary build space. PyTorch CUDA
wheels do not include the compiler. Do not build the 12.8 recipe with CUDA 13.0.

From the release root, using Python 3.10+ for the recipe generator:

```bash
export OCCSTRESS_CODE_ROOT="$PWD"
python tools/install_environment.py --model iiworld \
  --prefix /path/to/envs/occstress-iiworld \
  --cuda-home /path/to/cuda-12.8 --arch 8.0
```

This prints commands only. Add `--execute` to install into a **new prefix**;
existing prefixes are rejected. Use `--arch 9.0` for H100, or `12.0` for a
5090 with the modern recipe. Build only the needed architecture to save time.
MAX_JOBS=4 limits build memory. The recipe installs PyTorch first, builds MMCV
against it, then builds model extensions without dependency auto-upgrades.
No dataset/checkpoint is fetched.

Conda creation explicitly uses `conda-forge` without inherited channels;
`--conda-channel` can select another approved channel. The installer does not
accept channel terms or change global Conda configuration for you.
On HPC systems with Intel compiler modules, select a CUDA-compatible GCC pair
using `--cc /path/to/gcc --cxx /path/to/g++ --clean-build-flags`. This clears
module-specific C/C++/linker flags only inside installation subprocesses,
without changing the allocation or your shell modules. The diffusion recipe
pins matplotlib 3.5.2 to satisfy nuscenes-devkit 1.1.10, includes pandas for the
dataset samplers, and uses xformers 0.0.22 with PyTorch 2.0.1 for COME/DOME.
Ninja 1.13.0 avoids the older wheel's platform-tag validation failure.

Model forks are loaded directly from their own directory; extensions are built
in place. Their old setup.py dependency declarations are not installed as
editable metadata: some still require NumPy/Numba/networkx versions incompatible
with the recorded modern runtime. Use only the selected recipe requirements.

For COME use a different prefix, `--model come`, CUDA 11.8 and a supported GPU.
No RTX 5090 diffusion recipe is provided. Validate any alternative PyTorch
build with clean/corruption gates; do not silently substitute a nightly build.

SDGOcc is an explicit native-install exception: follow
`EXIST/3D/SDGOCC/doc/install.md` in its own prefix, build `mmdetection3d` and
`projects` there, and build the bundled NATTEN extensions with
`python setup.py build_ext --inplace` under `projects/natten/src`.
Add its root, `mmdetection3d`, `projects`, and `projects/natten/src` as absolute
PYTHONPATH entries and pass its native gates. Its vendored version assertion does
not accept the refreshed 1.7.2 profile. No automated refreshed install is claimed.

## Import And Kernel Gates

Use the model environment's Python, not base Python:

```bash
/path/to/envs/occstress-iiworld/bin/python tools/check_environment.py --model iiworld
/path/to/envs/occstress-iiworld/bin/python tools/check_environment.py --model iiworld --cuda
```

By default GPUs are hidden; source ownership, registries and versions are
checked. `--cuda` tests MMCV NMS and, for COME/DOME, xformers attention,
**not** every model kernel. Also exercise
EFFOcc's spconv/BEV pools, SparseWorld's sampling kernel and FusionOcc's
torch-scatter/BEV pool before use. Rebuild extensions, never borrow binaries.
STCOcc's native dataset import eagerly compiles the DVR ray-metric CUDA module;
its default CPU check therefore checks package ownership/versions only. Registry
loading is deferred to the explicit `--cuda` gate for that model.

DOME's native evaluator accepts `--deterministic` for strict migration and
repeatability comparisons. It disables cuDNN autotuning and requests
deterministic Torch algorithms; its selection is recorded in the output.
The default retains the historical runtime policy. Seeded diffusion alone does
not guarantee bitwise repeatability when convolution autotuning is enabled.

COME and OccWorld also accept opt-in `--deterministic`, which requests
deterministic cuDNN and disables cuDNN benchmarking. Their actual cuDNN flags
are recorded in result JSON. This is not a promise that every CUDA operation
is deterministic; unlike DOME's option it does not globally enable Torch's
deterministic-algorithm checks. Keep defaults unchanged for historical runs.

EFFOcc CARLA on A100 was checked with
`MMDET3D_SPCONV_FORCE_TRAIN_FORWARD=1`, its existing fallback for unavailable
inference-only sparse-convolution kernels. Only sparse convolutions use that
forward path; normalization remains in evaluation mode. Do not use
`model.train()` as a workaround or assume this validates another architecture.

Then verify checkpoint keys, run small clean/corruption anchor sets, check
targets/classes/controls/masks and compare recorded references. PyTorch 2.6+
may need explicit trusted legacy checkpoint handling; do not disable safe
loading for untrusted pickle files.

`tools/run_formal.py` resets PYTHONPATH to one model plus shared helpers and
rejects old temporal flags/config overrides. It defaults to dry-run. See
[formal evaluation](FORMAL_EVALUATION.md). Source/CPU checks do not certify
fresh solving, checkpoints, CUDA behavior or numerical reproduction.
