# Data And Weights

[configs/resources.json](../configs/resources.json) is the editable resource
registry. Do not put credentials, checkpoints or dataset payloads in the code tree.

## Dataset Packages

Repository: [insailab/OccStress](https://huggingface.co/datasets/insailab/OccStress).
The live [packages.json](https://huggingface.co/datasets/insailab/OccStress/resolve/main/packages.json)
lists metadata and independently downloadable `.tar.zst` shards, their sizes,
checksums and availability. Uploads can be partial: a repository link or an
uploaded metadata archive does **not** establish full dataset availability.
Only entries with `status=available` are ready.

Select one dataset and track before downloading. Run from the code checkout:

```bash
python -m pip install -e '.[download]'
python tools/list_data_packages.py --fetch --dataset waymo --track manual > selection.json
python tools/download_data.py --selection selection.json \
  --output-dir /datasets/OccStress-download
```

The selector fetches only the small index and resolves the requested revision
to an immutable dataset commit. It exits nonzero if any selected shard is
pending; the downloader also refuses pending selections. The downloader uses
the official Hub client, reuses valid local archives, resumes interrupted
downloads and checks each compressed archive's SHA256 and size. It does not
rehash individual voxels. Keep `selection.json` and `download-receipt.json`
with the experiment record.

Use `--dataset nuscenes|waymo|carla` and `--track manual|upstream|all`.
To select one upstream source, add `--track upstream --source SUBTRACK/MODEL`,
for example `--source camera_fusion/effocc` for Waymo EFFOcc camera stress.
For a saved index, use `--index /path/to/packages.json --revision COMMIT`
instead of `--fetch`; use the commit from which that index was obtained.
Do not mix different dataset revisions in one download or extraction directory.

Local verification is also available without network access:

```bash
python tools/download_data.py --selection selection.json \
  --output-dir /datasets/OccStress-download --verify-only
```

After successful download/verification, extract all selected metadata and
payload archives into the **same empty root**. The following uses Bash,
GNU tar and the `zstd` executable; install them using your system package manager:

```bash
export OCCSTRESS_DATA_ROOT="/datasets/OccStress"
mkdir -p "$OCCSTRESS_DATA_ROOT"
set -euo pipefail
find /datasets/OccStress-download/metadata /datasets/OccStress-download/archives \
  -type f -name '*.tar.zst' -print0 | \
  while IFS= read -r -d '' archive; do
    tar --zstd -xf "$archive" --no-same-owner -C "$OCCSTRESS_DATA_ROOT"
  done
```

Archive members already start with `meta/`, `protocols/`, `occ/` or `events/`;
do not add another dataset-specific root or strip path components.
Use an empty versioned directory for a new data release; do not overlay versions.
Raw images, LiDAR, original GT and auxiliary model metadata are separate
[external dependencies](EXTERNAL_DATA_PREPARATION.md). Only load trusted PKLs.
After preparing the dependencies, validate the selected dataset (replace Waymo
as appropriate), then run the chosen model's small clean test from the
[evaluation guide](FORMAL_EVALUATION.md):

```bash
python tools/prepare_external_data.py check --dataset waymo --trust-pickle
```

## Checkpoints

Prefer official author downloads; no third-party weights are bundled or
silently mirrored. Upstream weights are **not required** to forecast from
released 3D occupancy exports.

| Method | Exact selection / configuration |
| --- | --- |
| GenieDrive | `genie_occ.pth`; `configs/world_model/vae_e2e_occstress.py`. Recorded runtime SHA matches the official file. |
| II-World | `ii_scene_tokenizer_4f.pth` and `ii_generate_world.pth`; `configs/world_model/ii_generate_world_occstress.py`. Runtime digests are recorded; validate author downloads against them. |
| COME | 4-input / 6-future / 3-second variant: `best_miou_world_model.pth`, `unet_past2s_future3s.pth`, `best_miou_controlnet.pth`, `occvae_latest.pth`; `configs/local_eval_controlnet_occstress.py`. Registry links point to the exact official files. Do not use the file marked `corrupted_do_not_download`, 8-second weights, small-model weights or BEV-layout variants. |
| DOME | `dome_latest.pth` and `occvae_latest.pth`; `config/train_dome.py`. Runtime digests are recorded; official folder identity remains to be checked. |
| OccWorld | Gate used `latest.pth` (756,041,219 bytes); `config/occworld_occstress.py`. Select with `OCCWORLD_CKPT`; a filename/size match is not proof of exact official identity. |
| Upstream re-export | Use the source/dataset-specific configuration in [UPSTREAM_EXPORTS.md](UPSTREAM_EXPORTS.md). STCOcc gates used `iter_63288.pth`, not an asserted official model-zoo checkpoint. Experiment-trained EFFOcc/FlashOcc files remain user-supplied until their mirrors are approved. |

The registry distinguishes a runtime digest from one supplied by the official
host. An `external_reference` remains a reference until the exact experiment
file is matched; an `expected_sha256` enables local checking without claiming
that the file was mirrored or that full-paper scores were reproduced:

```bash
python tools/check_checkpoint.py geniedrive /path/to/genie_occ.pth
python tools/check_checkpoint.py iiworld-tokenizer /path/to/ii_scene_tokenizer_4f.pth
```

The checker reads only the selected file, does not deserialize it, and fails
on a mismatch or unavailable checksum. It never hashes the full dataset.
Use the method-specific [formal launch instructions](FORMAL_EVALUATION.md) to
pass all required weights and controls, then start with a small clean smoke test.

## Release Checks

```bash
python tools/release_status.py
python tools/release_status.py --require-downloads
```

The first validates the source/registry. The second is stricter: it fails while
any registered resource is pending, uploading or identity-unverified, including
optional re-export weights and identity-unverified model references. It is **not**
the code-only release gate and performs no network requests.

Original source access agreements and [distribution terms](ASSET_DISTRIBUTION.md)
still apply. Google Drive is reserved for separately approved project mirrors.
