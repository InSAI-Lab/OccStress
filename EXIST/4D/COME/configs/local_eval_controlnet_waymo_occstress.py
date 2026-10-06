# OccStress adapter/portability modifications; see docs/source-imports.json.
import os
from pathlib import Path

_base_ = ['./local_eval_controlnet_occstress.py']

protocol_path = str(Path(os.environ['OCCSTRESS_WAYMO_PROTOCOL']).resolve())
imageset = str(Path(os.environ['OCCSTRESS_WAYMO_BASE_INFO']).resolve())
scene_shard = os.environ.get('OCCSTRESS_WAYMO_SCENE_SHARD')
max_samples_value = os.environ.get('OCCSTRESS_WAYMO_MAX_SAMPLES')
max_samples = int(max_samples_value) if max_samples_value else None

waymo_dataset_config = dict(
    type='OccStressWaymoSceneDataset',
    data_path='',
    return_len=10,
    offset=0,
    imageset=imageset,
    protocol_path=protocol_path,
    test_mode=True,
    new_rel_pose=True,
    scene_shard=scene_shard,
    max_samples=max_samples,
)

train_dataset_config = waymo_dataset_config
val_dataset_config = waymo_dataset_config
train_loader = dict(
    batch_size=int(os.environ.get('OCCSTRESS_WAYMO_BATCH_SIZE', '1')),
    num_workers=int(os.environ.get('OCCSTRESS_WAYMO_NUM_WORKERS', '2')),
    shuffle=False,
)
val_loader = train_loader
compute_generative_metrics = False

del os, Path
