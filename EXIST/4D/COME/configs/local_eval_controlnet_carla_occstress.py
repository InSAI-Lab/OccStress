# OccStress adapter/portability modifications; see docs/source-imports.json.
import os
from pathlib import Path

_base_ = ['./local_eval_controlnet_occstress.py']

protocol_path = str(Path(os.environ['OCCSTRESS_CARLA_PROTOCOL']).resolve())
imageset = str(Path(os.environ['OCCSTRESS_CARLA_BASE_INFO']).resolve())
max_samples_value = os.environ.get('OCCSTRESS_CARLA_MAX_SAMPLES')
max_samples = int(max_samples_value) if max_samples_value else None

anchor_deterministic_seed = True
anchor_seed_base = int(os.environ.get('OCCSTRESS_CARLA_SEED', '42'))
compute_generative_metrics = False

carla_dataset_config = dict(
    type='OccStressCARLASceneDataset',
    data_path='',
    return_len=10,
    offset=0,
    imageset=imageset,
    protocol_path=protocol_path,
    test_mode=True,
    new_rel_pose=True,
    max_samples=max_samples,
)
train_dataset_config = carla_dataset_config
val_dataset_config = carla_dataset_config
train_loader = dict(
    batch_size=int(os.environ.get('OCCSTRESS_CARLA_BATCH_SIZE', '1')),
    num_workers=int(os.environ.get('OCCSTRESS_CARLA_NUM_WORKERS', '2')),
    shuffle=False,
)
val_loader = train_loader

del os, Path
