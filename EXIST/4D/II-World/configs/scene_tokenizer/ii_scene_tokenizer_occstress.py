_base_ = ['./ii_scene_tokenizer_4f.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.nuscenes_occstress_world_dataset',
        'mmdet3d.datasets.pipelines.loading_occstress',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_occstress',
    ],
    allow_failed_imports=False)

import os
import sys
from pathlib import Path

code_root = str(Path(os.environ.get('OCCSTRESS_CODE_ROOT', '../../..')).resolve())
if code_root not in sys.path:
    sys.path.insert(0, code_root)
from scripts.occstress_layout import dataset_occstress_root, resolve_manual_protocol_path

occstress_root = str(dataset_occstress_root(root=code_root))
protocol_name = os.environ.get('PROTOCOL_NAME', 'clean_H4_F6_val_backbone')
protocol_path = os.environ.get('PROTOCOL_PATH') or str(
    resolve_manual_protocol_path(protocol_name, root=code_root))
occstress_save_root = os.environ.get('OCCSTRESS_IIWORLD_SAVE_ROOT', str(Path(occstress_root) / 'cache/tokens' / protocol_name))
merged_ann_file = os.environ.get('IIWORLD_BASE_INFO', str(
    Path(code_root) / 'data/nuscenes/world-nuscenes_infos_trainval.pkl'))
del os, sys, Path, dataset_occstress_root, resolve_manual_protocol_path
dataset_name = 'occ3d'
eval_metric = 'miou'
occ_class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                   'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                   'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                   'vegetation', 'free']
bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

dataset_type = 'OccStressNuScenesTokenizerDataset'
data_root = 'data/nuscenes/'
samples_per_gpu = 1
workers_per_gpu = 0

collect_meta_keys = (
    'sample_idx', 'pts_filename', 'index', 'sequence_group_idx',
    'curr_to_prev_ego_rt', 'start_of_sequence', 'can_bus', 'bda_mat',
    'scene_name', 'occ_path', 'ego_from_sensor', 'protocol_sample_id',
    'occstress_save_token', 'anchor_token', 'occstress_occ_source', 'occstress_corruption')

model = dict(
    type='OccStressIISceneTokenizer',
    save_root=occstress_save_root,
    save_only_anchor=True,
)

train_pipeline = [
    dict(type='OccStressLoadStreamOcc3D', to_long=True),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='Collect3D', keys=['voxel_semantics'], meta_keys=collect_meta_keys),
]

test_pipeline = train_pipeline

share_data_config = dict(
    type=dataset_type,
    protocol_path=protocol_path,
    ann_file=merged_ann_file,
    classes=occ_class_names,
    dataset_name=dataset_name,
    eval_metric=eval_metric,
    use_sequence_group_flag=False,
)

test_data_config = dict(
    data_root=data_root,
    pipeline=test_pipeline,
    test_mode=True,
)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        data_root=data_root,
        protocol_path=protocol_path,
        ann_file=merged_ann_file,
        pipeline=train_pipeline,
        classes=occ_class_names,
        test_mode=False,
        dataset_name=dataset_name,
        eval_metric=eval_metric,
        use_sequence_group_flag=False,
    ),
    val=test_data_config,
    test=test_data_config,
)

for key in ['val', 'test']:
    data[key].update(share_data_config)
