_base_ = ['./ii_scene_tokenizer_occstress.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.nuscenes_occstress_world_dataset',
        'mmdet3d.datasets.pipelines.loading_occstress',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_occstress_history_consensus',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_vector_quantizer_history_consensus_fusion',
    ],
    allow_failed_imports=False)

protocol_name = 'semantic_easy_history_k1_H4_F6_val_backbone'
protocol_path = 'data/OccStress/protocols/semantic_easy_history_k1_H4_F6_val_backbone.pkl'
occstress_save_root = 'data/OccStress/save_dir_history_consensus/semantic_easy_history_k1_H4_F6_val_backbone'
merged_ann_file = 'data/nuscenes/world-nuscenes_infos_trainval.pkl'

model = dict(
    type='OccStressIISceneTokenizerHistoryConsensus',
    save_root=occstress_save_root,
    save_only_anchor=True,
    vq=dict(
        type='IntraInterVectorQuantizerHistoryConsensusFusion',
        n_e=512,
        e_dim=128,
        beta=1.0,
        z_channels=128,
        recover_time=4,
        use_voxel=False,
        recency_decay=0.35,
        current_trust_threshold=0.50,
        current_trust_scale=6.0,
        min_current_weight=0.15,
    ),
)
