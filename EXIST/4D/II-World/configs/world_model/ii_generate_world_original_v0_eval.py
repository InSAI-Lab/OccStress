_base_ = ['./ii_generate_world.py']

custom_imports = dict(
    imports=[
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_v0_original',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_vector_quantizer_v0_original',
        'mmdet3d.models.losses.focal_loss_torch',
    ],
    allow_failed_imports=False,
)

model = dict(
    vqvae_checkpoint='ckpts/ii_scene_tokenizer_4f.pth',
    vqvae=dict(
        type='IISceneTokenizerV0Original',
        vq=dict(
            type='IntraInterVectorQuantizerV0Original',
            recover_time=4,
            use_voxel=False,
        ),
        focal_loss=dict(
            type='CustomFocalLossTorch',
            loss_weight=10.0,
        ),
    ),
)

