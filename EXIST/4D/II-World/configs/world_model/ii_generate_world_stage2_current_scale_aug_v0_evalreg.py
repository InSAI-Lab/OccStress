_base_ = ['./ii_generate_world_original_evalreg.py']

custom_imports = dict(
    imports=[
        'mmdet3d.models.ii_world.world_model.ii_world_current_scale_aug',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='II_WorldCurrentScaleAug',
    vqvae_checkpoint='ckpts/ii_scene_tokenizer_4f.pth',
    scale_aug=dict(
        enabled=False,
        apply_at_test=False,
    ),
)
