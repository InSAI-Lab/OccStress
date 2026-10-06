_base_ = ['./ii_generate_world_original_evalreg.py']

custom_imports = dict(
    imports=[
        'mmdet3d.models.ii_world.world_model.ii_world_robust_finetune',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='II_WorldRobustFinetune',
    vqvae_checkpoint='ckpts/ii_scene_tokenizer_4f.pth',
    robust_finetune=dict(
        enabled=True,
        corr_ratio=0.5,
        seed=3407,
        gate_bias=-4.0,
        identity_weight=0.05,
        denoise_weight=0.1,
        gate_weight=0.0005,
        noise_std=0.03,
        channel_drop_prob=0.08,
        block_drop_prob=0.65,
        min_block=4,
        max_block=14,
        spatial_shift_prob=0.35,
        max_shift=4,
    ),
)
