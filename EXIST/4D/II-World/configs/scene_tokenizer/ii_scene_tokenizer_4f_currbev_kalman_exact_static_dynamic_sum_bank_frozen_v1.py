_base_ = ['./ii_scene_tokenizer_4f_currbev_kalman_exact_static_residual_bank_frozen_v1.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pipelines.loading_oracle_flow',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_currbev_kalman_exact_static_dynamic_sum_bank',
    ],
    allow_failed_imports=False)

model = dict(
    type='IISceneTokenizerCurrBevKalmanExactStaticDynamicSumBank',
)
