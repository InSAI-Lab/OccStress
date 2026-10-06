_base_ = ['./ii_scene_tokenizer_4f_currbev_kalman_exact_static_residual_bank_frozen_v1.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pipelines.loading_oracle_flow',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_currbev_kalman_exact_static_residual_bank_cached',
    ],
    allow_failed_imports=False)

model = dict(
    type='IISceneTokenizerCurrBevKalmanExactStaticResidualBankCached',
    unified_motion_use_dynamic_gate=False,
    unified_motion_dynamic_transport='scatter',
)
