_base_ = ['./ii_scene_tokenizer_4f_exact_static_occ_slot_aligned_baseline_cached_train_v1.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pipelines.loading_oracle_flow',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_currbev_kalman_exact_static_residual_bank_cached_occ_slots_history_warmup',
        'mmdet3d.core.hook.history_warmup_step',
    ],
    allow_failed_imports=False)

model = dict(
    type='IISceneTokenizerCurrBevKalmanExactStaticResidualBankCachedOccSlotsHistoryWarmup',
    history_warmup_enable=True,
    history_zero_until=2000,
    history_full_until=8000,
    history_switch_mode='batch',
    history_switch_seed=0,
)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=3998,
    ),
    dict(
        type='HistoryWarmupStepHook',
        priority='VERY_HIGH',
    ),
]
