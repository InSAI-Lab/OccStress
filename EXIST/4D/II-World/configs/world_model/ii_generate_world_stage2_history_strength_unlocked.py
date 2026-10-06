_base_ = ['./ii_generate_world_stage2_history_selective_gate_v1.py']

model = dict(
    history_gate=dict(
        history_strength_init=0.08,
        current_strength_init=0.0,
        history_gate_bias=1.0,
        current_gate_bias=-4.0,
        current_identity_weight=0.08,
        current_denoise_weight=0.12,
        current_gate_supervision_weight=0.02,
        history_gate_supervision_weight=0.02,
        strength_reg_weight=0.0001,
        clean_consistency_weight=0.0,
    ),
)
