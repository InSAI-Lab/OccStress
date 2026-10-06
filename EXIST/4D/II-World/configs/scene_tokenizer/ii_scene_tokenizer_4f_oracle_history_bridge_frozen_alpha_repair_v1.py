_base_ = ['./ii_scene_tokenizer_4f_oracle_history_bridge_frozen_alpha.py']

model = dict(
    oracle_bridge_repair_loss_weight=2.0,
    oracle_bridge_keep_baseline_loss_weight=0.5,
    oracle_bridge_bad_threshold=0.5,
    oracle_bridge_dynamic_repair_boost=1.0,
)
