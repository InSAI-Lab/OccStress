_base_ = ['./ii_scene_tokenizer_occstress_symmetric_recency.py']

model = dict(
    compare_history_at_test=True,
    eval_aligned_history=True,
    vote_mode_at_test='dynamic_gated',
    vote_static_class_ids=[0, 1, 8, 11, 12, 13, 14, 15, 16],
    vote_dynamic_class_ids=[2, 3, 4, 5, 6, 7, 9, 10],
    vote_history_min_agree=2,
    vote_neighborhood_radius=1,
)
