_base_ = ['./ii_scene_tokenizer_occstress_symmetric_recency_static_vote.py']

data = dict(
    test=dict(
        protocol_record_limit=64,
        protocol_record_offset=0,
    ),
)
