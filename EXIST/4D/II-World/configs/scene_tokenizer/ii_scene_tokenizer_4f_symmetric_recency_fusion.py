_base_ = ['./ii_scene_tokenizer_4f.py']

model = dict(
    vq=dict(
        type='IntraInterVectorQuantizerSymmetricRecencyFusion',
        gate_hidden=128,
        recency_decay=0.35,
        use_learnable_recency=True,
    ),
)
