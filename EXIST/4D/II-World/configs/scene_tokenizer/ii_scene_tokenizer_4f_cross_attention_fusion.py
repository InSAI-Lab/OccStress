_base_ = ['./ii_scene_tokenizer_4f.py']

model = dict(
    vq=dict(
        type='IntraInterVectorQuantizerCrossAttentionFusion',
        num_heads=4,
        attn_dropout=0.0,
        refine_hidden=128,
    ),
)
