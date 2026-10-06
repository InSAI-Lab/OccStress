_base_ = ['./ii_scene_tokenizer_4f.py']

model = dict(
    vq=dict(
        type='IntraInterVectorQuantizerRefineFusion',
        refine_hidden=128,
    ),
)
