_base_ = ['./ii_scene_tokenizer_4f.py']

model = dict(
    vq=dict(
        type='IntraInterVectorQuantizerHybridFusion',
        gate_hidden=128,
        dynamic_floor=0.2,
    )
)
