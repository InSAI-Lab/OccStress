_base_ = ['./ii_scene_tokenizer_4f.py']

model = dict(
    vq=dict(
        type='IntraInterVectorQuantizerGateContextFusion',
        gate_hidden=128,
        context_mix=0.5,
    ),
)
