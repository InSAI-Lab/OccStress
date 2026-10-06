_base_ = ['./ii_scene_tokenizer_4f.py']

model = dict(
    vq=dict(
        type='IntraInterVectorQuantizerDynStaticGateFusion',
        gate_hidden=128,
        dynamic_gate_scale=0.25,
    ),
)
