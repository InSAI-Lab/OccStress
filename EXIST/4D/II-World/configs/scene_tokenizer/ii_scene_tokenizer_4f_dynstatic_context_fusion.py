_base_ = ['./ii_scene_tokenizer_4f.py']

model = dict(
    vq=dict(
        type='IntraInterVectorQuantizerDynStaticContextFusion',
        gate_hidden=128,
        dynamic_gate_scale=0.25,
        context_mix=0.5,
    ),
)
