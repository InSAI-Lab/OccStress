_base_ = ['./ii_scene_tokenizer_4f.py']

custom_imports = dict(
    imports=[
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_currlatent_bilinear_slots',
    ],
    allow_failed_imports=False)

# Baseline ablation:
# keep the whole tokenizer/VQ/decoder training recipe unchanged,
# but replace the history-latent source of sampled_bev with repeated current
# latent under the same baseline bilinear warp operator.
model = dict(
    type='IISceneTokenizerCurrLatentBilinearSlots',
)
