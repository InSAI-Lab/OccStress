_base_ = ['./ii_generate_world_original_evalreg.py']

stage3_decoder_ckpt = (
    'work_dirs/ii_stage3_decoder_pred_only_0p5s_baseline_fp32/'
    'iter_4000_ema.pth'
)

model = dict(
    vqvae_checkpoint=stage3_decoder_ckpt,
)
