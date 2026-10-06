# export CUDA_VISIBLE_DEVICES=5


cfg=./config/train_dome.py
dir=./work_dir/dome
vae_ckpt=ckpts/occvae_latest.pth


python tools/train_diffusion.py \
    --py-config $cfg \
    --work-dir $dir 