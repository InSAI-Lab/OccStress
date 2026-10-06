# export CUDA_VISIBLE_DEVICES=5

cfg=./config/train_dome.py
dir=./work_dir/dome

vae_ckpt=ckpts/occvae_latest.pth
ckpt=ckpts/dome_latest.pth


python tools/visualize_demo.py \
    --py-config $cfg \
    --work-dir $dir \
    --resume-from $ckpt \
    --vae-resume-from $vae_ckpt \

