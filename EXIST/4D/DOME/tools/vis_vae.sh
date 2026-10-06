# export CUDA_VISIBLE_DEVICES=5

cfg=./config/train_occvae.py
dir=./work_dir/occ_vae

vae_ckpt=ckpts/occvae_latest.pth

python tools/visualize_demo_vae.py \
    --py-config $cfg \
    --work-dir $dir \
    --resume-from $vae_ckpt \
    --export_pcd

