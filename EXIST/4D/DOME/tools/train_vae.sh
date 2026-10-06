
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=7


cfg=./config/train_occvae.py
dir=./work_dir/occ_vae


python tools/train_vae.py \
    --py-config $cfg \
    --work-dir $dir 