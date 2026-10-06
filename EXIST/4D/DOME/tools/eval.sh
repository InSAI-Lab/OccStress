
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=7


cfg=./config/train_dome.py
dir=./work_dir/dome
ckpt=ckpts/dome_latest.pth
vae_ckpt=ckpts/occvae_latest.pth


python tools/eval_metric.py \
--py-config $cfg \
--work-dir $dir \
--resume-from $ckpt \
--vae-resume-from $vae_ckpt 
