_base_ = ['./alocc_3d_r50_256x704_bevdet_preatrain_16f.py']

num_gpus = 1
samples_per_gpu = 1

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=0,
)

runner = dict(type='IterBasedRunner', max_iters=1)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
