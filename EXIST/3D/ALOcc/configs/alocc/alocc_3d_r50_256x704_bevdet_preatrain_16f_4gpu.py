_base_ = ['./alocc_3d_r50_256x704_bevdet_preatrain_16f.py']

num_gpus = 4
samples_per_gpu = 2
num_epochs = 12
checkpoint_epoch_interval = 1
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu) * 4.554)

data = dict(
    samples_per_gpu=samples_per_gpu,
)

runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
checkpoint_config = dict(interval=checkpoint_epoch_interval * num_iters_per_epoch)
evaluation = dict(interval=num_epochs * num_iters_per_epoch)
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=1 * num_iters_per_epoch,
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_iter=num_iters_per_epoch * 2,
    ),
    dict(
        type='FusionRateControlDepthHook',
        temporal_start_iter=0,
        temporal_end_iter=num_iters_per_epoch * 6,
    ),
]
