_base_ = ['./flashocc-fusion-waymoD1-1f.py']

# The paper uses global batch 16 on four A6000 GPUs. RTX 5090 testing shows
# that the same four samples per GPU fit with substantial memory headroom.
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    persistent_workers=True,
)
