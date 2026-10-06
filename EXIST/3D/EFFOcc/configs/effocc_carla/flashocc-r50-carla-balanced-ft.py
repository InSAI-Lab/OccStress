_base_ = ["./flashocc-r50-carla-10hz.py"]

# Inverse-sqrt train-set frequencies, capped at 20 before expected-value
# normalization. The source audit is carla-train-camera-class-frequencies.json.
carla_class_weights = [
    0.0, 20.16009764, 0.0, 20.16009764, 0.0, 0.0,
    18.29586054, 8.66388539, 11.32518331, 9.45966316, 0.78683044,
]

model = dict(
    occ_head=dict(loss_occ=dict(class_weight=carla_class_weights)),
)
optimizer = dict(type="AdamW", lr=5e-5, weight_decay=1e-2)
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=100,
    warmup_ratio=0.1,
    step=[6],
)
runner = dict(type="EpochBasedRunner", max_epochs=8)
checkpoint_config = dict(interval=4, max_keep_ckpts=2)
evaluation = dict(interval=8)

