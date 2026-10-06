_base_ = ["./train_vqvae.py"]

port = 25110

train_loader = dict(
    batch_size=2,
    shuffle=True,
    num_workers=2,
)

val_loader = dict(
    batch_size=2,
    shuffle=False,
    num_workers=2,
)
