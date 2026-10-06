_base_ = ["./train_occworld.py"]

port = 25111
load_from = "out/occworld_vqvae_4gpu_bs2/latest.pth"

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
