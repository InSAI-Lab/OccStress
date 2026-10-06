_base_ = ["./train_occworld.py"]

port = 25112
load_from = "out/occworld_vqvae_4gpu_bs2/latest.pth"
accumulation_steps = 2

train_loader = dict(
    batch_size=1,
    shuffle=True,
    num_workers=2,
)

val_loader = dict(
    batch_size=1,
    shuffle=False,
    num_workers=2,
)
