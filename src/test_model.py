import torch

print("Loading DINOv2...")

model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
)

model.eval()

print("Model loaded successfully!")