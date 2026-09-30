import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# 2. Load DINOv2
# --------------------------------------------------

print("Loading DINOv2...")

model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
)

model = model.to(device)
model.eval()

print("Model loaded!")


# --------------------------------------------------
# 3. Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 4. Generate embedding
# --------------------------------------------------

def get_embedding(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        embedding = model(image_tensor)

    # Normalize embedding
    embedding = F.normalize(
        embedding,
        p=2,
        dim=1
    )

    return embedding


# --------------------------------------------------
# 5. Load reference images
# --------------------------------------------------

folder = Path("reference_images")

images = sorted([
    file for file in folder.iterdir()
    if file.suffix.lower() in [
        ".jpg",
        ".jpeg",
        ".png"
    ]
])


# --------------------------------------------------
# 6. Generate embeddings
# --------------------------------------------------

embeddings = {}

for image in images:

    print("Embedding:", image.name)

    embeddings[image.name] = get_embedding(image)


# --------------------------------------------------
# 7. Compare every image with every other image
# --------------------------------------------------

print()
print("=" * 60)
print("SIMILARITY RESULTS")
print("=" * 60)

for i in range(len(images)):

    for j in range(i + 1, len(images)):

        image_a = images[i].name
        image_b = images[j].name

        embedding_a = embeddings[image_a]
        embedding_b = embeddings[image_b]

        similarity = F.cosine_similarity(
            embedding_a,
            embedding_b
        ).item()

        print(
            f"{image_a} <-> {image_b} : "
            f"{similarity:.4f}"
        )