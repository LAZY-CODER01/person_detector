import torch
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

print("DINOv2 loaded successfully!")


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
# 4. Function to generate embedding
# --------------------------------------------------

def get_embedding(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        embedding = model(image_tensor)

    # Move back to CPU
    embedding = embedding.cpu()

    return embedding


# --------------------------------------------------
# 5. Find all reference images
# --------------------------------------------------

reference_folder = Path("reference_images")

image_files = [
    file for file in reference_folder.iterdir()
    if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

print()
print("Reference images found:", len(image_files))


# --------------------------------------------------
# 6. Generate embeddings
# --------------------------------------------------

reference_embeddings = {}

for image_file in image_files:

    print(f"Processing: {image_file.name}")

    embedding = get_embedding(image_file)

    reference_embeddings[image_file.name] = embedding

    print(
        f"Embedding shape: {embedding.shape}"
    )


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print()
print("=" * 50)
print("REFERENCE EMBEDDINGS")
print("=" * 50)

for image_name, embedding in reference_embeddings.items():

    print(
        f"{image_name} -> {embedding.shape}"
    )

print()
print("Total images embedded:", len(reference_embeddings))