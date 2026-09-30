import cv2
import numpy as np
from insightface.app import FaceAnalysis


# --------------------------------------------------
# 1. Load InsightFace
# --------------------------------------------------

print("Loading face model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("Model loaded!")


# --------------------------------------------------
# 2. Function to get face embedding
# --------------------------------------------------

def get_face_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    faces = app.get(image)

    if len(faces) == 0:
        raise ValueError(
            f"No face detected: {image_path}"
        )

    if len(faces) > 1:
        raise ValueError(
            f"Multiple faces detected: {image_path}"
        )

    face = faces[0]

    # ArcFace embedding
    embedding = face.embedding

    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# --------------------------------------------------
# 3. Reference images
# --------------------------------------------------
image_paths = [
    "reference_images/image1.png",
    "reference_images/image2_upright.png",
    "reference_images/image3.png",
    "reference_images/image4.png",
    "reference_images/image5.png"

]

# --------------------------------------------------
# 4. Generate embeddings
# --------------------------------------------------

embeddings = {}

for image_path in image_paths:

    print("Processing:", image_path)

    embeddings[image_path] = get_face_embedding(
        image_path
    )


# --------------------------------------------------
# 5. Cosine similarity
# --------------------------------------------------

def cosine_similarity(a, b):

    return np.dot(a, b)


# --------------------------------------------------
# 6. Compare all pairs
# --------------------------------------------------

print()
print("=" * 60)
print("ARCFACE SIMILARITY")
print("=" * 60)

for i in range(len(image_paths)):

    for j in range(i + 1, len(image_paths)):

        image_a = image_paths[i]
        image_b = image_paths[j]

        similarity = cosine_similarity(
            embeddings[image_a],
            embeddings[image_b]
        )

        print(
            f"{image_a} <-> {image_b} : "
            f"{similarity:.4f}"
        )
