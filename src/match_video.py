import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_INPUT = "videos/input.mov"
VIDEO_OUTPUT = "output/matched_video.mp4"

REFERENCE_IMAGES = [
    "reference_images/image1.png",
    "reference_images/image2_upright.png",
    "reference_images/image3.png"
]

# Initial threshold based on our image tests.
# We will calibrate this later.
SIMILARITY_THRESHOLD = 0.35

# Process every frame initially.
# Later we can optimize this.
FRAME_SKIP = 1


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded!")


# ============================================================
# FUNCTION: GET REFERENCE EMBEDDING
# ============================================================

def get_reference_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read: {image_path}"
        )

    faces = app.get(image)

    if len(faces) == 0:
        raise ValueError(
            f"No face found in: {image_path}"
        )

    # For our reference images we expect one face.
    face = faces[0]

    embedding = face.embedding

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# ============================================================
# LOAD REFERENCE EMBEDDINGS
# ============================================================

print()
print("Creating reference embeddings...")

reference_embeddings = []

for image_path in REFERENCE_IMAGES:

    print("Processing:", image_path)

    embedding = get_reference_embedding(
        image_path
    )

    reference_embeddings.append(
        embedding
    )

print(
    "Loaded",
    len(reference_embeddings),
    "reference embeddings."
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    return float(
        np.dot(a, b)
    )


# ============================================================
# OPEN VIDEO
# ============================================================

print()
print("Opening video...")

cap = cv2.VideoCapture(VIDEO_INPUT)

if not cap.isOpened():
    raise ValueError(
        f"Could not open video: {VIDEO_INPUT}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print()
print("=" * 60)
print("VIDEO")
print("=" * 60)

print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Total frames:", total_frames)


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    VIDEO_OUTPUT,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

matched_frames = 0

print()
print("Processing video...")
print()


while True:

    success, frame = cap.read()

    if not success:
        break

    # --------------------------------------------
    # Skip frames if configured
    # --------------------------------------------

    if frame_number % FRAME_SKIP != 0:

        out.write(frame)

        frame_number += 1

        continue


    # --------------------------------------------
    # Detect faces
    # --------------------------------------------

    faces = app.get(frame)


    # --------------------------------------------
    # Process every detected face
    # --------------------------------------------

    for face in faces:

        embedding = face.embedding

        embedding = (
            embedding /
            np.linalg.norm(embedding)
        )


        # ----------------------------------------
        # Compare against all reference images
        # ----------------------------------------

        similarities = []

        for reference_embedding in reference_embeddings:

            similarity = cosine_similarity(
                embedding,
                reference_embedding
            )

            similarities.append(
                similarity
            )


        # Best reference match

        best_similarity = max(
            similarities
        )


        # ----------------------------------------
        # Bounding box
        # ----------------------------------------

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox


        # Keep coordinates inside frame

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)


        # ----------------------------------------
        # MATCH
        # ----------------------------------------

        if best_similarity >= SIMILARITY_THRESHOLD:

            matched_frames += 1

            label = (
                f"MATCH "
                f"{best_similarity:.2f}"
            )

            # Green box
            color = (0, 255, 0)

        else:

            label = (
                f"Unknown "
                f"{best_similarity:.2f}"
            )

            # Red box
            color = (0, 0, 255)


        # ----------------------------------------
        # Draw bounding box
        # ----------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        # ----------------------------------------
        # Draw label
        # ----------------------------------------

        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


    # --------------------------------------------
    # Write frame
    # --------------------------------------------

    out.write(frame)


    # --------------------------------------------
    # Progress
    # --------------------------------------------

    if frame_number % 30 == 0:

        print(
            f"Processed "
            f"{frame_number}/{total_frames}"
        )


    frame_number += 1


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()


print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(
    "Processed frames:",
    frame_number
)

print(
    "Matched frames:",
    matched_frames
)

print(
    "Output:",
    VIDEO_OUTPUT
)