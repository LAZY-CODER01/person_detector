import cv2
from insightface.app import FaceAnalysis


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


image_paths = [
    "reference_images/image1.png",
    "reference_images/image2_upright.png",
    "reference_images/image3.png",
    "reference_images/image4.png",
    "reference_images/image5.png",
    
]


for image_path in image_paths:

    print()
    print("Processing:", image_path)

    image = cv2.imread(image_path)

    faces = app.get(image)

    print("Faces detected:", len(faces))

    for i, face in enumerate(faces):

        print(
            f"Face {i + 1}: "
            f"det_score={face.det_score:.4f}"
        )

        print(
            "Embedding shape:",
            face.embedding.shape
        )