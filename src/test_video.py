import cv2
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
# 2. Open video
# --------------------------------------------------

video_path = "videos/input.mov"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(
        f"Could not open video: {video_path}"
    )


# --------------------------------------------------
# 3. Video information
# --------------------------------------------------

fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


print()
print("=" * 50)
print("VIDEO INFORMATION")
print("=" * 50)

print("FPS:", fps)
print("Total frames:", total_frames)
print("Resolution:", width, "x", height)

duration = total_frames / fps if fps > 0 else 0

print("Duration:", round(duration, 2), "seconds")


# --------------------------------------------------
# 4. Test first 100 frames
# --------------------------------------------------

print()
print("Testing face detection...")
print()


frame_number = 0

while frame_number < 100:

    success, frame = cap.read()

    if not success:
        break

    faces = app.get(frame)

    print(
        f"Frame {frame_number}: "
        f"{len(faces)} face(s)"
    )

    frame_number += 1


# --------------------------------------------------
# 5. Cleanup
# --------------------------------------------------

cap.release()

print()
print("Video test completed.")