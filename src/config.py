# =============================================================================
# config.py  –  All tunable parameters in one place.
# Change any value here and restart the app to apply it.
# =============================================================================

# ---------------------------------------------------------------------------
# Similarity threshold.
# ArcFace cosine similarity is in [-1, 1].  Values measured on this project:
#   same person  → 0.52 – 1.00
#   diff person  → -0.02 – 0.16
# 0.35 is a conservative starting point; calibrate upward if you see
# false-positives or downward if the target isn't being detected.
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD: float = 0.35

# ---------------------------------------------------------------------------
# Frame processing.
# 1  = process every frame (highest accuracy, highest CPU usage).
# 2  = process every other frame (good balance on most Macs).
# 3  = process 1 in 3 frames  (lighter, still smooth-looking).
# ---------------------------------------------------------------------------
PROCESS_EVERY_N_FRAMES: int = 1

# ---------------------------------------------------------------------------
# Temporal smoothing / confirmation.
# The target label won't flip to MATCH until this many recent frames have
# all exceeded the threshold.  Prevents a single noisy frame from triggering
# a match.  Set to 1 to disable smoothing.
# ---------------------------------------------------------------------------
MATCH_CONFIRMATION_FRAMES: int = 3

# ---------------------------------------------------------------------------
# Smoothing window size.
# We keep a rolling buffer of this many similarity scores per face position
# and use the mean for the match decision.
# ---------------------------------------------------------------------------
SIMILARITY_SMOOTHING_WINDOW: int = 5

# ---------------------------------------------------------------------------
# InsightFace model settings.
# buffalo_l  = high accuracy,  buffalo_s = faster but less accurate.
# Change det_size to (320, 320) for a speed boost on slower machines.
# ---------------------------------------------------------------------------
INSIGHTFACE_MODEL: str = "buffalo_l"
INSIGHTFACE_DET_SIZE: tuple = (640, 640)

# ---------------------------------------------------------------------------
# Visual display settings.
# ---------------------------------------------------------------------------
# Bounding box colours  (BGR because OpenCV uses BGR).
MATCH_BOX_COLOR: tuple = (0, 220, 0)      # bright green
UNKNOWN_BOX_COLOR: tuple = (0, 80, 220)   # blue

# Whether to draw bounding boxes and labels for non-target ("Unknown") faces.
SHOW_UNKNOWN_FACES: bool = True

# Font scale for bounding-box labels drawn on the video frame.
LABEL_FONT_SCALE: float = 0.65
LABEL_THICKNESS: int = 2

# ---------------------------------------------------------------------------
# Gradio server settings.
# ---------------------------------------------------------------------------
GRADIO_SERVER_PORT: int = 7860
GRADIO_SERVER_NAME: str = "127.0.0.1"
