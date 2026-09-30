"""
face_engine.py
==============
Core face-recognition logic.
All InsightFace calls are isolated here; the UI layer never touches the model
directly.  The module loads InsightFace ONCE at import time so every call to
process_frame() reuses the same loaded model.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# InsightFace is loaded here, at module-import time.
# This guarantees the model is NEVER recreated inside a frame-processing loop.
# ---------------------------------------------------------------------------
_face_app: Optional[FaceAnalysis] = None
_model_load_error: Optional[str] = None


def _load_model() -> None:
    """Load InsightFace buffalo_l once and store in the module-level singleton."""
    global _face_app, _model_load_error
    try:
        logger.info("Loading InsightFace model '%s' …", config.INSIGHTFACE_MODEL)
        _face_app = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL,
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=config.INSIGHTFACE_DET_SIZE)
        logger.info("InsightFace loaded successfully.")
    except Exception as exc:
        _model_load_error = str(exc)
        logger.exception("Failed to load InsightFace: %s", exc)


_load_model()


# ---------------------------------------------------------------------------
# Public helper: check model is ready
# ---------------------------------------------------------------------------

def model_ready() -> Tuple[bool, str]:
    """Return (True, "") if the model loaded OK, else (False, error_message)."""
    if _face_app is not None:
        return True, ""
    return False, _model_load_error or "Unknown model loading error."


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _normalize(embedding: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D embedding vector so cosine similarity = dot product."""
    norm = np.linalg.norm(embedding)
    if norm < 1e-10:
        return embedding
    return embedding / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two already-normalised vectors (scalar in [-1, 1])."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# FaceRecognitionEngine
# ---------------------------------------------------------------------------

class FaceRecognitionEngine:
    """
    Stateful face-recognition engine.

    Usage
    -----
    engine = FaceRecognitionEngine()

    # Point at a target once:
    status, message = engine.set_target(image_bgr)

    # Per frame:
    annotated_frame, results = engine.process_frame(frame_bgr)
    """

    def __init__(self) -> None:
        # Target person state
        self._target_embedding: Optional[np.ndarray] = None
        self._target_set: bool = False

        # Temporal smoothing: a deque of recent similarity scores (one deque
        # per "slot" – but because face positions change between frames we use
        # a single sliding window on the *best* similarity found each frame).
        self._recent_sims: deque = deque(
            maxlen=config.SIMILARITY_SMOOTHING_WINDOW
        )
        # Rolling count of consecutive frames where smoothed sim >= threshold.
        self._consecutive_match_count: int = 0

        # The *last* confirmed match state.  This is what we render on the UI.
        self._confirmed_match: bool = False

        # Frame counter for PROCESS_EVERY_N_FRAMES.
        self._frame_counter: int = 0

        # Cache the last annotated frame so skipped frames still show output.
        self._last_annotated_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Target management
    # ------------------------------------------------------------------

    def set_target(self, image_bgr: np.ndarray) -> Tuple[bool, str]:
        """
        Set the target person from a BGR image (as returned by cv2.imread).

        Returns
        -------
        (success: bool, message: str)
        """
        ready, err = model_ready()
        if not ready:
            return False, f"Model not loaded: {err}"

        if image_bgr is None or image_bgr.size == 0:
            return False, "Received an empty image."

        try:
            faces = _face_app.get(image_bgr)
        except Exception as exc:
            return False, f"Face detection error: {exc}"

        if len(faces) == 0:
            return False, "⚠️ No face detected in the image. Please try a clearer photo."

        if len(faces) > 1:
            return (
                False,
                f"⚠️ {len(faces)} faces detected. Please provide an image "
                "containing exactly one person.",
            )

        embedding = _normalize(faces[0].embedding)
        self._target_embedding = embedding
        self._target_set = True

        # Reset temporal smoothing on every new target.
        self._recent_sims.clear()
        self._consecutive_match_count = 0
        self._confirmed_match = False
        self._last_annotated_frame = None

        return True, "✅ Target set successfully!"

    def clear_target(self) -> None:
        """Remove the current target."""
        self._target_embedding = None
        self._target_set = False
        self._recent_sims.clear()
        self._consecutive_match_count = 0
        self._confirmed_match = False
        self._last_annotated_frame = None

    @property
    def has_target(self) -> bool:
        return self._target_set and self._target_embedding is not None

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect faces in *frame_bgr* and annotate it.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw BGR frame from OpenCV / Gradio webcam.

        Returns
        -------
        annotated_frame : np.ndarray   BGR frame with boxes drawn.
        results         : list[dict]   Per-face result dicts with keys:
                              bbox, label, similarity, is_target, is_match
        """
        ready, err = model_ready()
        if not ready:
            return frame_bgr, []

        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, []

        self._frame_counter += 1

        # --- Frame skipping -----------------------------------------------
        if self._frame_counter % config.PROCESS_EVERY_N_FRAMES != 0:
            if self._last_annotated_frame is not None:
                return self._last_annotated_frame, []
            return frame_bgr, []

        annotated = frame_bgr.copy()
        results: List[dict] = []

        try:
            faces = _face_app.get(frame_bgr)
        except Exception as exc:
            logger.warning("Face detection error on frame: %s", exc)
            return annotated, []

        # Best similarity found across all faces in this frame.
        best_sim_this_frame: float = -1.0
        best_face_idx: int = -1

        if self.has_target and len(faces) > 0:
            for idx, face in enumerate(faces):
                emb = _normalize(face.embedding)
                sim = cosine_similarity(self._target_embedding, emb)
                if sim > best_sim_this_frame:
                    best_sim_this_frame = sim
                    best_face_idx = idx

            # Temporal smoothing: push the best sim into the rolling window.
            self._recent_sims.append(best_sim_this_frame)
            smoothed_sim = float(np.mean(self._recent_sims))

            # Confirmation counter.
            if smoothed_sim >= config.SIMILARITY_THRESHOLD:
                self._consecutive_match_count += 1
            else:
                self._consecutive_match_count = 0

            self._confirmed_match = (
                self._consecutive_match_count >= config.MATCH_CONFIRMATION_FRAMES
            )

        # --- Annotate every detected face ----------------------------------
        h, w = annotated.shape[:2]

        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = (
                max(0, bbox[0]),
                max(0, bbox[1]),
                min(w - 1, bbox[2]),
                min(h - 1, bbox[3]),
            )

            if self.has_target:
                emb = _normalize(face.embedding)
                sim = cosine_similarity(self._target_embedding, emb)
                is_best = (idx == best_face_idx)
                is_target_match = is_best and self._confirmed_match
            else:
                sim = 0.0
                is_target_match = False

            if is_target_match:
                color = config.MATCH_BOX_COLOR
                label_top = "TARGET"
                label_bot = f"Match: {sim * 100:.1f}%"
                results.append(
                    dict(
                        bbox=(x1, y1, x2, y2),
                        label="TARGET",
                        similarity=sim,
                        is_target=True,
                        is_match=True,
                    )
                )
                _draw_box_with_label(
                    annotated, x1, y1, x2, y2,
                    label_top, label_bot, color,
                )
            else:
                if config.SHOW_UNKNOWN_FACES:
                    color = config.UNKNOWN_BOX_COLOR
                    if self.has_target:
                        label_top = "Unknown"
                        label_bot = f"Sim: {sim * 100:.1f}%"
                    else:
                        label_top = "Face"
                        label_bot = ""
                    results.append(
                        dict(
                            bbox=(x1, y1, x2, y2),
                            label="Unknown",
                            similarity=sim,
                            is_target=False,
                            is_match=False,
                        )
                    )
                    _draw_box_with_label(
                        annotated, x1, y1, x2, y2,
                        label_top, label_bot, color,
                    )

        self._last_annotated_frame = annotated
        return annotated, results

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def smoothed_similarity(self) -> Optional[float]:
        """Current smoothed similarity (mean of recent window), or None."""
        if not self._recent_sims:
            return None
        return float(np.mean(self._recent_sims))

    @property
    def confirmed_match(self) -> bool:
        return self._confirmed_match


# ---------------------------------------------------------------------------
# Drawing helpers (internal)
# ---------------------------------------------------------------------------

def _draw_box_with_label(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label_top: str,
    label_bot: str,
    color: Tuple[int, int, int],
) -> None:
    """Draw a coloured rectangle + one or two text labels."""
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = config.LABEL_FONT_SCALE
    text_th = config.LABEL_THICKNESS

    # Top label (above the box).
    if label_top:
        (tw, th), baseline = cv2.getTextSize(label_top, font, scale, text_th)
        ty = max(y1 - 8, th + 4)
        # Background rectangle for readability.
        cv2.rectangle(
            frame,
            (x1, ty - th - baseline - 2),
            (x1 + tw + 4, ty + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label_top, (x1 + 2, ty - baseline),
            font, scale, (255, 255, 255), text_th, cv2.LINE_AA,
        )

    # Bottom label (below the box).
    if label_bot:
        (tw2, th2), base2 = cv2.getTextSize(label_bot, font, scale * 0.85, text_th)
        by = min(y2 + th2 + 8, frame.shape[0] - 2)
        cv2.rectangle(
            frame,
            (x1, by - th2 - base2 - 2),
            (x1 + tw2 + 4, by + base2),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label_bot, (x1 + 2, by - base2),
            font, scale * 0.85, (255, 255, 255), text_th, cv2.LINE_AA,
        )
