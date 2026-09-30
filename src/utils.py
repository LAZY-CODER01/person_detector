"""
utils.py
========
Small image-conversion helpers used by both app.py and face_engine.py.
Centralising conversions here prevents the classic RGB/BGR swap bugs.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to a numpy BGR array for OpenCV / InsightFace."""
    rgb = np.array(pil_image.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()   # flip channel order; .copy() for contiguity
    return bgr


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR numpy array to RGB (for Gradio image display)."""
    return bgr[:, :, ::-1].copy()


def numpy_to_pil(rgb: np.ndarray) -> Image.Image:
    """Convert an RGB numpy array to a PIL Image."""
    return Image.fromarray(rgb.astype(np.uint8))
