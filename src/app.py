"""
app.py
======
Gradio web interface for the real-time face-recognition application.

Architecture notes
------------------
* face_engine.py is imported once; InsightFace loads at that point.
* The FaceRecognitionEngine instance (``engine``) is a module-level singleton
  so it persists across Gradio callbacks without reloading the model.
* Gradio's gr.Image(sources=["webcam"]) component with streaming=True
  calls the process_webcam_frame callback for every frame the browser
  sends.  The callback must return a numpy RGB array (or PIL image) that
  Gradio then displays in a second gr.Image output component.
* Target selection (upload OR webcam snapshot) routes through set_target_image().
"""

from __future__ import annotations

import logging
import sys
import os

# ── Make sure src/ imports work regardless of cwd ─────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import gradio as gr

import config
from face_engine import FaceRecognitionEngine, model_ready
from utils import pil_to_bgr, bgr_to_rgb

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton engine  (InsightFace is loaded when face_engine is imported above)
# ---------------------------------------------------------------------------
engine = FaceRecognitionEngine()

# ---------------------------------------------------------------------------
# Startup check
# ---------------------------------------------------------------------------
_model_ok, _model_err = model_ready()
if not _model_ok:
    logger.error("InsightFace failed to load: %s", _model_err)
else:
    logger.info("InsightFace ready.  App starting …")


# ============================================================================
# Callback functions
# ============================================================================

def set_target_image(input_image):
    """
    Called when the user clicks 'Set as Target'.

    Parameters
    ----------
    input_image : np.ndarray | PIL.Image | None
        Whatever Gradio passes from the Image component (numpy RGB array).

    Returns
    -------
    status_text : str   Message shown in the status box.
    preview     : np.ndarray | None   The image to show in the preview box.
    """
    if input_image is None:
        return "⚠️  No image provided. Please upload or capture a photo.", None

    # Gradio Image components return numpy arrays in RGB.
    # InsightFace / OpenCV needs BGR.
    if isinstance(input_image, np.ndarray):
        bgr = input_image[:, :, ::-1].copy()
    else:
        # PIL fallback
        bgr = pil_to_bgr(input_image)

    success, message = engine.set_target(bgr)

    if success:
        # Return the RGB image for preview
        preview = input_image if isinstance(input_image, np.ndarray) else np.array(input_image)
        return message, preview
    else:
        return message, None


def clear_target():
    """Clear the current target person."""
    engine.clear_target()
    return "🗑️  Target cleared.", None


def process_webcam_frame(frame):
    """
    Streaming callback: called for every webcam frame Gradio receives.

    Parameters
    ----------
    frame : np.ndarray | None
        RGB numpy array from the webcam Image component.
        May be None if the camera isn't streaming yet.

    Returns
    -------
    annotated_rgb : np.ndarray
        Annotated frame in RGB (Gradio displays this in the output Image).
    status_text : str
        Human-readable status for the status Textbox.
    similarity_text : str
        Current smoothed similarity for the similarity display.
    """
    if frame is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _draw_centered_text(blank, "Waiting for camera …")
        return blank, _get_engine_status(), "—"

    # Gradio sends RGB; face_engine needs BGR.
    bgr = frame[:, :, ::-1].copy()

    annotated_bgr, results = engine.process_frame(bgr)

    # Convert back to RGB for Gradio display.
    annotated_rgb = annotated_bgr[:, :, ::-1].copy()

    return annotated_rgb, _get_engine_status(), _get_similarity_text()


def update_threshold(value: float):
    """Slider callback: update the similarity threshold live."""
    config.SIMILARITY_THRESHOLD = float(value)
    engine._recent_sims.clear()          # reset smoothing on threshold change
    engine._consecutive_match_count = 0
    engine._confirmed_match = False
    return f"Threshold set to {value:.2f}"


def update_frame_skip(value: int):
    """Slider callback: update frame-skip live."""
    config.PROCESS_EVERY_N_FRAMES = int(value)
    return f"Process every {int(value)} frame(s)"


def update_confirmation_frames(value: int):
    """Slider callback: update confirmation frames live."""
    config.MATCH_CONFIRMATION_FRAMES = int(value)
    engine._consecutive_match_count = 0
    engine._confirmed_match = False
    return f"Confirmation frames: {int(value)}"


def toggle_unknown_faces(show: bool):
    """Checkbox callback: show/hide Unknown face boxes."""
    config.SHOW_UNKNOWN_FACES = show
    return f"Unknown faces: {'shown' if show else 'hidden'}"


# ============================================================================
# Internal helpers
# ============================================================================

def _get_engine_status() -> str:
    if not _model_ok:
        return f"❌ Model error: {_model_err}"
    if not engine.has_target:
        return "ℹ️  No target set. Upload or capture a photo on the left."
    sim = engine.smoothed_similarity
    if sim is None:
        return "🎯 Target set. Waiting for camera frames …"
    if engine.confirmed_match:
        return f"🟢 TARGET DETECTED  (sim = {sim:.3f})"
    return f"🔍 Scanning …  (best sim = {sim:.3f}, threshold = {config.SIMILARITY_THRESHOLD:.2f})"


def _get_similarity_text() -> str:
    sim = engine.smoothed_similarity
    if sim is None:
        return "—"
    pct = sim * 100
    match_str = "✅ MATCH" if engine.confirmed_match else "❌ no match"
    return f"{pct:.1f}%  ({match_str})"


def _draw_centered_text(frame: np.ndarray, text: str) -> None:
    """Draw dark overlay text on a blank frame."""
    import cv2
    h, w = frame.shape[:2]
    cv2.putText(
        frame, text,
        (w // 2 - len(text) * 6, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA,
    )


# ============================================================================
# Gradio UI
# ============================================================================

CSS = """
/* ── Global ──────────────────────────────────────────────────────────────── */
:root {
    --bg-primary:   #0f1117;
    --bg-card:      #1a1d27;
    --bg-card-alt:  #1e2132;
    --accent:       #6c63ff;
    --accent-green: #00e676;
    --accent-red:   #ff5252;
    --text-main:    #e8e8f0;
    --text-muted:   #7a7a9a;
    --border:       #2e3150;
    --radius:       14px;
}

body, .gradio-container {
    background: var(--bg-primary) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Cards ───────────────────────────────────────────────────────────────── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

/* ── Section titles ──────────────────────────────────────────────────────── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
}

/* ── Status box ─────────────────────────────────────────────────────────── */
.status-box textarea, .status-box input {
    background: #111320 !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}

/* ── Similarity display ──────────────────────────────────────────────────── */
.sim-display textarea, .sim-display input {
    background: #111320 !important;
    color: var(--accent-green) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 1.1rem !important;
    font-weight: 700;
    text-align: center;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
button.primary-btn {
    background: linear-gradient(135deg, #6c63ff, #9c63ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    transition: transform 0.15s, box-shadow 0.15s;
}
button.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(108,99,255,0.45);
}

button.danger-btn {
    background: linear-gradient(135deg, #ff5252, #ff1744) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
}

/* ── Sliders ─────────────────────────────────────────────────────────────── */
.slider-label {
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-bottom: 4px;
}

/* ── Image containers ────────────────────────────────────────────────────── */
.image-preview img, .live-feed img {
    border-radius: 10px;
    border: 1px solid var(--border);
}

/* ── Settings accordion ──────────────────────────────────────────────────── */
.settings-row {
    background: var(--bg-card-alt);
    border: 1px solid var(--border);
    border-radius: var(--radius);
}

/* ── Gradient header ─────────────────────────────────────────────────────── */
#app-header {
    text-align: center;
    padding: 28px 0 16px;
}
#app-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6c63ff 0%, #00e5ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
#app-subtitle {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-top: 6px;
}

/* ── Misc ────────────────────────────────────────────────────────────────── */
label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
}
"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="FaceTrack – Real-time Face Recognition",
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="app-header">
            <h1 id="app-title">⚡ FaceTrack</h1>
            <p id="app-subtitle">
                Real-time local face recognition &nbsp;·&nbsp;
                InsightFace ArcFace &nbsp;·&nbsp; 100% private
            </p>
        </div>
        """)

        # ── Main two-column layout ─────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── LEFT: Target Person ───────────────────────────────────────
            with gr.Column(scale=1, min_width=340, elem_classes="card"):
                gr.HTML('<div class="section-title">🎯 Target Person</div>')

                target_input = gr.Image(
                    label="Upload photo  OR  capture with webcam",
                    sources=["upload", "webcam"],
                    type="numpy",
                    height=260,
                    elem_classes="image-preview",
                )

                with gr.Row():
                    set_btn = gr.Button(
                        "✅  Set as Target",
                        variant="primary",
                        elem_classes="primary-btn",
                    )
                    clear_btn = gr.Button(
                        "🗑️  Clear",
                        variant="secondary",
                        elem_classes="danger-btn",
                    )

                target_status = gr.Textbox(
                    label="Target Status",
                    value="ℹ️  No target set yet.",
                    interactive=False,
                    max_lines=2,
                    elem_classes="status-box",
                )

                gr.HTML('<div class="section-title" style="margin-top:14px;">📸 Target Preview</div>')
                target_preview = gr.Image(
                    label="",
                    type="numpy",
                    height=180,
                    interactive=False,
                    elem_classes="image-preview",
                )

            # ── RIGHT: Live Camera ────────────────────────────────────────
            with gr.Column(scale=2, min_width=520, elem_classes="card"):
                gr.HTML('<div class="section-title">📷 Live Camera</div>')

                with gr.Row():
                    with gr.Column(scale=1):
                        # Input stream from the browser webcam
                        webcam_input = gr.Image(
                            label="Camera input  (click ▶ to start)",
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            height=320,
                        )
                    with gr.Column(scale=1):
                        # Output: annotated frames from the engine
                        live_output = gr.Image(
                            label="Recognition output",
                            type="numpy",
                            height=320,
                            interactive=False,
                            elem_classes="live-feed",
                        )

                with gr.Row():
                    engine_status = gr.Textbox(
                        label="Engine Status",
                        value=_get_engine_status(),
                        interactive=False,
                        max_lines=2,
                        elem_classes="status-box",
                        scale=3,
                    )
                    similarity_display = gr.Textbox(
                        label="Similarity",
                        value="—",
                        interactive=False,
                        max_lines=1,
                        elem_classes="sim-display",
                        scale=1,
                    )

        # ── Settings accordion ────────────────────────────────────────────
        with gr.Accordion("⚙️  Settings  (click to expand)", open=False):
            gr.Markdown(
                "> Changes take effect immediately — no restart needed."
            )
            with gr.Row(elem_classes="settings-row"):
                with gr.Column():
                    threshold_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.90,
                        step=0.01,
                        value=config.SIMILARITY_THRESHOLD,
                        label="Similarity Threshold  (higher = stricter)",
                        info="ArcFace cosine similarity must exceed this to call it a match.",
                    )
                    threshold_status = gr.Textbox(
                        label="",
                        value=f"Threshold: {config.SIMILARITY_THRESHOLD:.2f}",
                        interactive=False,
                        max_lines=1,
                    )

                with gr.Column():
                    frame_skip_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=config.PROCESS_EVERY_N_FRAMES,
                        label="Process every N frames  (1 = every frame)",
                        info="Increase to reduce CPU usage at the cost of responsiveness.",
                    )
                    frame_skip_status = gr.Textbox(
                        label="",
                        value=f"Process every {config.PROCESS_EVERY_N_FRAMES} frame(s)",
                        interactive=False,
                        max_lines=1,
                    )

                with gr.Column():
                    confirm_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=config.MATCH_CONFIRMATION_FRAMES,
                        label="Confirmation Frames  (temporal smoothing)",
                        info="Target must match this many consecutive frames before being confirmed.",
                    )
                    confirm_status = gr.Textbox(
                        label="",
                        value=f"Confirmation frames: {config.MATCH_CONFIRMATION_FRAMES}",
                        interactive=False,
                        max_lines=1,
                    )

            with gr.Row():
                show_unknown_checkbox = gr.Checkbox(
                    label="Show Unknown faces (draw blue boxes on non-target faces)",
                    value=config.SHOW_UNKNOWN_FACES,
                )
                unknown_status = gr.Textbox(
                    label="",
                    value=f"Unknown faces: {'shown' if config.SHOW_UNKNOWN_FACES else 'hidden'}",
                    interactive=False,
                    max_lines=1,
                )

        # ── How it works ──────────────────────────────────────────────────
        with gr.Accordion("ℹ️  How it works", open=False):
            gr.Markdown("""
### Quick guide

1. **Set a target** — upload a photo or take a webcam snapshot on the **left panel**, then click *Set as Target*.  
   The image must contain **exactly one face**.

2. **Start the camera** — click the ▶ button inside the *Camera input* box to begin streaming.  
   The right panel shows the annotated live feed.

3. **Interpret the output**
   | Colour | Meaning |
   |--------|---------|
   | 🟩 Green box | Target person detected |
   | 🟦 Blue box  | Unknown / non-target face |

4. **Adjust the threshold** in the ⚙️ Settings accordion:
   - **Too many false positives?** → raise the threshold (e.g. 0.45–0.55).
   - **Target not detected?** → lower the threshold (e.g. 0.25–0.35).

5. **Performance** — if the CPU is overloaded, increase *Process every N frames* to 2 or 3.

### Similarity score reference (ArcFace)
| Range | Interpretation |
|-------|---------------|
| 0.7 – 1.0 | Strong match (very likely same person) |
| 0.4 – 0.7 | Moderate match (probably same person) |
| 0.2 – 0.4 | Weak / borderline |
| < 0.2  | Different people |

> These ranges depend on photo quality, angle, and lighting.  
> The default threshold of **0.35** is conservative and safe to start with.
""")

        # ── Wiring callbacks ──────────────────────────────────────────────

        set_btn.click(
            fn=set_target_image,
            inputs=[target_input],
            outputs=[target_status, target_preview],
        )

        clear_btn.click(
            fn=clear_target,
            inputs=[],
            outputs=[target_status, target_preview],
        )

        webcam_input.stream(
            fn=process_webcam_frame,
            inputs=[webcam_input],
            outputs=[live_output, engine_status, similarity_display],
            time_limit=None,
        )

        threshold_slider.change(
            fn=update_threshold,
            inputs=[threshold_slider],
            outputs=[threshold_status],
        )

        frame_skip_slider.change(
            fn=update_frame_skip,
            inputs=[frame_skip_slider],
            outputs=[frame_skip_status],
        )

        confirm_slider.change(
            fn=update_confirmation_frames,
            inputs=[confirm_slider],
            outputs=[confirm_status],
        )

        show_unknown_checkbox.change(
            fn=toggle_unknown_faces,
            inputs=[show_unknown_checkbox],
            outputs=[unknown_status],
        )

    return demo


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import gradio

    gr_version = tuple(int(x) for x in gradio.__version__.split(".")[:2])
    logger.info("Gradio version: %s", gradio.__version__)

    demo = build_ui()

    demo.launch(
        server_name=config.GRADIO_SERVER_NAME,
        server_port=config.GRADIO_SERVER_PORT,
        show_error=True,
        inbrowser=True,          # opens the browser tab automatically
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css=CSS,
    )
