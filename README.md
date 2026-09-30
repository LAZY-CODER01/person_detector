# FaceTrack — Real-time Local Face Recognition

**100% private. Runs entirely on your Mac. No cloud, no internet required.**

FaceTrack lets you select any person as a "target" (via photo upload or webcam snapshot) and then tracks them in your live webcam feed using **InsightFace ArcFace** embeddings.

---

## What it does

1. You provide a reference photo of the person you want to find.
2. FaceTrack extracts an ArcFace face embedding from that photo.
3. It processes the live webcam stream frame-by-frame using the same model.
4. Detected faces are compared against the target embedding using cosine similarity.
5. The target person gets a **green bounding box** with a similarity score.
6. All other faces get a **blue box** labelled "Unknown".

---

## Project structure

```
Nag/
│
├── src/
│   ├── app.py            ← Main Gradio UI (entry point)
│   ├── face_engine.py    ← InsightFace logic (FaceRecognitionEngine class)
│   ├── config.py         ← All tunable parameters
│   └── utils.py          ← RGB/BGR conversion helpers
│
├── reference_images/     ← Sample reference images (not used at runtime)
├── output/               ← (unused by default, kept for your own saves)
├── requirements.txt
├── README.md
└── .venv/                ← Python virtual environment
```

---

## Installation

```bash
# 1. Clone / navigate to the project
cd /Users/avinash/Desktop/Nag

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** InsightFace will automatically download the `buffalo_l` model
> (~280 MB) on the first run.  Subsequent starts are instant.

---

## Running the application

```bash
cd /Users/avinash/Desktop/Nag
source .venv/bin/activate
python src/app.py
```

Then open your browser at:

```
http://127.0.0.1:7860
```

The browser tab opens automatically if your system allows it.

---

## How to use

### 1 — Select the target person

- On the **left panel**, either:
  - Click **"Upload"** to pick an image file, or
  - Click **"Webcam"** to take a snapshot with your camera.
- Click **"✅ Set as Target"**.

The image must contain **exactly one face**. If zero or multiple faces are detected you will see an error message.

### 2 — Start live recognition

- Click the **▶ (play)** button inside the *Camera input* box to begin streaming.
- The right panel shows the annotated live feed in real time.

### 3 — Read the output

| Colour | Meaning |
|--------|---------|
| 🟩 Green box | Target detected (`similarity ≥ threshold`) |
| 🟦 Blue box  | Unknown / non-target face |

The **Similarity** display shows the current smoothed score as a percentage.

---

## Similarity threshold

FaceTrack uses ArcFace **cosine similarity** in the range `[-1, 1]`.

| Similarity | Interpretation |
|------------|---------------|
| 0.7 – 1.0  | Strong match |
| 0.4 – 0.7  | Moderate match |
| 0.2 – 0.4  | Borderline |
| < 0.2      | Different people |

**Default threshold: 0.35** — deliberately conservative.  
You can change it in two ways:

### Option A — Live slider (no restart needed)

Open **⚙️ Settings** in the UI and drag the *Similarity Threshold* slider.

### Option B — Edit `config.py`

```python
# src/config.py
SIMILARITY_THRESHOLD = 0.35   # ← change this
```

Then restart the app.

---

## Performance settings

All performance settings are in **⚙️ Settings** in the UI (live) or in `src/config.py` (persistent):

| Setting | Default | Description |
|---------|---------|-------------|
| `PROCESS_EVERY_N_FRAMES` | `1` | Process every frame. Set to `2` or `3` to reduce CPU load. |
| `MATCH_CONFIRMATION_FRAMES` | `3` | How many consecutive matching frames before a confirmed detection. |
| `SIMILARITY_SMOOTHING_WINDOW` | `5` | Rolling average window to reduce flickering. |
| `INSIGHTFACE_DET_SIZE` | `(640, 640)` | Detection resolution. `(320, 320)` is faster on slow Macs. |

---

## How matching works (technical)

1. **Target embedding** — when you click *Set as Target*, the engine runs `FaceAnalysis.get()` on your image, extracts the ArcFace 512-D embedding, and L2-normalises it.
2. **Per-frame pipeline** — for every processed frame:
   - `FaceAnalysis.get(frame_bgr)` detects all faces.
   - Each face's embedding is extracted and normalised.
   - Cosine similarity = `dot(target_emb, face_emb)` (both normalised).
   - The face with the highest similarity that also exceeds the threshold is the candidate.
3. **Temporal smoothing** — similarities are averaged over a rolling window (`SIMILARITY_SMOOTHING_WINDOW` frames).  The "confirmed match" state only triggers after `MATCH_CONFIRMATION_FRAMES` consecutive above-threshold results, preventing flickering.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| App doesn't start | Missing dependency | `pip install -r requirements.txt` |
| Camera not found | macOS camera permissions | System Preferences → Privacy → Camera → allow Terminal |
| "No face detected" on target | Low-quality or small face | Use a clearer, front-facing photo |
| Target never matches | Threshold too high | Lower threshold to 0.25–0.30 in Settings |
| Too many false matches | Threshold too low | Raise threshold to 0.45–0.55 in Settings |
| High CPU usage | Processing every frame | Set *Process every N frames* to 2 or 3 in Settings |
| Jumpy match / unmatched | Smoothing window too small | Increase `SIMILARITY_SMOOTHING_WINDOW` in `config.py` |
| `buffalo_l` download fails | No internet on first run | Connect to internet once; model cached in `~/.insightface/` |

---

## Privacy

- **No data leaves your machine.** The camera feed is processed entirely in local Python.
- No cloud APIs, no telemetry, no persistent storage of captured faces.
- The target embedding is stored only in memory and discarded when you click *Clear* or close the app.

---

## macOS notes

- The app uses **CPUExecutionProvider** for maximum compatibility.
- Apple Silicon (M1/M2/M3) support via CoreML can be added later by switching to `CoreMLExecutionProvider` in `config.py` — the architecture is already open for it.
- First run is slower because InsightFace downloads and caches the model.
