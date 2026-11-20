"""Sync-licensing auto-tagger API.

POST an audio file to /predict and get back genre / mood / instrument tags,
BPM and key as JSON. Every result is saved to "the crate" (Firestore on Cloud
Run, a local JSON file otherwise), where the user's edits are kept too.
Interactive docs at /docs; the upload page is at /.
"""
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from storage import make_store
from tagger import Tagger

MAX_MB = 30   # Cloud Run caps HTTP request bodies at 32 MB; stay safely under it
ALLOWED = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
HEADS = ("genre", "mood_theme", "instrument")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the models and connect to storage ONCE when the container starts
    state["tagger"] = Tagger("models")
    state["store"] = make_store()
    yield
    state.clear()


app = FastAPI(
    title="Sync Tagger",
    description="Auto-tags music for sync licensing submissions: genre, mood/theme, "
                "instrumentation, BPM and key, from an uploaded audio file. "
                "Results and edits are kept in a crate of past analyses.",
    version="1.1.0",
    lifespan=lifespan,
)


# ---------- schemas ----------
class Tag(BaseModel):
    tag: str
    score: float


class NearMiss(Tag):
    of_threshold: float = Field(description="Score as a fraction of the tag's threshold")


class Suggestions(BaseModel):
    genre: List[NearMiss]
    mood_theme: List[NearMiss]
    instrument: List[NearMiss]


class Vocals(BaseModel):
    voice: float = Field(description="Probability the track has vocals (sung or rapped)")
    instrumental: bool


class TempoReading(BaseModel):
    bpm: float
    label: str


class Prediction(BaseModel):
    id: str
    status: str
    filename: str
    genre: List[Tag]
    mood_theme: List[Tag]
    instrument: List[Tag]
    suggestions: Suggestions = Field(description="Tags that scored just under their threshold")
    vocals: Optional[Vocals]
    windows: List[List[float]] = Field(description="Seconds of audio the model listened to")
    bpm: float
    bpm_alt: float
    bpm_alts: List[TempoReading] = Field(description="Half/double-time and 3:2 readings of the same pulse")
    bpm_confidence: float
    key: str
    key_relative: str
    camelot: str
    key_strength: float
    duration_sec: float
    processing_sec: float


class Edits(BaseModel):
    removed: List[str] = Field(default=[], max_length=300, description='Model tags the user turned off, e.g. "genre:pop"')
    added: List[str] = Field(default=[], max_length=50, description="The user's own tags")
    bpm: Optional[float] = Field(default=None, gt=0, lt=400, description="The tempo the user chose")
    vocals: Optional[bool] = Field(default=None, description="The user's correction: true = has vocals")


def trim(result: dict, top_k: int) -> dict:
    if top_k:
        result = {**result, **{h: result[h][:top_k] for h in HEADS}}
    return result


# ---------- pages ----------
@app.get("/", include_in_schema=False)
def home():
    """The upload page (static/index.html)."""
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    """Health check: confirms the service is running and the models are loaded."""
    return {"status": "online", "models_loaded": "tagger" in state,
            "crate_store": state["store"].kind if "store" in state else None}


# ---------- analysis ----------
@app.post("/predict", response_model=Prediction)
def predict(
    file: UploadFile = File(..., description="Audio file (wav, mp3, flac, ogg, m4a, aiff)"),
    top_k: int = Query(5, ge=0, le=60, description="Max tags per category (0 = all above threshold)"),
):
    """Analyze one track, save it to the crate, and return its tags, BPM and key."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Use one of: {sorted(ALLOWED)}")

    data = file.file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large ({len(data) / 1e6:.1f} MB). Max is {MAX_MB} MB.")

    # Essentia reads from a path, so write the upload to a temp file
    # (keeping the extension so the decoder recognizes the format)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name

    t0 = time.time()
    try:
        result = state["tagger"].predict(path)
    except ValueError as e:                      # e.g. audio too short
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")
    finally:
        os.remove(path)

    result = {"status": "success", "filename": file.filename,
              "processing_sec": round(time.time() - t0, 2), **result}
    # the crate keeps every tag above threshold; the response is trimmed to top_k
    result["id"] = state["store"].add({"created_at": time.time(), "filename": file.filename,
                                       "result": result, "edits": {}})
    return trim(result, top_k)


# ---------- the crate ----------
@app.get("/tracks")
def list_tracks(limit: int = Query(60, ge=1, le=200)):
    """Past analyses, newest first."""
    return state["store"].list(limit)


@app.get("/tracks/{track_id}")
def get_track(track_id: str):
    """One past analysis with the user's edits."""
    rec = state["store"].get(track_id)
    if not rec:
        raise HTTPException(404, "No track with that id in the crate.")
    return rec


@app.patch("/tracks/{track_id}")
def save_edits(track_id: str, edits: Edits):
    """Save the user's tag edits and tempo choice for a track."""
    if not state["store"].update_edits(track_id, edits.model_dump()):
        raise HTTPException(404, "No track with that id in the crate.")
    return {"status": "saved"}


@app.delete("/tracks/{track_id}")
def delete_track(track_id: str):
    """Remove a track from the crate."""
    if not state["store"].delete(track_id):
        raise HTTPException(404, "No track with that id in the crate.")
    return {"status": "deleted"}
