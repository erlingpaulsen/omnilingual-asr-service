import os
import tempfile
import traceback
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Request,
    HTTPException,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

# ---------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------

MAX_CHUNK_SECONDS = float(os.getenv("MAX_CHUNK_SECONDS", 30.0))  # model limit per chunk
MAX_FILE_SECONDS = float(os.getenv("MAX_FILE_SECONDS", 3600.0))  # safety: 1 hour per file to avoid OOM

DEFAULT_LANG = "nob_Latn" if "nob_Latn" in supported_langs else "eng_Latn"
LANG_CHOICES = sorted(supported_langs)

MODEL_CARD = os.getenv("ASR_MODEL_CARD", "omniASR_LLM_1B")

app = FastAPI(title="Omnilingual ASR Service", version="0.2.0")

# Load templates and static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global pipeline instance
PIPELINE: Optional[ASRInferencePipeline] = None


# ---------------------------------------------------------------------
# Startup: load the model once
# ---------------------------------------------------------------------
@app.on_event("startup")
def load_model() -> None:
    global PIPELINE
    PIPELINE = ASRInferencePipeline(model_card=MODEL_CARD)


# ---------------------------------------------------------------------
# Helpers: audio loading + chunking + transcription
# ---------------------------------------------------------------------
def load_and_chunk_audio(
    path: str,
    max_chunk_seconds: float = MAX_CHUNK_SECONDS,
    max_file_seconds: float = MAX_FILE_SECONDS,
) -> Tuple[List[dict], float]:
    """
    Load audio file, convert to mono, and chunk into <= max_chunk_seconds.

    Returns:
        audio_inputs: list of {"waveform": np.ndarray, "sample_rate": int}
        total_duration: float seconds
    """
    data, sr = sf.read(path)
    if data.ndim > 1:
        # Convert multi-channel to mono
        data = np.mean(data, axis=1)

    if sr <= 0 or data.size == 0:
        raise ValueError("Invalid or empty audio file")

    total_secs = float(len(data)) / float(sr)
    if total_secs > max_file_seconds:
        raise ValueError(
            f"Audio is too long: {total_secs:.1f}s. "
            f"Max supported is {max_file_seconds:.0f}s."
        )

    chunk_size = int(max_chunk_seconds * sr)
    audio_inputs: List[dict] = []

    for start in range(0, len(data), chunk_size):
        chunk = data[start:start + chunk_size]
        audio_inputs.append(
            {
                "waveform": chunk,
                "sample_rate": sr,
            }
        )

    return audio_inputs, total_secs


def run_transcription_pipeline(file_path: str, lang_code: str) -> str:
    """
    Synchronous helper: chunk audio then call the ASR pipeline.
    Returns concatenated transcription for all chunks.
    """
    if PIPELINE is None:
        raise RuntimeError("ASR pipeline not initialized")

    audio_inputs, total_secs = load_and_chunk_audio(file_path)

    # One language code per chunk
    lang_list = [lang_code] * len(audio_inputs)

    # Pipeline supports waveform dicts as input
    texts: List[str] = PIPELINE.transcribe(
        audio_inputs,
        lang=lang_list,
        batch_size=1,
    )

    if not texts:
        raise RuntimeError("Pipeline returned no transcription")

    # Join chunks with blank lines so you can see boundaries
    combined = "\n\n".join(texts)
    return combined


# ---------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------
class TranscriptionResponse(BaseModel):
    filename: str
    language: str
    text: str


# ---------------------------------------------------------------------
# JSON API endpoint
# ---------------------------------------------------------------------
@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_api(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
):
    # Persist upload to a temp file (needed for libsndfile/soundfile)
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)

    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        # Run heavy work in a threadpool
        text = await run_in_threadpool(run_transcription_pipeline, tmp_path, lang)

        return TranscriptionResponse(
            filename=file.filename,
            language=lang,
            text=text,
        )

    except Exception:
        error_text = traceback.format_exc()
        # Expose full traceback in the API error for debugging
        raise HTTPException(status_code=500, detail=error_text)
    finally:
        # Always delete uploaded temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------
# Web UI routes
# ---------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "languages": LANG_CHOICES,
            "selected_lang": DEFAULT_LANG,
            "result_text": None,
            "is_error": False,
            "filename": None,
        },
    )


@app.post("/ui/transcribe", response_class=HTMLResponse)
async def transcribe_ui(
    request: Request,
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)

    result_text: Optional[str] = None
    is_error = False

    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        text = await run_in_threadpool(run_transcription_pipeline, tmp_path, lang)
        result_text = text
        is_error = False

    except Exception:
        # Capture traceback so UI can show "error log" as requested
        result_text = traceback.format_exc()
        is_error = True

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "languages": LANG_CHOICES,
            "selected_lang": lang,
            "result_text": result_text,
            "is_error": is_error,
            "filename": file.filename,
        },
    )
