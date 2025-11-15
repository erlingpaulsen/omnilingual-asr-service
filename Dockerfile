FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps: Python, libsndfile, ffmpeg for audio
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA support and required Python packages
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch && \
    python3 -m pip install \
        omnilingual-asr \
        fastapi \
        "uvicorn[standard]" \
        jinja2 \
        soundfile \
        python-multipart

# Copy application code
COPY app ./app

EXPOSE 8000

# ASR_MODEL_CARD is configurable; default 1B model
ENV ASR_MODEL_CARD=omniASR_LLM_1B
ENV MAX_CHUNK_SECONDS=30.0
ENV MAX_FILE_SECONDS=3600.0

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
