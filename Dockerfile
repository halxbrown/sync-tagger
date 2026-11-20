# Use lightweight official Python image
FROM python:3.11-slim

WORKDIR /app

# Dependencies first: this layer stays cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model files, then code (code changes most often, so it goes last)
COPY models/ models/
COPY static/ static/
COPY tagger.py storage.py main.py ./

# Cloud Run provides a PORT environment variable; default to 8080 locally
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
