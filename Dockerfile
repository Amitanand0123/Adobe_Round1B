# Use a slim Python base image with the specified platform
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')" || echo "Spacy model download skipped"

# Copy source code
COPY pdf_processor.py .
COPY hierarchy_builder.py .
COPY content_chunker.py .
COPY semantic_ranker.py .
COPY main.py .

# Copy input directory with all PDFs and JSON files
COPY input/ /app/input/

# Create output directory
RUN mkdir -p /app/output

# Set the command to run the application
CMD ["python", "main.py"]