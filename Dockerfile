# Use a slim Python base image with the specified platform
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Poetry and Transformers cache
ENV POETRY_NO_INTERACTION=1
ENV HF_HOME=/app/models

# Install system dependencies required for pdf2image and opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies, including CPU-specific torch versions
RUN pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Download and cache the NLP models to be included in the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -m spacy download en_core_web_sm

# Copy application source code
COPY src/ ./src/

# Set the entry point for the container
CMD ["python", "-m", "src.main"]