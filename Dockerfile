# Dockerfile - Lightweight Version

# Use a slim Python image as the base
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for pdf2image and pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your source code into the container
COPY . .

# Create the output directory
RUN mkdir -p /app/output

# Set the command to run the application when the container starts
CMD ["python", "main.py"]