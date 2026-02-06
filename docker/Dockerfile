# CoVision Server for RunPod
# GPU-accelerated vision inference server

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy application
COPY src/ ./src/
COPY server/ ./server/
COPY web/ ./web/
COPY configs/ ./configs/
COPY pyproject.toml .

# Install covision package
RUN pip install -e .

# Create models directory
RUN mkdir -p models/embeddings

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python", "-m", "server.app"]
