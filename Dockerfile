# mini-gen-search/textgen-service-rag/Dockerfile
# Use the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Set environment variables to avoid interactive prompts
ENV PYTHONUNBUFFERED 1
# QDRANT_HOST/PORT are placeholders but useful for the wait script
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333

# Install system dependencies needed by Unstructured
# FIX: Add netcat for service health checks in the entrypoint
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libxml2-dev \
    libxslt1-dev \
    libreoffice \
    libgomp1 \
    poppler-utils \
    netcat-traditional \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements before installing to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- RAG Setup: Download Embedding Model Locally ---
# This speeds up container startup by preventing model download at runtime
RUN mkdir -p /app/embeddings_model
# Use Python to download the model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('/app/embeddings_model')"

# Copy the application code and the PDF
COPY app.py textgen_api.py rag_api.py ./
COPY assets/138.pdf /app/138.pdf

# Expose the application port
EXPOSE 5001

# FIX: Run a bash script to wait for Qdrant service before running the application
# It uses the QDRANT_HOST/PORT environment variables (set in k8s)
CMD ["/bin/bash", "-c", "echo 'Waiting for Qdrant service...' && while ! nc -z qdrant-service 6333; do sleep 1; done; echo 'Qdrant is up. Starting TextGen app.' && python app.py"]