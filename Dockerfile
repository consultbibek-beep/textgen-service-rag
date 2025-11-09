# mini-gen-search/textgen-service-rag/Dockerfile
# Use the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Set environment variables to avoid interactive prompts
ENV PYTHONUNBUFFERED 1
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333

# Install system dependencies needed by Unstructured
# Including libgomp1 for openblas/numpy/scipy dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libxml2-dev \
    libxslt1-dev \
    libreoffice \
    libgomp1 \
    poppler-utils \
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
# NOTE: Ensure '138.pdf' is present in the textgen-service-rag directory or parent directory
COPY app.py textgen_api.py rag_api.py ./
# CRITICAL: Copy the PDF document (assuming it is located at the project root or in the textgen-service-rag dir)
# If it's in the project root: COPY ../138.pdf /app/138.pdf
# Assuming it's available for copying:

COPY assets/138.pdf /app/138.pdf

# Expose the application port
EXPOSE 5001

# Run the consolidated application
CMD ["python", "app.py"]