FROM python:3.14-slim

WORKDIR /app

# System deps for pdfplumber, openpyxl, and pip wheel builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps first so the layer is cached across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY benchmark/ ./benchmark/
COPY notebooks/ ./notebooks/

# Pre-built artifacts so the reviewer does not need an API key for the demo path
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/
COPY results/ ./results/

# Bring .env.example along so the reviewer can see the expected shape
COPY .env.example ./

EXPOSE 8888

# Default: launch Jupyter so the reviewer can open the results notebook at
# http://localhost:8888 with no token required
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=", \
     "--ServerApp.password="]
