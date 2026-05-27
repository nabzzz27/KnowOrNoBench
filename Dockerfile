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

# Pre-computed artifacts that are portable across machines (the notebook reads
# from results/; build_index reads the chunks JSONL from data/processed/)
COPY data/ ./data/
COPY results/ ./results/

# chroma_db is NOT copied here. The HNSW segment format is not portable across
# machines, so the index is built locally on first run via:
#     docker compose run --rm rag python -m scripts.build_index
# The rebuilt index lands in /app/chroma_db, which is volume-mounted to
# ./chroma_db on the host (see docker-compose.yml) so it persists.

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
