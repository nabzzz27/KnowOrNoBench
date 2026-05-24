"""Central configuration for the SSOC RAG system.

Constants only — no logic. Every other module imports its paths, model names, and tuning
constants from here so there is a single place to change them. Values were validated by the
de-risking spike (see `spike_log.md`).
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

EXCEL_PATH = DATA_RAW / "ssoc2024-detailed-definitions.xlsx"
PDF_PATH = DATA_RAW / "ssoc2024report.pdf"
CHUNKS_PATH = DATA_PROCESSED / "ssoc_chunks.jsonl"        # canonical chunk artifact
CHUNKS_PREVIEW_PATH = DATA_PROCESSED / "chunks_preview.txt"  # human-readable dump
CHROMA_PATH = REPO_ROOT / "chroma_db"                    # persistent vector store

# --- Models -----------------------------------------------------------------
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"

# --- Retrieval / generation constants --------------------------------------
EMBED_DIM = 768                  # Matryoshka-truncated from 3072; validated in spike
TOP_K = 4                        # retrieved chunks per query (recall@5 was 100% in spike)
COLLECTION_NAME = "ssoc_2024"
GEN_TEMPERATURE = 0.0            # deterministic — the eval must be reproducible

# Asymmetric embedding task types (confusing the two silently degrades retrieval).
DOC_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"

# --- Ingest constants -------------------------------------------------------
EXCEL_HEADER_ROW = 4             # 0-indexed; column names sit on the 5th sheet row
EXCEL_CODE_LENGTHS = (4, 5)      # chunk 4-digit unit groups + 5-digit occupations
REPORT_PAGES = (8, 23)           # inclusive, human page numbers (1-indexed)
EMBED_TOKEN_LIMIT = 2048         # gemini-embedding-001 per-text input limit

# --- Embedding rate-limit pacing (discovered live in the spike) -------------
# Free-tier embedding quota is counted per text at ~100/min; pace below that and
# wait a full window on a 429 (short exponential backoff is not enough).
EMBED_TEXTS_PER_MIN = 95
RATE_LIMIT_WAIT = 60
EMBED_BATCH = 100                # texts per embedding request
