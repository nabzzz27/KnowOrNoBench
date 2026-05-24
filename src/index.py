"""Persistent Chroma cosine collection: build, load, query.

The collection metadata sets `hnsw:space=cosine` explicitly — Chroma defaults to L2 and the
mismatch silently degrades retrieval (a known risk the spike flagged).

`build_index` is idempotent and crash-resumable: re-running skips ids already present in the
collection, so a 429 storm mid-build can be recovered by simply re-running.
"""

from __future__ import annotations

from typing import Callable

import chromadb
from chromadb.config import Settings

from src import config
from src.embed import embed_documents


# ---------------------------------------------------------------------------
# Chroma plumbing
# ---------------------------------------------------------------------------
def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(config.CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection() -> chromadb.Collection:
    """Get-or-create the SSOC collection with cosine space set explicitly."""
    return _client().get_or_create_collection(
        config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _flatten_meta(c: dict) -> dict:
    """Chroma metadata values must be scalars (str/int/float/bool); flatten our nested
    `meta` and merge source/code/title onto the top level so retrieval gets them back."""
    flat = {"source": c["source"], "code": c["code"], "title": c["title"]}
    for k, v in c.get("meta", {}).items():
        flat[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return flat


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------
def build_index(
    chunks: list[dict],
    rebuild: bool = False,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> chromadb.Collection:
    """Embed missing chunks and add them to the persistent Chroma collection.

    Args:
        chunks: list of chunk dicts (as produced by `src.ingest.ingest_all`).
        rebuild: if True, delete the existing collection first (fresh start).
        embed_fn: embedding function; defaults to `src.embed.embed_documents`.
                  Tests inject a deterministic stub to avoid API calls.

    Returns the collection. Re-running with the same `chunks` is a no-op (idempotent).
    """
    if rebuild:
        try:
            _client().delete_collection(config.COLLECTION_NAME)
        except Exception:  # noqa: BLE001 — fine if it didn't exist
            pass

    col = get_collection()
    embed = embed_fn or embed_documents

    existing = set(col.get(include=[])["ids"])
    missing = [c for c in chunks if c["id"] not in existing]
    total = len(chunks)

    if not missing:
        print(f"[index] collection already has {col.count()}/{total} chunks; nothing to do")
        return col

    print(f"[index] {len(missing)}/{total} chunks to embed "
          f"(existing in collection: {len(existing)})")

    batch_size = config.EMBED_BATCH
    n_batches = (len(missing) + batch_size - 1) // batch_size
    for b, i in enumerate(range(0, len(missing), batch_size), start=1):
        batch = missing[i:i + batch_size]
        vectors = embed([c["text"] for c in batch])
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[_flatten_meta(c) for c in batch],
        )
        print(f"[index] batch {b}/{n_batches}: +{len(batch)} chunks "
              f"(collection now {col.count()}/{total})")
    return col


def load_index() -> chromadb.Collection:
    """Return the persistent collection; raise a clear error if it's empty."""
    col = get_collection()
    if col.count() == 0:
        raise RuntimeError(
            f"Collection '{config.COLLECTION_NAME}' at {config.CHROMA_PATH} is empty. "
            f"Build it first:  python -m scripts.build_index")
    return col


def index_size() -> int:
    return get_collection().count()
