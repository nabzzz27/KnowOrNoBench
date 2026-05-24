"""Tests for the Chroma index builder (no API calls — uses a stub embedder).

CHROMA_PATH is monkeypatched to a tmp dir so tests don't touch the real `chroma_db/`. The
collection name is also overridden so a botched test never collides with the real index.
"""

from __future__ import annotations

import pytest

from src import config, index


def _stub_embedder():
    """Deterministic fake embedder: returns a 768-dim vector keyed off the text length so
    each text gets a distinguishable vector (good enough for ingest-side tests)."""
    def _embed(texts):
        return [[float((len(t) + i) % 7)] * config.EMBED_DIM for i, t in enumerate(texts)]
    return _embed


def _sample_chunks():
    return [
        {"id": "alpha",  "source": "ssoc_excel", "code": "alpha",
         "title": "Alpha", "text": "alpha definition", "meta": {"level": 4}},
        {"id": "beta",   "source": "ssoc_excel", "code": "beta",
         "title": "Beta",  "text": "beta definition",  "meta": {"level": 5}},
        {"id": "report-1.1", "source": "ssoc_report", "code": "report-1.1",
         "title": "Intro", "text": "intro text",
         "meta": {"page": 8, "para": "1.1", "section": "Introduction"}},
    ]


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    """Point Chroma at a tmp dir and use a test-only collection name."""
    monkeypatch.setattr(config, "CHROMA_PATH", tmp_path / "chroma")
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_ssoc_2024")
    yield


def test_build_index_populates_collection(isolated):
    chunks = _sample_chunks()
    col = index.build_index(chunks, embed_fn=_stub_embedder())
    assert col.count() == len(chunks)
    got = col.get(ids=["alpha"], include=["metadatas", "documents"])
    assert got["documents"][0] == "alpha definition"
    md = got["metadatas"][0]
    assert md["source"] == "ssoc_excel"
    assert md["code"] == "alpha"
    assert md["title"] == "Alpha"
    assert md["level"] == 4


def test_cosine_space_is_set(isolated):
    index.build_index(_sample_chunks(), embed_fn=_stub_embedder())
    col = index.get_collection()
    assert col.metadata.get("hnsw:space") == "cosine"


def test_meta_flattens_non_scalars(isolated):
    chunks = [{"id": "x", "source": "s", "code": "x", "title": "X", "text": "t",
               "meta": {"page": 8, "tags": ["a", "b"]}}]  # tags is a list -> str-coerced
    col = index.build_index(chunks, embed_fn=_stub_embedder())
    md = col.get(ids=["x"], include=["metadatas"])["metadatas"][0]
    assert md["page"] == 8
    assert isinstance(md["tags"], str)


def test_idempotent_resumable(isolated):
    """Re-running build_index with the same chunks is a no-op."""
    chunks = _sample_chunks()
    embed = _stub_embedder()
    index.build_index(chunks, embed_fn=embed)
    n_before = index.index_size()
    # second call should add nothing
    index.build_index(chunks, embed_fn=embed)
    assert index.index_size() == n_before


def test_partial_resume_fills_missing(isolated):
    """If only some ids exist, the next call embeds only the missing ones."""
    chunks = _sample_chunks()
    index.build_index(chunks[:1], embed_fn=_stub_embedder())
    assert index.index_size() == 1
    # adding the rest fills in the missing ones
    index.build_index(chunks, embed_fn=_stub_embedder())
    assert index.index_size() == len(chunks)


def test_rebuild_wipes(isolated):
    """--rebuild deletes the existing collection before adding."""
    a = _sample_chunks()
    index.build_index(a, embed_fn=_stub_embedder())
    assert index.index_size() == 3
    b = [{"id": "only", "source": "ssoc_excel", "code": "only",
          "title": "Only", "text": "only text", "meta": {"level": 5}}]
    index.build_index(b, rebuild=True, embed_fn=_stub_embedder())
    assert index.index_size() == 1


def test_load_index_raises_when_empty(isolated):
    with pytest.raises(RuntimeError, match="empty"):
        index.load_index()


def test_load_index_returns_populated(isolated):
    index.build_index(_sample_chunks(), embed_fn=_stub_embedder())
    col = index.load_index()
    assert col.count() == 3
