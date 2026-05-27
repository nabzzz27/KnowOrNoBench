"""Tests for src.rag.retrieve (no API calls — stubs embed_fn and the Chroma collection)."""

from __future__ import annotations

import pytest

from src import config
from src.rag import retrieve as retrieve_mod


class _StubCollection:
    """Mimics chromadb.Collection.query() return shape."""

    def __init__(self, ids, distances, documents, metadatas):
        self._ids = ids
        self._distances = distances
        self._documents = documents
        self._metadatas = metadatas
        self.calls: list[dict] = []

    def query(self, query_embeddings, n_results, include):
        self.calls.append({
            "n_results": n_results,
            "include": include,
            "query_dim": len(query_embeddings[0]),
        })
        return {
            "ids": [self._ids],
            "distances": [self._distances],
            "documents": [self._documents],
            "metadatas": [self._metadatas],
        }


def _stub_embed_fn(text: str):
    return [0.0] * config.EMBED_DIM


def _populated_col():
    return _StubCollection(
        ids=["25121", "25122", "report-2.5", "31301"],
        distances=[0.21, 0.24, 0.31, 0.40],
        documents=["software dev text", "sysadmin text", "report 2.5 text", "tech text"],
        metadatas=[
            {"source": "ssoc_excel", "code": "25121", "title": "Software Developer", "level": 5},
            {"source": "ssoc_excel", "code": "25122", "title": "Systems Administrator", "level": 5},
            {"source": "ssoc_report", "code": "report-2.5", "title": "Hierarchy",
             "page": 9, "para": "2.5", "section": "Structure"},
            {"source": "ssoc_excel", "code": "31301", "title": "Computer Technician", "level": 5},
        ],
    )


def test_retrieve_returns_top_k_chunks_in_contract_shape():
    col = _populated_col()
    chunks = retrieve_mod.retrieve("What is SSOC 25121?", embed_fn=_stub_embed_fn, col=col)
    assert len(chunks) == 4
    keys = {"id", "text", "distance", "metadata"}
    for c in chunks:
        assert set(c.keys()) == keys
    assert chunks[0]["id"] == "25121"
    assert chunks[0]["text"] == "software dev text"
    assert chunks[0]["distance"] == pytest.approx(0.21)
    assert chunks[0]["metadata"]["title"] == "Software Developer"


def test_retrieve_passes_top_k_and_query_dim_to_collection():
    col = _populated_col()
    retrieve_mod.retrieve("q", top_k=2, embed_fn=_stub_embed_fn, col=col)
    assert col.calls[0]["n_results"] == 2
    assert col.calls[0]["query_dim"] == config.EMBED_DIM
    assert set(col.calls[0]["include"]) == {"documents", "metadatas", "distances"}


def test_retrieve_defaults_to_config_top_k():
    col = _populated_col()
    retrieve_mod.retrieve("q", embed_fn=_stub_embed_fn, col=col)
    assert col.calls[0]["n_results"] == config.TOP_K


def test_retrieve_returns_empty_on_empty_collection():
    col = _StubCollection(ids=[], distances=[], documents=[], metadatas=[])
    chunks = retrieve_mod.retrieve("q", embed_fn=_stub_embed_fn, col=col)
    assert chunks == []


def test_retrieve_returns_empty_on_blank_question():
    col = _populated_col()
    chunks = retrieve_mod.retrieve("   ", embed_fn=_stub_embed_fn, col=col)
    assert chunks == []
