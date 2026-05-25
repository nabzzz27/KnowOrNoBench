"""Retrieve top-k chunks from the persistent Chroma collection.

Wraps `embed_query` + `collection.query` and returns chunk dicts in the contract shape
the eval expects. `embed_fn` and `col` are injectable so tests can stub them.
"""

from __future__ import annotations

from typing import Callable

from src import config


def retrieve(
    question: str,
    top_k: int | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
    col=None,
) -> list[dict]:
    """Embed `question` and return the top-k chunk dicts ordered by rank.

    Returns [] for blank questions or empty collections — the caller (the eval) should
    handle the no-context case in its prompt template, not have to defend against an
    exception here.
    """
    if not question or not question.strip():
        return []

    if embed_fn is None:
        from src.embed import embed_query
        embed_fn = embed_query
    if col is None:
        from src.index import load_index
        col = load_index()

    k = top_k if top_k is not None else config.TOP_K
    query_vec = embed_fn(question)
    res = col.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ids = res["ids"][0]
    if not ids:
        return []

    return [
        {
            "id": ids[i],
            "text": res["documents"][0][i],
            "distance": float(res["distances"][0][i]),
            "metadata": dict(res["metadatas"][0][i]),
        }
        for i in range(len(ids))
    ]
