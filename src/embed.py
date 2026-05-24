"""Paced wrapper around gemini-embedding-001 (validated by the de-risking spike).

Free-tier embedding quota is counted per *text* at ~100/min (the spike measured this live),
so this module:
  - Paces calls to <= EMBED_TEXTS_PER_MIN per rolling 60s window (proactive throttle).
  - Waits a full RATE_LIMIT_WAIT (~60s) on a 429 (short exponential backoff is not enough
    against a per-minute quota window).
  - Falls back to smaller batches if a full batch errors for non-rate-limit reasons.

Two functions enforce the asymmetric task types at the call site:
  - embed_documents(texts) -> RETRIEVAL_DOCUMENT, batched
  - embed_query(text)      -> RETRIEVAL_QUERY, single

Confusing the two silently degrades retrieval, so the function names ARE the API.
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src import config

load_dotenv(config.REPO_ROOT / ".env")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazy module-level client; raises clearly if the key is missing."""
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env at the repo root.")
        _client = genai.Client(api_key=key)
    return _client


# ---------------------------------------------------------------------------
# Pacer + retry  (lifted from the spike, now production)
# ---------------------------------------------------------------------------
class _EmbedPacer:
    """Keep embedding calls under EMBED_TEXTS_PER_MIN within a rolling 60s window.

    Throttles *before* hitting a 429 (cheaper than the recovery wait). The 429-aware retry
    in `_with_retry` is the backstop if pacing under-counts.
    """

    def __init__(self):
        self.events: list[tuple[float, int]] = []  # (timestamp, n_texts)

    def reset(self):
        self.events = []

    def reserve(self, n: int) -> None:
        now = time.time()
        self.events = [(t, c) for t, c in self.events if now - t < 60]
        used = sum(c for _, c in self.events)
        if self.events and used + n > config.EMBED_TEXTS_PER_MIN:
            sleep = 60 - (now - self.events[0][0]) + 1
            if sleep > 0:
                print(f"[embed pace] {used}+{n} > {config.EMBED_TEXTS_PER_MIN}/min; "
                      f"sleeping {sleep:.0f}s")
                time.sleep(sleep)
            self.reset()
        self.events.append((time.time(), n))


_pacer = _EmbedPacer()


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "resource_exhausted" in s or ("rate" in s and "limit" in s)


def _with_retry(fn, what: str, max_tries: int = 5):
    """Retry transient errors. On 429 wait the full per-minute window (RATE_LIMIT_WAIT);
    on other transient errors use short exponential backoff."""
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if attempt == max_tries:
                raise
            if _is_rate_limit(e):
                wait = config.RATE_LIMIT_WAIT
                _pacer.reset()
                print(f"[embed retry] {what} attempt {attempt} hit 429; waiting {wait}s")
            else:
                wait = 2 ** attempt
                print(f"[embed retry] {what} attempt {attempt} failed "
                      f"({type(e).__name__}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("_with_retry: exhausted retries without returning")  # unreachable


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed document texts at 768-dim, RETRIEVAL_DOCUMENT, paced + retried.

    Batches in `config.EMBED_BATCH` per request and aligns output to input order. Falls back
    to batch=25 then batch=1 if a full batch errors for non-rate-limit reasons (the
    rate-limit case is handled by `_with_retry`).
    """
    if not texts:
        return []
    client = _get_client()
    cfg = types.EmbedContentConfig(
        task_type=config.DOC_TASK, output_dimensionality=config.EMBED_DIM)

    vectors: list[list[float]] = []
    batch_size = config.EMBED_BATCH
    i = 0
    while i < len(texts):
        chunk = texts[i:i + batch_size]
        _pacer.reserve(len(chunk))
        try:
            res = _with_retry(
                lambda: client.models.embed_content(
                    model=config.EMBED_MODEL, contents=chunk, config=cfg),
                f"embed_documents[{i}:{i+len(chunk)}]",
            )
            vectors.extend(e.values for e in res.embeddings)
            i += batch_size
        except Exception as e:  # noqa: BLE001
            if batch_size > 1:
                new = 25 if batch_size > 25 else 1
                print(f"[embed fallback] batch={batch_size} failed ({type(e).__name__}); "
                      f"falling back to batch={new}")
                batch_size = new
                continue
            raise
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string at 768-dim, RETRIEVAL_QUERY, paced + retried."""
    client = _get_client()
    cfg = types.EmbedContentConfig(
        task_type=config.QUERY_TASK, output_dimensionality=config.EMBED_DIM)
    _pacer.reserve(1)
    res = _with_retry(
        lambda: client.models.embed_content(
            model=config.EMBED_MODEL, contents=text, config=cfg),
        "embed_query",
    )
    return res.embeddings[0].values
