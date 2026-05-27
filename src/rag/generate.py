"""Gemini 2.5 Flash generation with tiny inline retry.

Temperature is fixed at config.GEN_TEMPERATURE (0.0) so the eval is deterministic.
The rate-limit detector is duplicated from src.embed on purpose — the spec explicitly
calls for no shared retry helper; 3 lines of duplication beats a premature abstraction.
"""

from __future__ import annotations

import os
import time
from typing import Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src import config

load_dotenv(config.REPO_ROOT / ".env")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env at the repo root.")
        _client = genai.Client(api_key=key)
    return _client


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "resource_exhausted" in s or ("rate" in s and "limit" in s)


def _default_gen_fn(prompt: str) -> str:
    client = _get_client()
    res = client.models.generate_content(
        model=config.GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=config.GEN_TEMPERATURE),
    )
    return res.text


def generate(
    prompt: str,
    gen_fn: Callable[[str], str] | None = None,
    max_tries: int = 3,
) -> str:
    """Call Gemini Flash (or `gen_fn` stub) with retry. Returns the response text."""
    fn = gen_fn or _default_gen_fn
    last_exc: Exception | None = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn(prompt)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt == max_tries:
                break
            wait = config.RATE_LIMIT_WAIT if _is_rate_limit(e) else 2 ** attempt
            print(f"[gen retry] attempt {attempt} failed ({type(e).__name__}); "
                  f"sleeping {wait}s")
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc
