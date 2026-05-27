"""Tests for src.rag.generate (no API calls — stubs the LLM call)."""

from __future__ import annotations

import time

import pytest

from src.rag import generate as gen_mod


def test_generate_returns_gen_fn_output():
    out = gen_mod.generate("hello", gen_fn=lambda p: f"echo: {p}")
    assert out == "echo: hello"


def test_generate_retries_on_rate_limit(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    calls = {"n": 0}

    def flaky(p):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")
        return "ok"

    out = gen_mod.generate("p", gen_fn=flaky)
    assert out == "ok"
    assert calls["n"] == 2
    assert slept and slept[0] >= 30  # rate-limit wait, not short backoff


def test_generate_retries_with_short_backoff_on_other_errors(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    calls = {"n": 0}

    def flaky(p):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("ConnectionError: read timeout")
        return "ok"

    out = gen_mod.generate("p", gen_fn=flaky)
    assert out == "ok"
    assert slept and slept[0] < 30  # short backoff, not the rate-limit wait


def test_generate_raises_after_max_attempts(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def always_fails(p):
        raise RuntimeError("HTTP 429")

    with pytest.raises(RuntimeError):
        gen_mod.generate("p", gen_fn=always_fails)


def test_is_rate_limit_detects_429_variants():
    assert gen_mod._is_rate_limit(RuntimeError("HTTP 429"))
    assert gen_mod._is_rate_limit(RuntimeError("RESOURCE_EXHAUSTED"))
    assert gen_mod._is_rate_limit(RuntimeError("rate limit exceeded"))
    assert not gen_mod._is_rate_limit(RuntimeError("read timeout"))
