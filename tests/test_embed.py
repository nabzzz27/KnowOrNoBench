"""Unit tests for the embedding rate-limit pacer (no API calls)."""

from __future__ import annotations

import time

from src import config, embed


def test_pacer_no_sleep_when_under_limit(monkeypatch):
    p = embed._EmbedPacer()
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    p.reserve(50)
    p.reserve(40)
    assert slept == []  # 50 + 40 = 90 <= 95/min


def test_pacer_sleeps_when_window_full(monkeypatch):
    p = embed._EmbedPacer()
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    p.reserve(config.EMBED_TEXTS_PER_MIN)  # fills the per-minute window
    p.reserve(10)                          # would exceed -> must sleep
    assert len(slept) == 1
    assert slept[0] > 0


def test_pacer_window_rolls_off(monkeypatch):
    """A reservation older than 60s no longer counts toward the rolling window."""
    p = embed._EmbedPacer()
    p.events = [(time.time() - 90, 100)]   # simulate an event 90s ago
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    p.reserve(50)
    assert slept == []


def test_is_rate_limit_detects_429():
    class FakeErr(Exception): pass
    assert embed._is_rate_limit(FakeErr("HTTP 429 RESOURCE_EXHAUSTED quota exceeded"))
    assert embed._is_rate_limit(FakeErr("rate limit exceeded"))
    assert not embed._is_rate_limit(FakeErr("timeout reading body"))
