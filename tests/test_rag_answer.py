"""Tests for src.rag.answer orchestrator (no API calls — stubs retrieve + generate)."""

from __future__ import annotations

import pytest

from src.rag.answer import answer


def _fake_chunks():
    return [
        {"id": "25121", "text": "software dev text", "distance": 0.21,
         "metadata": {"source": "ssoc_excel", "code": "25121",
                      "title": "Software Developer", "level": 5}},
    ]


def _retrieve_stub(question):
    return _fake_chunks()


def _generate_stub(prompt):
    return f"LLM_OUTPUT[{len(prompt)}]"


def test_answer_returns_full_contract_dict():
    out = answer(
        "What is SSOC 25121?",
        config="strict",
        retrieve_fn=_retrieve_stub,
        generate_fn=_generate_stub,
    )
    assert set(out.keys()) == {"question", "config", "retrieved_chunks", "prompt", "response"}
    assert out["question"] == "What is SSOC 25121?"
    assert out["config"] == "strict"
    assert out["retrieved_chunks"] == _fake_chunks()
    assert "25121" in out["prompt"]
    assert "What is SSOC 25121?" in out["prompt"]
    assert out["response"] == _generate_stub(out["prompt"])


def test_answer_neutral_uses_retrieval_and_neutral_prompt():
    out = answer(
        "q", config="neutral", retrieve_fn=_retrieve_stub, generate_fn=_generate_stub
    )
    assert len(out["retrieved_chunks"]) == 1
    # neutral has no MUST / no I-don't-know rule
    assert "MUST" not in out["prompt"]
    assert "I don't know" not in out["prompt"]


def test_answer_forced_uses_forced_prompt():
    out = answer(
        "q", config="forced", retrieve_fn=_retrieve_stub, generate_fn=_generate_stub
    )
    assert "MUST" in out["prompt"]
    assert "I don't know" in out["prompt"]  # appears as forbidden language
    assert len(out["retrieved_chunks"]) == 1


def test_answer_strict_uses_strict_prompt():
    out = answer(
        "q", config="strict", retrieve_fn=_retrieve_stub, generate_fn=_generate_stub
    )
    assert "I don't know" in out["prompt"]
    assert "MUST" not in out["prompt"]  # distinguishes strict from forced
    assert len(out["retrieved_chunks"]) == 1


def test_answer_rejects_unknown_config():
    with pytest.raises(ValueError, match="config"):
        answer("q", config="bogus",
               retrieve_fn=_retrieve_stub, generate_fn=_generate_stub)


def test_answer_default_config_is_strict():
    out = answer(
        "q", retrieve_fn=_retrieve_stub, generate_fn=_generate_stub
    )
    assert out["config"] == "strict"
    assert "I don't know" in out["prompt"]
    assert "MUST" not in out["prompt"]
