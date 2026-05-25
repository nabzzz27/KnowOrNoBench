"""Tests for src.rag.prompts. These pin the SHAPE of the frozen prompts.

We deliberately do not assert full-text equality of prompts — that would make every
trivial copy-edit a test failure. We assert the structural invariants the eval relies on.
"""

from __future__ import annotations

from src.rag import prompts


def test_neutral_prompt_has_context_and_question_placeholders():
    assert "{context}" in prompts.NEUTRAL_PROMPT
    assert "{question}" in prompts.NEUTRAL_PROMPT


def test_forced_prompt_forbids_refusal():
    assert "{context}" in prompts.FORCED_PROMPT
    assert "{question}" in prompts.FORCED_PROMPT
    # the whole point: forced must explicitly forbid refusal
    assert "MUST" in prompts.FORCED_PROMPT
    assert "I don't know" in prompts.FORCED_PROMPT  # appears as forbidden language


def test_strict_prompt_has_idk_rule():
    assert "{context}" in prompts.STRICT_PROMPT
    assert "{question}" in prompts.STRICT_PROMPT
    assert "I don't know" in prompts.STRICT_PROMPT


def test_three_prompts_are_distinct():
    """The contrast surface is the whole point — no two prompts may be identical."""
    s = {prompts.NEUTRAL_PROMPT, prompts.FORCED_PROMPT, prompts.STRICT_PROMPT}
    assert len(s) == 3


def test_format_context_renders_chunks_with_code_title_text():
    chunks = [
        {"id": "25121", "text": "software dev text", "distance": 0.21,
         "metadata": {"source": "ssoc_excel", "code": "25121",
                      "title": "Software Developer", "level": 5}},
        {"id": "25122", "text": "sysadmin text", "distance": 0.24,
         "metadata": {"source": "ssoc_excel", "code": "25122",
                      "title": "Systems Administrator", "level": 5}},
    ]
    ctx = prompts.format_context(chunks)
    assert "25121" in ctx
    assert "Software Developer" in ctx
    assert "software dev text" in ctx
    assert "25122" in ctx
    assert "Systems Administrator" in ctx
    assert "sysadmin text" in ctx


def test_format_context_returns_empty_string_on_no_chunks():
    assert prompts.format_context([]) == ""


def test_neutral_prompt_renders_end_to_end():
    chunks = [{"id": "25121", "text": "text", "distance": 0.2,
               "metadata": {"code": "25121", "title": "Software Developer"}}]
    rendered = prompts.NEUTRAL_PROMPT.format(
        context=prompts.format_context(chunks), question="What is 25121?"
    )
    assert "25121" in rendered
    assert "What is 25121?" in rendered
