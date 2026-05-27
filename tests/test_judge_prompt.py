"""Tests for src.eval.judge_prompt — structural invariants on the frozen template."""

from __future__ import annotations

from src.eval.judge_prompt import (
    JUDGE_PROMPT_V0, VALID_LABELS, format_judge_prompt,
)


def test_template_has_all_five_placeholders():
    for placeholder in ("{question}", "{expected_behavior}", "{ground_truth}",
                        "{retrieved_context}", "{response}"):
        assert placeholder in JUDGE_PROMPT_V0


def test_valid_labels_are_exactly_the_five_prd_labels():
    assert set(VALID_LABELS) == {
        "CORRECT_ANSWER", "WRONG_ANSWER", "OVER_REFUSAL",
        "CORRECT_ABSTAIN", "HALLUCINATION",
    }
    assert len(VALID_LABELS) == 5


def test_template_documents_all_five_labels():
    """The label rubric must mention each label so the LLM has the full menu."""
    for label in VALID_LABELS:
        assert label in JUDGE_PROMPT_V0


def test_template_mentions_hedged_claims_rule():
    """The 'hedged claims count' rule is load-bearing per PRD — must be in the prompt."""
    assert "HEDGED" in JUDGE_PROMPT_V0 or "hedg" in JUDGE_PROMPT_V0.lower()


def test_format_substitutes_all_five_fields():
    out = format_judge_prompt(
        question="What is SSOC 25121?",
        expected_behavior="answer",
        ground_truth="25121 Software developer",
        retrieved_context="[SSOC 25121] Software developer\nDefinition: ...",
        response="The code is 25121.",
    )
    assert "What is SSOC 25121?" in out
    assert "answer" in out
    assert "25121 Software developer" in out
    assert "[SSOC 25121]" in out
    assert "The code is 25121." in out


def test_format_renders_null_ground_truth_when_unanswerable():
    out = format_judge_prompt(
        question="What's the SSOC code for a unicorn trainer?",
        expected_behavior="abstain",
        ground_truth=None,
        retrieved_context="(some context)",
        response="I don't know.",
    )
    # Look for the literal 'null' string near the GROUND_TRUTH section
    assert "GROUND_TRUTH: null" in out


def test_format_handles_braces_in_input_text():
    """User-facing fields can contain literal {} — `.format()` only interprets template
    braces, not substituted values. Regression test against a Python footgun."""
    out = format_judge_prompt(
        question="What about {curly} characters?",
        expected_behavior="answer",
        ground_truth="x",
        retrieved_context="text with {literal} braces",
        response="response with {more} braces",
    )
    assert "{curly}" in out
    assert "{literal}" in out
    assert "{more}" in out