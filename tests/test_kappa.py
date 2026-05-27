"""Tests for src.eval.kappa — Cohen's κ + bootstrap CI + confusion matrix."""

from __future__ import annotations

import pytest

from src.eval.kappa import (
    bootstrap_kappa_ci, cohen_kappa, compute_kappa_block, confusion_matrix,
)
from src.eval.judge_prompt import VALID_LABELS


def test_perfect_agreement_gives_kappa_one():
    a = ["CORRECT_ANSWER", "HALLUCINATION", "CORRECT_ABSTAIN"]
    b = list(a)
    assert cohen_kappa(a, b) == 1.0


def test_total_disagreement_gives_negative_kappa():
    # Two raters who disagree on every item should produce κ < 0.
    a = ["CORRECT_ANSWER"] * 5 + ["HALLUCINATION"] * 5
    b = ["HALLUCINATION"] * 5 + ["CORRECT_ANSWER"] * 5
    k = cohen_kappa(a, b)
    assert k < 0


def test_random_chance_agreement_near_zero():
    # Two raters drawing from balanced binary distributions independently.
    a = ["X", "Y"] * 50  # 50/50
    b = ["X", "X", "Y", "Y"] * 25  # 50/50 but different pattern
    k = cohen_kappa(a, b)
    # Half match by chance (50% agreement, 50% expected): κ ≈ 0
    assert abs(k) < 0.05


def test_known_textbook_example():
    """Cohen 1960 example: 2x2 table

        Yes No
    Yes  20 10
    No   15 15
    n=60, p_o = (20+15)/60 = 35/60 ≈ 0.5833
    p_e = (30/60)*(35/60) + (30/60)*(25/60) = 0.5
    κ = (0.5833 − 0.5) / (1 − 0.5) = 0.1667
    """
    a = (["Yes"] * 20 + ["Yes"] * 10
         + ["No"] * 15 + ["No"] * 15)
    b = (["Yes"] * 20 + ["No"] * 10
         + ["Yes"] * 15 + ["No"] * 15)
    k = cohen_kappa(a, b)
    assert abs(k - 0.1667) < 0.001


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        cohen_kappa([], [])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        cohen_kappa(["A"], ["A", "B"])


def test_single_label_universe_perfect_match():
    """Edge case: both annotators always pick the same label → κ defined as 1.0."""
    a = ["X"] * 10
    b = ["X"] * 10
    assert cohen_kappa(a, b) == 1.0


def test_confusion_matrix_counts_correctly():
    a = ["CORRECT_ANSWER", "CORRECT_ANSWER", "HALLUCINATION"]
    b = ["CORRECT_ANSWER", "WRONG_ANSWER", "HALLUCINATION"]
    cm = confusion_matrix(a, b, VALID_LABELS)
    assert cm["CORRECT_ANSWER"]["CORRECT_ANSWER"] == 1
    assert cm["CORRECT_ANSWER"]["WRONG_ANSWER"] == 1
    assert cm["HALLUCINATION"]["HALLUCINATION"] == 1
    # Other cells zero
    assert cm["WRONG_ANSWER"]["WRONG_ANSWER"] == 0
    # All 5 row labels present
    assert set(cm.keys()) == set(VALID_LABELS)


def test_confusion_matrix_rejects_unknown_label():
    with pytest.raises(ValueError, match="label outside universe"):
        confusion_matrix(["FOO"], ["BAR"], VALID_LABELS)


def test_bootstrap_ci_bounds_kappa_for_perfect_agreement():
    a = ["CORRECT_ANSWER"] * 5 + ["HALLUCINATION"] * 5
    b = list(a)
    lo, hi = bootstrap_kappa_ci(a, b, n_resamples=200, seed=42)
    # Some bootstrap samples may have only one label → kappa undefined; but most should
    # have κ near 1. The high bound should be >= 0.8 at least.
    assert hi >= 0.8
    assert lo <= 1.0


def test_bootstrap_ci_is_deterministic_under_seed():
    a = ["CORRECT_ANSWER"] * 5 + ["HALLUCINATION"] * 5
    b = ["CORRECT_ANSWER"] * 4 + ["HALLUCINATION"] * 6
    first = bootstrap_kappa_ci(a, b, n_resamples=100, seed=42)
    second = bootstrap_kappa_ci(a, b, n_resamples=100, seed=42)
    assert first == second


def test_compute_kappa_block_returns_full_shape():
    a = ["CORRECT_ANSWER", "HALLUCINATION", "CORRECT_ABSTAIN",
         "CORRECT_ANSWER", "HALLUCINATION"]
    b = ["CORRECT_ANSWER", "HALLUCINATION", "CORRECT_ABSTAIN",
         "WRONG_ANSWER", "HALLUCINATION"]
    result = compute_kappa_block(a, b, VALID_LABELS, n_resamples=100)
    assert set(result.keys()) == {"kappa", "ci_low", "ci_high", "n", "confusion_matrix"}
    assert result["n"] == 5
    assert isinstance(result["confusion_matrix"], dict)
    assert 0 <= result["kappa"] <= 1
