"""Unit tests for src.eval.metrics — pure-function tests on synthetic DataFrames."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eval.metrics import (
    confusion_2x2, correct_abstention_rate, correct_answer_rate,
    hallucination_rate, headline_comparison, over_refusal_rate,
    per_category_rates, retrieval_cascade_rate, sample_failures,
    wrong_answer_rate,
)


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_hallucination_rate():
    df = _df([
        # 4 unanswerable; 3 hallucinated
        {"answerable": False, "label": "HALLUCINATION",  "category": "fab", "retrieval_provided_answer": True},
        {"answerable": False, "label": "HALLUCINATION",  "category": "fab", "retrieval_provided_answer": False},
        {"answerable": False, "label": "HALLUCINATION",  "category": "fab", "retrieval_provided_answer": False},
        {"answerable": False, "label": "CORRECT_ABSTAIN","category": "fab", "retrieval_provided_answer": True},
        # 2 answerable (shouldn't affect rate)
        {"answerable": True,  "label": "CORRECT_ANSWER", "category": "answerable", "retrieval_provided_answer": True},
        {"answerable": True,  "label": "CORRECT_ANSWER", "category": "answerable", "retrieval_provided_answer": True},
    ])
    assert hallucination_rate(df) == 0.75  # 3 / 4


def test_hallucination_rate_zero_unanswerable_returns_zero():
    df = _df([{"answerable": True, "label": "CORRECT_ANSWER", "category": "answerable", "retrieval_provided_answer": True}])
    assert hallucination_rate(df) == 0.0


def test_correct_abstention_rate():
    df = _df([
        {"answerable": False, "label": "CORRECT_ABSTAIN", "category": "fab", "retrieval_provided_answer": True},
        {"answerable": False, "label": "CORRECT_ABSTAIN", "category": "fab", "retrieval_provided_answer": True},
        {"answerable": False, "label": "HALLUCINATION",   "category": "fab", "retrieval_provided_answer": False},
    ])
    assert correct_abstention_rate(df) == 0.6667


def test_over_refusal_rate():
    df = _df([
        {"answerable": True, "label": "OVER_REFUSAL",  "category": "answerable", "retrieval_provided_answer": True},
        {"answerable": True, "label": "CORRECT_ANSWER","category": "answerable", "retrieval_provided_answer": True},
        {"answerable": True, "label": "CORRECT_ANSWER","category": "answerable", "retrieval_provided_answer": True},
    ])
    assert over_refusal_rate(df) == 0.3333


def test_correct_answer_rate():
    df = _df([
        {"answerable": True, "label": "CORRECT_ANSWER", "category": "answerable", "retrieval_provided_answer": True},
        {"answerable": True, "label": "WRONG_ANSWER",   "category": "answerable", "retrieval_provided_answer": True},
    ])
    assert correct_answer_rate(df) == 0.5


def test_wrong_answer_rate():
    df = _df([
        {"answerable": True, "label": "WRONG_ANSWER",   "category": "answerable", "retrieval_provided_answer": True},
        {"answerable": True, "label": "CORRECT_ANSWER", "category": "answerable", "retrieval_provided_answer": True},
        {"answerable": True, "label": "CORRECT_ANSWER", "category": "answerable", "retrieval_provided_answer": True},
        {"answerable": True, "label": "CORRECT_ANSWER", "category": "answerable", "retrieval_provided_answer": True},
    ])
    assert wrong_answer_rate(df) == 0.25


def test_confusion_2x2_structure_and_counts():
    df = _df([
        # answerable answered (correct or wrong) = 2
        {"answerable": True,  "label": "CORRECT_ANSWER", "category": "answerable", "retrieval_provided_answer": True},
        {"answerable": True,  "label": "WRONG_ANSWER",   "category": "answerable", "retrieval_provided_answer": True},
        # answerable abstained = 1
        {"answerable": True,  "label": "OVER_REFUSAL",   "category": "answerable", "retrieval_provided_answer": False},
        # unanswerable answered (= hallucinated) = 3
        {"answerable": False, "label": "HALLUCINATION",  "category": "fab", "retrieval_provided_answer": False},
        {"answerable": False, "label": "HALLUCINATION",  "category": "fab", "retrieval_provided_answer": False},
        {"answerable": False, "label": "HALLUCINATION",  "category": "fab", "retrieval_provided_answer": False},
        # unanswerable abstained = 2
        {"answerable": False, "label": "CORRECT_ABSTAIN","category": "fab", "retrieval_provided_answer": True},
        {"answerable": False, "label": "CORRECT_ABSTAIN","category": "fab", "retrieval_provided_answer": True},
    ])
    cm = confusion_2x2(df)
    assert cm == {
        "answerable":   {"answered": 2, "abstained": 1},
        "unanswerable": {"answered": 3, "abstained": 2},
    }


def test_per_category_rates_shape_and_values():
    df = _df([
        # answerable category: 2 correct, 1 wrong
        {"answerable": True,  "label": "CORRECT_ANSWER",  "category": "answerable", "retrieval_provided_answer": True},
        {"answerable": True,  "label": "CORRECT_ANSWER",  "category": "answerable", "retrieval_provided_answer": True},
        {"answerable": True,  "label": "WRONG_ANSWER",    "category": "answerable", "retrieval_provided_answer": True},
        # ssic_confusion: 1 hallu, 1 abstain
        {"answerable": False, "label": "HALLUCINATION",   "category": "ssic_confusion", "retrieval_provided_answer": False},
        {"answerable": False, "label": "CORRECT_ABSTAIN", "category": "ssic_confusion", "retrieval_provided_answer": True},
    ])
    out = per_category_rates(df)
    assert set(out["category"]) == {"answerable", "ssic_confusion"}

    ans_row = out[out["category"] == "answerable"].iloc[0]
    assert ans_row["n"] == 3
    assert ans_row["correct_answer_rate"] == 0.6667
    assert ans_row["wrong_answer_rate"] == 0.3333

    ssic_row = out[out["category"] == "ssic_confusion"].iloc[0]
    assert ssic_row["n"] == 2
    assert ssic_row["hallucination_rate"] == 0.5
    assert ssic_row["correct_abstention_rate"] == 0.5


def test_retrieval_cascade_rate():
    df = _df([
        # 4 hallucinations: 1 with correct retrieval, 3 without
        {"answerable": False, "label": "HALLUCINATION", "category": "fab", "retrieval_provided_answer": True},
        {"answerable": False, "label": "HALLUCINATION", "category": "fab", "retrieval_provided_answer": False},
        {"answerable": False, "label": "HALLUCINATION", "category": "fab", "retrieval_provided_answer": False},
        {"answerable": False, "label": "HALLUCINATION", "category": "fab", "retrieval_provided_answer": False},
        # 1 correct abstain (ignored by cascade calc)
        {"answerable": False, "label": "CORRECT_ABSTAIN", "category": "fab", "retrieval_provided_answer": True},
    ])
    result = retrieval_cascade_rate(df)
    assert result["n_hallucinations"] == 4
    assert result["with_correct_retrieval"] == 1
    assert result["without_correct_retrieval"] == 3
    assert result["retrieval_cascade_rate"] == 0.75


def test_retrieval_cascade_rate_no_hallucinations():
    df = _df([
        {"answerable": False, "label": "CORRECT_ABSTAIN", "category": "fab", "retrieval_provided_answer": True},
    ])
    result = retrieval_cascade_rate(df)
    assert result["n_hallucinations"] == 0
    assert result["retrieval_cascade_rate"] == 0.0


def test_headline_comparison_shape():
    base = lambda label, ans=True: {"answerable": ans, "label": label,
                                    "category": "answerable" if ans else "fab",
                                    "retrieval_provided_answer": True}
    df_neutral = _df([base("CORRECT_ANSWER")] * 4
                     + [base("HALLUCINATION", ans=False)] * 2
                     + [base("CORRECT_ABSTAIN", ans=False)] * 2)
    df_forced = _df([base("CORRECT_ANSWER")] * 4
                    + [base("HALLUCINATION", ans=False)] * 4)
    df_strict = _df([base("CORRECT_ANSWER")] * 4
                    + [base("CORRECT_ABSTAIN", ans=False)] * 4)

    out = headline_comparison({"neutral": df_neutral, "forced": df_forced, "strict": df_strict})
    assert list(out.columns) == ["neutral", "forced", "strict"]
    # 5 metric rows
    assert set(out.index) == {"hallucination_rate", "correct_abstention_rate",
                              "over_refusal_rate", "correct_answer_rate",
                              "wrong_answer_rate"}
    assert out.loc["hallucination_rate", "strict"] == 0.0
    assert out.loc["hallucination_rate", "forced"] == 1.0
    assert out.loc["hallucination_rate", "neutral"] == 0.5


def test_sample_failures_returns_n_and_columns():
    df = _df([
        {"answerable": False, "label": "HALLUCINATION", "category": "fab", "benchmark_id": "fab-001",
         "question": "Q1?", "response": "R1", "judge_rationale": "rationale1",
         "retrieval_provided_answer": True},
        {"answerable": False, "label": "HALLUCINATION", "category": "fab", "benchmark_id": "fab-002",
         "question": "Q2?", "response": "R2", "judge_rationale": "rationale2",
         "retrieval_provided_answer": True},
        {"answerable": False, "label": "CORRECT_ABSTAIN", "category": "fab", "benchmark_id": "fab-003",
         "question": "Q3?", "response": "R3", "judge_rationale": "rationale3",
         "retrieval_provided_answer": True},
    ])
    out = sample_failures(df, "HALLUCINATION", n=2)
    assert len(out) == 2
    assert set(out["benchmark_id"]) == {"fab-001", "fab-002"}
    assert "judge_rationale" in out.columns


def test_recall_at_k(tmp_path):
    """Build a small responses.xlsx and check recall@4."""
    rows = [
        # answerable, GT 25121 in retrieved_context → hit
        {"benchmark_id": "ans-001", "answerable": True, "ground_truth_code": "25121",
         "ground_truth_title": "Software dev", "category": "answerable",
         "retrieved_context": "[SSOC 25121] Software developer\nDef..."},
        # answerable, GT 22200 NOT in context → miss
        {"benchmark_id": "ans-002", "answerable": True, "ground_truth_code": "22200",
         "ground_truth_title": "Nurse", "category": "answerable",
         "retrieved_context": "[SSOC 99999] Some other text"},
        # unanswerable (ignored)
        {"benchmark_id": "fab-001", "answerable": False, "ground_truth_code": "",
         "ground_truth_title": "", "category": "fab",
         "retrieved_context": "anything"},
    ]
    df = pd.DataFrame(rows)
    p = tmp_path / "responses.xlsx"
    df.to_excel(p, index=False)

    from src.eval.metrics import recall_at_k
    result = recall_at_k(p, k=4)
    assert result["n_answerable"] == 2
    assert result["hits"] == 1
    assert result["recall_at_k"] == 0.5


def test_bool_coercion_for_answerable_strings():
    """Excel sometimes writes True/False as strings; metrics must handle both."""
    df = pd.DataFrame([
        {"answerable": "True",  "label": "CORRECT_ANSWER", "category": "answerable",
         "retrieval_provided_answer": True},
        {"answerable": "False", "label": "HALLUCINATION",  "category": "fab",
         "retrieval_provided_answer": False},
    ])
    assert correct_answer_rate(df) == 1.0
    assert hallucination_rate(df) == 1.0
