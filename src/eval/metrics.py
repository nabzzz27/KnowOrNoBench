"""Pure functions that aggregate judged DataFrames into the headline + breakdown metrics.

All functions take pandas DataFrames matching the schema in `src/eval/judge.py`
(`JUDGED_COLUMNS`) and return scalars or DataFrames. No side effects, no Excel I/O —
the notebook is the only place that touches files. This makes the metric definitions
unit-testable in isolation.

Label vocabulary (per `src/eval/judge_prompt.py:VALID_LABELS`):
  - CORRECT_ANSWER / WRONG_ANSWER / OVER_REFUSAL (on answerable questions)
  - CORRECT_ABSTAIN / HALLUCINATION (on unanswerable questions)
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_answerable(df: pd.DataFrame) -> pd.Series:
    """`answerable` column may be bool, "True"/"False" string, or numpy bool."""
    col = df["answerable"]
    if col.dtype == bool:
        return col
    return col.astype(str).str.strip().str.lower().isin({"true", "1"})


def _safe_rate(numer: int, denom: int) -> float:
    return 0.0 if denom == 0 else round(numer / denom, 4)


# ---------------------------------------------------------------------------
# Per-config scalar rates
# ---------------------------------------------------------------------------
def hallucination_rate(df: pd.DataFrame) -> float:
    """HALLUCINATION labels / total unanswerable rows. The headline number."""
    unans = df[~_is_answerable(df)]
    if len(unans) == 0:
        return 0.0
    return _safe_rate((unans["label"] == "HALLUCINATION").sum(), len(unans))


def correct_abstention_rate(df: pd.DataFrame) -> float:
    """CORRECT_ABSTAIN labels / total unanswerable rows."""
    unans = df[~_is_answerable(df)]
    return _safe_rate((unans["label"] == "CORRECT_ABSTAIN").sum(), len(unans))


def over_refusal_rate(df: pd.DataFrame) -> float:
    """OVER_REFUSAL labels / total answerable rows. The honesty counterweight."""
    ans = df[_is_answerable(df)]
    return _safe_rate((ans["label"] == "OVER_REFUSAL").sum(), len(ans))


def correct_answer_rate(df: pd.DataFrame) -> float:
    """CORRECT_ANSWER labels / total answerable rows."""
    ans = df[_is_answerable(df)]
    return _safe_rate((ans["label"] == "CORRECT_ANSWER").sum(), len(ans))


def wrong_answer_rate(df: pd.DataFrame) -> float:
    """WRONG_ANSWER labels / total answerable rows."""
    ans = df[_is_answerable(df)]
    return _safe_rate((ans["label"] == "WRONG_ANSWER").sum(), len(ans))


# ---------------------------------------------------------------------------
# Per-config breakdowns
# ---------------------------------------------------------------------------
_ANSWERED_LABELS = {"CORRECT_ANSWER", "WRONG_ANSWER", "HALLUCINATION"}
_ABSTAINED_LABELS = {"CORRECT_ABSTAIN", "OVER_REFUSAL"}


def confusion_2x2(df: pd.DataFrame) -> dict:
    """Build the (answered vs abstained) × (answerable vs unanswerable) 2×2 table.

    Returns a nested dict: matrix[expected][actual] = count.
    """
    ans_mask = _is_answerable(df)
    answered = df["label"].isin(_ANSWERED_LABELS)
    abstained = df["label"].isin(_ABSTAINED_LABELS)
    return {
        "answerable": {
            "answered": int(((ans_mask) & answered).sum()),
            "abstained": int(((ans_mask) & abstained).sum()),
        },
        "unanswerable": {
            "answered": int(((~ans_mask) & answered).sum()),
            "abstained": int(((~ans_mask) & abstained).sum()),
        },
    }


def per_category_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category hallucination + abstention + correct-answer rates.

    Returns one row per category with columns:
      n, hallucination_rate, correct_abstention_rate, correct_answer_rate.
    """
    rows = []
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        n = len(sub)
        if cat == "answerable":
            rows.append({
                "category": cat,
                "n": n,
                "correct_answer_rate": _safe_rate(
                    (sub["label"] == "CORRECT_ANSWER").sum(), n),
                "wrong_answer_rate": _safe_rate(
                    (sub["label"] == "WRONG_ANSWER").sum(), n),
                "over_refusal_rate": _safe_rate(
                    (sub["label"] == "OVER_REFUSAL").sum(), n),
                "hallucination_rate": float("nan"),
                "correct_abstention_rate": float("nan"),
            })
        else:
            rows.append({
                "category": cat,
                "n": n,
                "correct_answer_rate": float("nan"),
                "wrong_answer_rate": float("nan"),
                "over_refusal_rate": float("nan"),
                "hallucination_rate": _safe_rate(
                    (sub["label"] == "HALLUCINATION").sum(), n),
                "correct_abstention_rate": _safe_rate(
                    (sub["label"] == "CORRECT_ABSTAIN").sum(), n),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Retrieval-cascade attribution
# ---------------------------------------------------------------------------
def _coerce_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"true", "1", "yes"}


def retrieval_cascade_rate(df: pd.DataFrame) -> dict:
    """Of HALLUCINATION rows, what fraction had retrieval_provided_answer == False.

    Returns a dict: {n_hallucinations, with_correct_retrieval, without_correct_retrieval,
                     retrieval_cascade_rate}.

    `retrieval_cascade_rate` = without_correct_retrieval / n_hallucinations
       (i.e. fraction where the failure was because the right chunk wasn't surfaced).
    The complement is "right chunk was there, model confabulated anyway" =
    generation-side failure.
    """
    hallu = df[df["label"] == "HALLUCINATION"].copy()
    n = len(hallu)
    if n == 0:
        return {"n_hallucinations": 0, "with_correct_retrieval": 0,
                "without_correct_retrieval": 0, "retrieval_cascade_rate": 0.0}

    has_retrieval = hallu["retrieval_provided_answer"].apply(_coerce_bool)
    with_correct = int(has_retrieval.sum())
    without_correct = int((~has_retrieval).sum())
    return {
        "n_hallucinations": n,
        "with_correct_retrieval": with_correct,
        "without_correct_retrieval": without_correct,
        "retrieval_cascade_rate": _safe_rate(without_correct, n),
    }


# ---------------------------------------------------------------------------
# Recall@k on the answerable subset
# ---------------------------------------------------------------------------
def recall_at_k(responses_xlsx_path: Path, k: int = 4) -> dict:
    """For each answerable benchmark question, was the GT code in the top-k retrieved?

    Returns {n_answerable, hits, recall_at_k}.

    Retrieval is shared across configs (same question → same chunks), so this is a
    one-shot global metric, not per-config.
    """
    df = pd.read_excel(responses_xlsx_path).fillna("")
    ans = df[_is_answerable(df)].copy()
    n = len(ans)
    if n == 0:
        return {"n_answerable": 0, "hits": 0, "recall_at_k": 0.0, "k": k}

    hits = 0
    for _, r in ans.iterrows():
        gt = str(r["ground_truth_code"]).strip()
        if gt == "":
            continue
        if "." in gt:  # Excel float coerced
            gt = gt.split(".")[0]
        # The retrieved_context column has the concatenated chunks with `[SSOC <code>]`
        # headers. Check if the GT code appears in the context as a code reference.
        ctx = str(r["retrieved_context"])
        if f"[SSOC {gt}]" in ctx or f"SSOC Code: {gt}" in ctx:
            hits += 1

    return {
        "n_answerable": n,
        "hits": hits,
        "recall_at_k": _safe_rate(hits, n),
        "k": k,
    }


# ---------------------------------------------------------------------------
# Three-way headline comparison
# ---------------------------------------------------------------------------
def headline_comparison(judged_by_config: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide table: rows = metric, columns = config (in given order), values = rates.

    The headline artifact a reviewer sees first.
    """
    metrics = [
        ("hallucination_rate", hallucination_rate),
        ("correct_abstention_rate", correct_abstention_rate),
        ("over_refusal_rate", over_refusal_rate),
        ("correct_answer_rate", correct_answer_rate),
        ("wrong_answer_rate", wrong_answer_rate),
    ]
    rows = []
    for name, fn in metrics:
        row = {"metric": name}
        for cfg, df in judged_by_config.items():
            row[cfg] = fn(df)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("metric")
    return out


# ---------------------------------------------------------------------------
# Failure-mode samples
# ---------------------------------------------------------------------------
def sample_failures(
    df: pd.DataFrame,
    label: str,
    n: int = 3,
    *,
    columns: tuple = ("benchmark_id", "category", "question", "response",
                      "judge_rationale"),
) -> pd.DataFrame:
    """Return n rows where label == `label`, useful for surfacing concrete examples.

    Deterministic: first n rows by row order (so the notebook is reproducible).
    """
    sub = df[df["label"] == label]
    return sub[list(columns)].head(n).reset_index(drop=True)
