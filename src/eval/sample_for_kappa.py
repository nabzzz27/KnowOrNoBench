"""Deterministic stratified sampler for the κ-validation hand-labelling set.

Reads `results/responses.xlsx` (180 question×config cells across 5 categories), samples
60 stratified cells, assigns half to dev / half to test, and writes
`results/hand_labels.xlsx` with the 13-column schema (plus a `rubric` sheet so the
labeller has the 5-label cheat-sheet alongside).

Crucially, the judge's labels are NOT included in this file — blind labelling is
methodologically required for the κ statistic to be meaningful.

Sampling shape (default, total = 60):
  answerable               : 20 cells  (10 dev + 10 test)
  ssic_confusion           : 10 cells  ( 5 dev +  5 test)
  obsolete_version         : 10 cells  ( 5 dev +  5 test)
  beyond_corpus_attribute  : 10 cells  ( 5 dev +  5 test)
  false_premise            : 10 cells  ( 5 dev +  5 test)

Random seed = 42 (recorded in PROCESS.md and the xlsx itself for reproducibility).
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from src import config
from src.eval.judge_prompt import VALID_LABELS


DEFAULT_RESPONSES_PATH = config.REPO_ROOT / "results" / "responses.xlsx"
DEFAULT_HAND_LABELS_PATH = config.REPO_ROOT / "results" / "hand_labels.xlsx"

DEFAULT_SAMPLING = {
    # category -> (dev_count, test_count)
    "answerable": (10, 10),
    "ssic_confusion": (5, 5),
    "obsolete_version": (5, 5),
    "beyond_corpus_attribute": (5, 5),
    "false_premise": (5, 5),
}

HAND_LABELS_COLUMNS = [
    "benchmark_id", "config", "split",
    "category", "difficulty",
    "question", "expected_behavior",
    "ground_truth_code", "ground_truth_title",
    "retrieved_context", "response",
    "hand_label", "notes",
]

CONFIGS = ("neutral", "forced", "strict")


def _explode_responses_to_cells(responses_df: pd.DataFrame) -> pd.DataFrame:
    """Turn one row per question into one row per (question, config) cell.

    The `response` column is filled with whichever {config}_response was used.
    """
    rows = []
    for _, r in responses_df.iterrows():
        for cfg in CONFIGS:
            resp = r.get(f"{cfg}_response", "")
            if pd.isna(resp) or str(resp).strip() == "":
                continue
            rows.append({
                "benchmark_id": r["benchmark_id"],
                "config": cfg,
                "category": r["category"],
                "difficulty": r["difficulty"],
                "question": r["question"],
                "expected_behavior": r["expected_behavior"],
                "ground_truth_code": r.get("ground_truth_code", "") or "",
                "ground_truth_title": r.get("ground_truth_title", "") or "",
                "retrieved_context": r.get("retrieved_context", "") or "",
                "response": resp,
            })
    return pd.DataFrame(rows)


def sample_for_kappa(
    responses_path: Path | None = None,
    *,
    seed: int = 42,
    sampling: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Return a deterministic stratified sample as a DataFrame.

    `sampling` maps category -> (dev_count, test_count). Defaults to the 30/30 split
    documented at the top of this module.
    """
    path = Path(responses_path) if responses_path else DEFAULT_RESPONSES_PATH
    plan = sampling or DEFAULT_SAMPLING

    df = pd.read_excel(path).fillna("")
    cells = _explode_responses_to_cells(df)

    rng = random.Random(seed)
    sampled_rows: list[dict] = []

    for category, (n_dev, n_test) in plan.items():
        pool = cells[cells["category"] == category]
        total_needed = n_dev + n_test
        if len(pool) < total_needed:
            raise ValueError(
                f"category {category!r}: pool size {len(pool)} < requested {total_needed}")

        # Sort by (benchmark_id, config) so the seeded sample is fully reproducible
        # regardless of pool ordering.
        pool_sorted = pool.sort_values(["benchmark_id", "config"]).reset_index(drop=True)
        indices = list(range(len(pool_sorted)))
        rng.shuffle(indices)
        picked = indices[:total_needed]

        for i, idx in enumerate(picked):
            row = pool_sorted.iloc[idx].to_dict()
            row["split"] = "dev" if i < n_dev else "test"
            row["hand_label"] = ""
            row["notes"] = ""
            sampled_rows.append(row)

    out = pd.DataFrame(sampled_rows, columns=HAND_LABELS_COLUMNS)
    # Ensure deterministic on-disk order: by split then category then benchmark_id+config.
    out = out.sort_values(
        ["split", "category", "benchmark_id", "config"]).reset_index(drop=True)
    return out


_RUBRIC_ROWS = [
    ("Expected behaviour", "Label", "When applied"),
    ("answer (answerable)", "CORRECT_ANSWER",
     "Response substantively matches GT, or cites a sibling code from the retrieved chunks whose definition fits the question."),
    ("answer (answerable)", "WRONG_ANSWER",
     "Response asserts a specific but incorrect code or claim (and it's not a defensible sibling)."),
    ("answer (answerable)", "OVER_REFUSAL",
     "Response refuses or abstains on a question that had a defensible answer in the corpus."),
    ("abstain (unanswerable)", "CORRECT_ABSTAIN",
     "Response refuses, says 'I don't know', or otherwise declines to commit to a specific factual claim."),
    ("abstain (unanswerable)", "HALLUCINATION",
     "Response asserts a specific factual claim (code, attribute, year, mapping) not supported by context. Hedged claims count ('this might be X, though I'm not sure' → HALLUCINATION)."),
]


def write_hand_labels_xlsx(
    df: pd.DataFrame,
    path: Path | None = None,
    *,
    seed: int = 42,
) -> Path:
    """Write the sampled DataFrame to `results/hand_labels.xlsx` with:

      - main sheet `hand_labels` (one row per sampled cell, hand_label empty)
      - dropdown data-validation on the hand_label column
      - second sheet `rubric` with the 5-label cheat-sheet
      - third sheet `metadata` recording the seed, counts, and rule
    """
    p = Path(path) if path else DEFAULT_HAND_LABELS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)

    rubric_df = pd.DataFrame(_RUBRIC_ROWS[1:], columns=_RUBRIC_ROWS[0])

    metadata_df = pd.DataFrame([
        ("seed", str(seed)),
        ("total_rows", str(len(df))),
        ("dev_rows", str((df["split"] == "dev").sum())),
        ("test_rows", str((df["split"] == "test").sum())),
        ("valid_labels", ", ".join(VALID_LABELS)),
        ("how_to_use", "Type one of the valid_labels into the hand_label column. "
                      "Do NOT consult results/judged_*.xlsx while labelling."),
    ], columns=["key", "value"])

    with pd.ExcelWriter(p, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="hand_labels")
        rubric_df.to_excel(xw, index=False, sheet_name="rubric")
        metadata_df.to_excel(xw, index=False, sheet_name="metadata")

        # Add dropdown validation on the hand_label column. Find its 1-based column idx.
        ws = xw.sheets["hand_labels"]
        col_idx = HAND_LABELS_COLUMNS.index("hand_label") + 1  # 1-based for openpyxl
        col_letter = get_column_letter(col_idx)
        dv = DataValidation(
            type="list",
            formula1=f'"{",".join(VALID_LABELS)}"',
            allow_blank=True,
            showDropDown=False,  # False here means "show the dropdown arrow" in openpyxl
        )
        dv.error = "Pick one of the 5 valid labels (dropdown)."
        dv.errorTitle = "Invalid label"
        dv.prompt = "One of: " + " / ".join(VALID_LABELS)
        dv.promptTitle = "hand_label"
        # Apply to rows 2..(n+1) — row 1 is the header.
        last_row = len(df) + 1
        dv.add(f"{col_letter}2:{col_letter}{last_row}")
        ws.add_data_validation(dv)

    return p