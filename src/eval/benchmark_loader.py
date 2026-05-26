"""Load and validate the hand-curated benchmark from ``benchmark/questions.xlsx``.

Excel is the canonical source — humans edit the xlsx directly, the pipeline reads it via
pandas. This module enforces the schema at read time so a typo'd row is caught before
the eval spends API calls on it.

Schema (one row per question):
    id (str)                  unique, prefix matches category (ans-/ssic-/obs-/bca-/fab-)
    category (str)            one of the five valid category names
    question (str)            user-facing question text
    answerable (bool)         True iff category == "answerable"
    expected_behavior (str)   "answer" | "abstain"
    ground_truth_code (str)   required for answerable; blank otherwise
    ground_truth_title (str)  required for answerable; blank otherwise
    rationale (str)           one-sentence justification
    difficulty (str)          "easy" | "medium" | "hard"
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


BENCHMARK_PATH = config.REPO_ROOT / "benchmark" / "questions.xlsx"

VALID_CATEGORIES = (
    "answerable",
    "ssic_confusion",
    "obsolete_version",
    "beyond_corpus_attribute",
    "false_premise",
)

_PREFIX_TO_CATEGORY = {
    "ans": "answerable",
    "ssic": "ssic_confusion",
    "obs": "obsolete_version",
    "bca": "beyond_corpus_attribute",
    "fab": "false_premise",
}

VALID_DIFFICULTIES = ("easy", "medium", "hard")
VALID_EXPECTED_BEHAVIOURS = ("answer", "abstain")

REQUIRED_COLUMNS = (
    "id", "category", "question", "answerable", "expected_behavior",
    "ground_truth_code", "ground_truth_title", "rationale", "difficulty",
)


def _normalise_blank(v) -> str:
    """Treat NaN/None/whitespace as empty string for optional text fields.

    Also strips Excel's habit of coercing ID-shaped strings (e.g. SSOC codes like '25121')
    to floats — '25121.0' is rewritten back to '25121'.
    """
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def _coerce_bool(v, row_id: str) -> bool:
    """Excel writes TRUE/FALSE; pandas may give bool or string. Be strict."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    raise ValueError(f"row {row_id}: cannot coerce answerable={v!r} to bool")


def load_benchmark(path: Path | None = None) -> list[dict]:
    """Read ``benchmark/questions.xlsx`` and return a list of typed question dicts.

    Validates row-by-row: required columns present, types coercible, ID prefix matches
    category, answerable rows have non-empty ground-truth fields. Raises ValueError on
    any violation, naming the offending row id.
    """
    p = path or BENCHMARK_PATH
    if not p.exists():
        raise FileNotFoundError(f"benchmark file not found: {p}")

    df = pd.read_excel(p)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"benchmark xlsx missing required columns: {missing_cols}. "
            f"Got: {list(df.columns)}")

    seen_ids: set[str] = set()
    out: list[dict] = []
    for i, row in df.iterrows():
        rid = _normalise_blank(row["id"])
        if not rid:
            raise ValueError(f"row {i+2}: empty id (row {i+2} in xlsx; first data row is 2)")
        if rid in seen_ids:
            raise ValueError(f"row {rid}: duplicate id")
        seen_ids.add(rid)

        category = _normalise_blank(row["category"])
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"row {rid}: category {category!r} not in {VALID_CATEGORIES}")

        # ID prefix must match category
        prefix = rid.split("-", 1)[0]
        expected_category = _PREFIX_TO_CATEGORY.get(prefix)
        if expected_category != category:
            raise ValueError(
                f"row {rid}: id prefix {prefix!r} implies category {expected_category!r} "
                f"but row says {category!r}")

        question = _normalise_blank(row["question"])
        if not question:
            raise ValueError(f"row {rid}: empty question")
        if len(question) > 400:
            raise ValueError(
                f"row {rid}: question too long ({len(question)} chars; max 400)")

        answerable = _coerce_bool(row["answerable"], rid)
        # answerable bool must match category
        if answerable and category != "answerable":
            raise ValueError(
                f"row {rid}: answerable=True but category={category!r}")
        if not answerable and category == "answerable":
            raise ValueError(
                f"row {rid}: answerable=False but category='answerable'")

        expected_behavior = _normalise_blank(row["expected_behavior"])
        if expected_behavior not in VALID_EXPECTED_BEHAVIOURS:
            raise ValueError(
                f"row {rid}: expected_behavior {expected_behavior!r} not in "
                f"{VALID_EXPECTED_BEHAVIOURS}")
        if answerable and expected_behavior != "answer":
            raise ValueError(
                f"row {rid}: answerable=True must have expected_behavior='answer'")
        if not answerable and expected_behavior != "abstain":
            raise ValueError(
                f"row {rid}: answerable=False must have expected_behavior='abstain'")

        gt_code = _normalise_blank(row["ground_truth_code"])
        gt_title = _normalise_blank(row["ground_truth_title"])
        if answerable:
            if not gt_code:
                raise ValueError(f"row {rid}: answerable row missing ground_truth_code")
            if not gt_title:
                raise ValueError(f"row {rid}: answerable row missing ground_truth_title")
        else:
            if gt_code or gt_title:
                raise ValueError(
                    f"row {rid}: unanswerable row must leave ground_truth_code "
                    f"and ground_truth_title blank")

        rationale = _normalise_blank(row["rationale"])
        if not rationale:
            raise ValueError(f"row {rid}: empty rationale")

        difficulty = _normalise_blank(row["difficulty"]).lower()
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"row {rid}: difficulty {difficulty!r} not in {VALID_DIFFICULTIES}")

        out.append({
            "id": rid,
            "category": category,
            "question": question,
            "answerable": answerable,
            "expected_behavior": expected_behavior,
            "ground_truth_code": gt_code or None,
            "ground_truth_title": gt_title or None,
            "rationale": rationale,
            "difficulty": difficulty,
        })

    return out
