"""Tests for src.eval.benchmark_loader (no API calls — uses tmp xlsx fixtures)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.eval import benchmark_loader
from src.eval.benchmark_loader import (
    VALID_CATEGORIES, VALID_DIFFICULTIES, load_benchmark,
)


_BASE_ROW = {
    "id": "ans-001",
    "category": "answerable",
    "question": "What's the SSOC code for a software developer?",
    "answerable": True,
    "expected_behavior": "answer",
    "ground_truth_code": "25121",
    "ground_truth_title": "Software developer",
    "rationale": "Canonical lookup.",
    "difficulty": "easy",
}

_UNANS_ROW = {
    "id": "ssic-001",
    "category": "ssic_confusion",
    "question": "What's the SSIC code for software developers?",
    "answerable": False,
    "expected_behavior": "abstain",
    "ground_truth_code": "",
    "ground_truth_title": "",
    "rationale": "SSIC is a different system; source is SSOC.",
    "difficulty": "easy",
}


def _write_xlsx(rows: list[dict], path: Path) -> Path:
    df = pd.DataFrame(rows, columns=list(_BASE_ROW.keys()))
    df.to_excel(path, index=False, sheet_name="questions")
    return path


def test_loader_reads_valid_rows(tmp_path):
    p = _write_xlsx([_BASE_ROW, _UNANS_ROW], tmp_path / "q.xlsx")
    rows = load_benchmark(p)
    assert len(rows) == 2
    assert rows[0]["id"] == "ans-001"
    assert rows[0]["answerable"] is True
    assert rows[0]["ground_truth_code"] == "25121"
    assert rows[1]["answerable"] is False
    assert rows[1]["ground_truth_code"] is None  # blank -> None
    assert rows[1]["ground_truth_title"] is None


def test_loader_raises_on_missing_columns(tmp_path):
    p = tmp_path / "q.xlsx"
    pd.DataFrame([{"id": "ans-001", "question": "x"}]).to_excel(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_benchmark(p)


def test_loader_raises_on_duplicate_ids(tmp_path):
    row2 = dict(_BASE_ROW)
    p = _write_xlsx([_BASE_ROW, row2], tmp_path / "q.xlsx")
    with pytest.raises(ValueError, match="duplicate id"):
        load_benchmark(p)


def test_loader_raises_on_prefix_category_mismatch(tmp_path):
    bad = dict(_BASE_ROW)
    bad["id"] = "ssic-001"  # prefix says ssic but category says answerable
    p = _write_xlsx([bad], tmp_path / "q.xlsx")
    with pytest.raises(ValueError, match="prefix"):
        load_benchmark(p)


def test_loader_raises_on_answerable_without_ground_truth(tmp_path):
    bad = dict(_BASE_ROW)
    bad["ground_truth_code"] = ""
    p = _write_xlsx([bad], tmp_path / "q.xlsx")
    with pytest.raises(ValueError, match="ground_truth_code"):
        load_benchmark(p)


def test_loader_raises_on_unanswerable_with_ground_truth(tmp_path):
    bad = dict(_UNANS_ROW)
    bad["ground_truth_code"] = "99999"
    p = _write_xlsx([bad], tmp_path / "q.xlsx")
    with pytest.raises(ValueError, match="blank"):
        load_benchmark(p)


def test_loader_raises_on_invalid_category(tmp_path):
    bad = dict(_BASE_ROW)
    bad["category"] = "bogus_category"
    bad["id"] = "bogus-001"
    p = _write_xlsx([bad], tmp_path / "q.xlsx")
    with pytest.raises(ValueError):
        load_benchmark(p)


def test_loader_raises_on_invalid_difficulty(tmp_path):
    bad = dict(_BASE_ROW)
    bad["difficulty"] = "extreme"
    p = _write_xlsx([bad], tmp_path / "q.xlsx")
    with pytest.raises(ValueError, match="difficulty"):
        load_benchmark(p)


def test_loader_raises_on_overlong_question(tmp_path):
    bad = dict(_BASE_ROW)
    bad["question"] = "x" * 500
    p = _write_xlsx([bad], tmp_path / "q.xlsx")
    with pytest.raises(ValueError, match="too long"):
        load_benchmark(p)


def test_loader_raises_on_answerable_with_abstain_behavior(tmp_path):
    bad = dict(_BASE_ROW)
    bad["expected_behavior"] = "abstain"
    p = _write_xlsx([bad], tmp_path / "q.xlsx")
    with pytest.raises(ValueError, match="expected_behavior"):
        load_benchmark(p)


def test_constants_in_sync():
    """Sanity: the VALID_CATEGORIES / VALID_DIFFICULTIES tuples match what we doc."""
    assert set(VALID_CATEGORIES) == {
        "answerable", "ssic_confusion", "obsolete_version",
        "beyond_corpus_attribute", "false_premise",
    }
    assert set(VALID_DIFFICULTIES) == {"easy", "medium", "hard"}
