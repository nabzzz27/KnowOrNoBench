"""Tests for src.eval.sample_for_kappa — stratified sampler + xlsx writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.eval.sample_for_kappa import (
    CONFIGS, DEFAULT_SAMPLING, HAND_LABELS_COLUMNS,
    sample_for_kappa, write_hand_labels_xlsx,
)
from src.eval.judge_prompt import VALID_LABELS


def _make_synthetic_responses_xlsx(path: Path) -> Path:
    """Make a stub responses.xlsx matching the production schema with enough rows in
    each category to satisfy DEFAULT_SAMPLING."""
    rows = []
    # 20 answerable, 10 of each unanswerable cat — same shape as the real benchmark.
    for i in range(20):
        rows.append(_row(f"ans-{i+1:03d}", "answerable", True, "answer",
                         "25121", "Software developer"))
    for cat, prefix in [("ssic_confusion", "ssic"),
                        ("obsolete_version", "obs"),
                        ("beyond_corpus_attribute", "bca"),
                        ("false_premise", "fab")]:
        for i in range(10):
            rows.append(_row(f"{prefix}-{i+1:03d}", cat, False, "abstain", "", ""))

    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, sheet_name="responses")
    return path


def _row(bid, cat, ans, eb, gtc, gtt):
    return {
        "benchmark_id": bid,
        "category": cat,
        "difficulty": "easy",
        "question": f"Q for {bid}?",
        "answerable": ans,
        "expected_behavior": eb,
        "ground_truth_code": gtc,
        "ground_truth_title": gtt,
        "benchmark_rationale": "test row",
        "retrieved_context": "(ctx)",
        "neutral_response": f"n-resp-{bid}",
        "forced_response": f"f-resp-{bid}",
        "strict_response": f"s-resp-{bid}",
        "neutral_error": "",
        "forced_error": "",
        "strict_error": "",
    }


def test_sampler_is_deterministic(tmp_path):
    inp = _make_synthetic_responses_xlsx(tmp_path / "responses.xlsx")
    df1 = sample_for_kappa(inp, seed=42)
    df2 = sample_for_kappa(inp, seed=42)
    # Compare as sets-of-tuples; sort order is also stable but we want true determinism.
    keys1 = list(zip(df1["benchmark_id"], df1["config"], df1["split"]))
    keys2 = list(zip(df2["benchmark_id"], df2["config"], df2["split"]))
    assert keys1 == keys2


def test_sampler_total_is_60_split_30_30(tmp_path):
    inp = _make_synthetic_responses_xlsx(tmp_path / "responses.xlsx")
    df = sample_for_kappa(inp, seed=42)
    assert len(df) == 60
    assert (df["split"] == "dev").sum() == 30
    assert (df["split"] == "test").sum() == 30


def test_sampler_stratification_per_category(tmp_path):
    inp = _make_synthetic_responses_xlsx(tmp_path / "responses.xlsx")
    df = sample_for_kappa(inp, seed=42)
    for cat, (n_dev, n_test) in DEFAULT_SAMPLING.items():
        cat_rows = df[df["category"] == cat]
        assert (cat_rows["split"] == "dev").sum() == n_dev, f"{cat} dev count"
        assert (cat_rows["split"] == "test").sum() == n_test, f"{cat} test count"


def test_sampled_pairs_exist_in_input(tmp_path):
    inp = _make_synthetic_responses_xlsx(tmp_path / "responses.xlsx")
    df = sample_for_kappa(inp, seed=42)
    src = pd.read_excel(inp).fillna("")
    valid_ids = set(src["benchmark_id"].astype(str))
    for bid in df["benchmark_id"]:
        assert bid in valid_ids
    for cfg in df["config"]:
        assert cfg in CONFIGS


def test_response_column_matches_per_config_response(tmp_path):
    inp = _make_synthetic_responses_xlsx(tmp_path / "responses.xlsx")
    df = sample_for_kappa(inp, seed=42)
    # Pick one row and verify its response matches the source per-config response.
    r = df.iloc[0]
    expected = f"{r['config'][0]}-resp-{r['benchmark_id']}"
    assert r["response"] == expected


def test_hand_label_and_notes_columns_start_empty(tmp_path):
    inp = _make_synthetic_responses_xlsx(tmp_path / "responses.xlsx")
    df = sample_for_kappa(inp, seed=42)
    assert (df["hand_label"] == "").all()
    assert (df["notes"] == "").all()


def test_write_xlsx_has_three_sheets_and_validation(tmp_path):
    inp = _make_synthetic_responses_xlsx(tmp_path / "responses.xlsx")
    df = sample_for_kappa(inp, seed=42)
    out_path = tmp_path / "hand_labels.xlsx"
    write_hand_labels_xlsx(df, out_path, seed=42)
    assert out_path.exists()

    sheets = pd.ExcelFile(out_path).sheet_names
    assert "hand_labels" in sheets
    assert "rubric" in sheets
    assert "metadata" in sheets

    read_back = pd.read_excel(out_path, sheet_name="hand_labels").fillna("")
    assert list(read_back.columns) == HAND_LABELS_COLUMNS
    assert len(read_back) == 60
    assert (read_back["hand_label"] == "").all()

    # Verify dropdown validation was attached
    from openpyxl import load_workbook
    wb = load_workbook(out_path)
    ws = wb["hand_labels"]
    dvs = list(ws.data_validations.dataValidation)
    assert len(dvs) == 1
    dv = dvs[0]
    for lbl in VALID_LABELS:
        assert lbl in dv.formula1


def test_metadata_sheet_records_seed_and_counts(tmp_path):
    inp = _make_synthetic_responses_xlsx(tmp_path / "responses.xlsx")
    df = sample_for_kappa(inp, seed=42)
    out_path = tmp_path / "hand_labels.xlsx"
    write_hand_labels_xlsx(df, out_path, seed=42)
    meta = pd.read_excel(out_path, sheet_name="metadata")
    md = dict(zip(meta["key"], meta["value"]))
    assert md["seed"] == "42"
    assert md["total_rows"] == "60"
    assert md["dev_rows"] == "30"
    assert md["test_rows"] == "30"


def test_sampler_raises_when_pool_too_small(tmp_path):
    # Create an xlsx with only 2 answerable rows — far less than the 20 we need.
    rows = [_row("ans-001", "answerable", True, "answer", "25121", "Software developer"),
            _row("ans-002", "answerable", True, "answer", "25121", "Software developer")]
    for cat, prefix in [("ssic_confusion", "ssic"),
                        ("obsolete_version", "obs"),
                        ("beyond_corpus_attribute", "bca"),
                        ("false_premise", "fab")]:
        for i in range(10):
            rows.append(_row(f"{prefix}-{i+1:03d}", cat, False, "abstain", "", ""))

    p = tmp_path / "responses.xlsx"
    pd.DataFrame(rows).to_excel(p, index=False)
    with pytest.raises(ValueError, match="answerable"):
        sample_for_kappa(p, seed=42)
