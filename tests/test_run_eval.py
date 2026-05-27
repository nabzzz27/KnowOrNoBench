"""Tests for src.eval.run_eval (no API calls — stubs answer_fn)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eval.run_eval import RESPONSES_COLUMNS, run_eval


def _stub_benchmark(n: int = 3) -> list[dict]:
    """Three tiny benchmark rows — mix of answerable + unanswerable."""
    return [
        {
            "id": "ans-001",
            "category": "answerable",
            "question": "Q1: software developer code?",
            "answerable": True,
            "expected_behavior": "answer",
            "ground_truth_code": "25121",
            "ground_truth_title": "Software developer",
            "rationale": "Direct lookup.",
            "difficulty": "easy",
        },
        {
            "id": "ssic-001",
            "category": "ssic_confusion",
            "question": "Q2: SSIC code for X?",
            "answerable": False,
            "expected_behavior": "abstain",
            "ground_truth_code": None,
            "ground_truth_title": None,
            "rationale": "SSIC not in source.",
            "difficulty": "easy",
        },
        {
            "id": "fab-001",
            "category": "false_premise",
            "question": "Q3: SSOC 99999?",
            "answerable": False,
            "expected_behavior": "abstain",
            "ground_truth_code": None,
            "ground_truth_title": None,
            "rationale": "Code does not exist.",
            "difficulty": "easy",
        },
    ][:n]


def _make_answer_stub():
    """A deterministic stub that returns a per-config response."""
    def _answer(question: str, *, config: str) -> dict:
        return {
            "question": question,
            "config": config,
            "retrieved_chunks": [{
                "id": "25121",
                "text": "software dev text",
                "distance": 0.27,
                "metadata": {"source": "ssoc_excel", "code": "25121",
                             "title": "Software developer"},
            }],
            "prompt": f"<{config}-prompt for {question}>",
            "response": f"<{config}-resp:{question[:6]}>",
        }
    return _answer


@pytest.fixture()
def patched_loader(monkeypatch):
    monkeypatch.setattr("src.eval.run_eval.load_benchmark", _stub_benchmark)


def test_run_eval_writes_xlsx_with_expected_columns(tmp_path, patched_loader):
    out = tmp_path / "responses.xlsx"
    run_eval(output_path=out, answer_fn=_make_answer_stub())
    df = pd.read_excel(out)
    assert list(df.columns) == RESPONSES_COLUMNS


def test_run_eval_one_row_per_question_with_all_three_configs(tmp_path, patched_loader):
    out = tmp_path / "responses.xlsx"
    summary = run_eval(output_path=out, answer_fn=_make_answer_stub())
    df = pd.read_excel(out)
    assert len(df) == 3
    # 3 questions × 3 configs = 9 cells filled
    for cfg in ("neutral", "forced", "strict"):
        assert df[f"{cfg}_response"].astype(str).str.startswith(f"<{cfg}-resp").all()
    # retrieved_context populated once per row
    assert (df["retrieved_context"].astype(str).str.contains("software dev text")).all()
    assert summary["processed"] == 9
    assert summary["errored"] == 0


def test_run_eval_is_idempotent_resume(tmp_path, patched_loader):
    out = tmp_path / "responses.xlsx"
    # First run fills everything
    run_eval(output_path=out, answer_fn=_make_answer_stub())

    # Second run should skip all 9 cells
    summary = run_eval(output_path=out, answer_fn=_make_answer_stub())
    assert summary["processed"] == 0
    assert summary["skipped"] == 9


def test_run_eval_handles_per_row_errors(tmp_path, patched_loader):
    out = tmp_path / "responses.xlsx"

    def flaky(question, *, config):
        # Fail only for the second benchmark question's "forced" config
        if "Q2" in question and config == "forced":
            raise RuntimeError("simulated failure")
        return _make_answer_stub()(question, config=config)

    summary = run_eval(output_path=out, answer_fn=flaky)
    df = pd.read_excel(out)

    # Failing cell: blank response, populated error
    q2 = df[df["benchmark_id"] == "ssic-001"].iloc[0]
    assert (q2["forced_response"] == "") or pd.isna(q2["forced_response"])
    assert "simulated failure" in str(q2["forced_error"])

    # Other 8 cells should have succeeded
    assert summary["processed"] == 8
    assert summary["errored"] == 1


def test_run_eval_rebuild_flag_starts_fresh(tmp_path, patched_loader):
    out = tmp_path / "responses.xlsx"
    run_eval(output_path=out, answer_fn=_make_answer_stub())

    # Pre-existing xlsx: rebuild should start over
    summary = run_eval(output_path=out, answer_fn=_make_answer_stub(), rebuild=True)
    assert summary["processed"] == 9
    assert summary["skipped"] == 0


def test_run_eval_only_runs_requested_configs(tmp_path, patched_loader):
    out = tmp_path / "responses.xlsx"
    summary = run_eval(output_path=out, configs=("strict",),
                       answer_fn=_make_answer_stub())
    df = pd.read_excel(out).fillna("")
    # strict_response populated for all 3 rows
    assert df["strict_response"].astype(str).str.startswith("<strict-resp").all()
    # neutral and forced cells should remain blank (read_excel turns blanks into NaN)
    assert (df["neutral_response"].astype(str) == "").all()
    assert (df["forced_response"].astype(str) == "").all()
    assert summary["processed"] == 3


def test_run_eval_rejects_unknown_config(tmp_path, patched_loader):
    out = tmp_path / "responses.xlsx"
    with pytest.raises(ValueError, match="unknown config"):
        run_eval(output_path=out, configs=("bogus",), answer_fn=_make_answer_stub())


def test_run_eval_limit_only_processes_first_n(tmp_path, patched_loader):
    out = tmp_path / "responses.xlsx"
    run_eval(output_path=out, answer_fn=_make_answer_stub(), limit=2)
    df = pd.read_excel(out)
    assert len(df) == 2
    assert set(df["benchmark_id"]) == {"ans-001", "ssic-001"}