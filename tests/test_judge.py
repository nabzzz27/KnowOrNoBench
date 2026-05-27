"""Tests for src.eval.judge — both per-call judge() and orchestrator run_judge()."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from src.eval.judge import (
    JUDGED_COLUMNS, _parse_and_validate, _strip_code_fences,
    judge, run_judge,
)


# ===========================================================================
# judge() — per-call tests
# ===========================================================================

_VALID_JSON = '{"label": "CORRECT_ANSWER", "rationale": "matches GT", "retrieval_provided_answer": true}'


def test_judge_happy_path():
    def call(prompt, model):
        return _VALID_JSON
    out = judge("any prompt", _call=call)
    assert out == {
        "label": "CORRECT_ANSWER",
        "rationale": "matches GT",
        "retrieval_provided_answer": True,
    }


def test_judge_strips_json_code_fences():
    def call(prompt, model):
        return f"```json\n{_VALID_JSON}\n```"
    out = judge("any prompt", _call=call)
    assert out["label"] == "CORRECT_ANSWER"


def test_judge_strips_unlabeled_code_fences():
    def call(prompt, model):
        return f"```\n{_VALID_JSON}\n```"
    out = judge("any prompt", _call=call)
    assert out["label"] == "CORRECT_ANSWER"


def test_judge_invalid_label_raises_after_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    def call(prompt, model):
        return '{"label": "MAYBE", "rationale": "x", "retrieval_provided_answer": true}'
    with pytest.raises(ValueError, match="label"):
        judge("p", _call=call)


def test_judge_missing_rpa_raises_after_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    def call(prompt, model):
        return '{"label": "CORRECT_ANSWER", "rationale": "x"}'
    with pytest.raises(ValueError, match="retrieval_provided_answer"):
        judge("p", _call=call)


def test_judge_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = {"n": 0}
    def call(prompt, model):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("HTTP 429 rate limit exceeded")
        return _VALID_JSON
    out = judge("p", _call=call)
    assert out["label"] == "CORRECT_ANSWER"
    assert calls["n"] == 2


def test_judge_retries_on_json_parse_error(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = {"n": 0}
    def call(prompt, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return _VALID_JSON
    out = judge("p", _call=call)
    assert out["label"] == "CORRECT_ANSWER"
    assert calls["n"] == 2


def test_strip_code_fences_helper():
    assert _strip_code_fences("```json\n{\"a\":1}\n```") == '{"a":1}'
    assert _strip_code_fences("```\n{\"a\":1}\n```") == '{"a":1}'
    assert _strip_code_fences('{"a":1}') == '{"a":1}'


def test_parse_and_validate_rejects_non_object():
    with pytest.raises(ValueError, match="non-object"):
        _parse_and_validate("[1, 2, 3]")


# ===========================================================================
# run_judge() — orchestrator tests
# ===========================================================================

def _make_responses_xlsx(path: Path, n: int = 3) -> Path:
    """Build a tiny responses.xlsx with 3 questions × 3 configs."""
    rows = []
    for i in range(n):
        rows.append({
            "benchmark_id": f"ans-{i+1:03d}" if i < 2 else "fab-001",
            "category": "answerable" if i < 2 else "false_premise",
            "difficulty": "easy",
            "question": f"Q{i+1}?",
            "answerable": i < 2,
            "expected_behavior": "answer" if i < 2 else "abstain",
            "ground_truth_code": "25121" if i < 2 else "",
            "ground_truth_title": "Software developer" if i < 2 else "",
            "benchmark_rationale": "test row",
            "retrieved_context": "(stub context)",
            "neutral_response": f"neutral-{i+1}",
            "forced_response": f"forced-{i+1}",
            "strict_response": f"strict-{i+1}",
            "neutral_error": "",
            "forced_error": "",
            "strict_error": "",
        })
    pd.DataFrame(rows).to_excel(path, index=False, sheet_name="responses")
    return path


def _judge_stub(prompt: str) -> dict:
    """Deterministic stub: returns CORRECT_ANSWER unless prompt says 'abstain'."""
    if "abstain" in prompt[:500]:
        return {"label": "HALLUCINATION", "rationale": "stub",
                "retrieval_provided_answer": False}
    return {"label": "CORRECT_ANSWER", "rationale": "stub",
            "retrieval_provided_answer": True}


def test_run_judge_writes_three_per_config_files(tmp_path):
    inp = _make_responses_xlsx(tmp_path / "responses.xlsx")
    summary = run_judge(input_path=inp, output_dir=tmp_path, judge_fn=_judge_stub)

    expected = [tmp_path / f"judged_{c}.xlsx" for c in ("neutral", "forced", "strict")]
    for p in expected:
        assert p.exists(), f"missing {p}"
        df = pd.read_excel(p)
        assert list(df.columns) == JUDGED_COLUMNS
        assert len(df) == 3

    assert summary["processed"] == 9  # 3 rows × 3 configs
    assert summary["errored"] == 0


def test_run_judge_response_column_holds_per_config_response(tmp_path):
    inp = _make_responses_xlsx(tmp_path / "responses.xlsx")
    run_judge(input_path=inp, output_dir=tmp_path, judge_fn=_judge_stub)

    df_strict = pd.read_excel(tmp_path / "judged_strict.xlsx")
    df_neutral = pd.read_excel(tmp_path / "judged_neutral.xlsx")
    # strict file should hold strict_response in the renamed 'response' column
    assert df_strict.iloc[0]["response"] == "strict-1"
    assert df_neutral.iloc[0]["response"] == "neutral-1"


def test_run_judge_is_idempotent_resume(tmp_path):
    inp = _make_responses_xlsx(tmp_path / "responses.xlsx")
    # First pass labels everything
    run_judge(input_path=inp, output_dir=tmp_path, judge_fn=_judge_stub)
    # Second pass should skip all 9
    summary = run_judge(input_path=inp, output_dir=tmp_path, judge_fn=_judge_stub)
    assert summary["processed"] == 0
    assert summary["skipped"] == 9


def test_run_judge_handles_per_cell_errors(tmp_path):
    inp = _make_responses_xlsx(tmp_path / "responses.xlsx")

    def flaky(prompt):
        if "Q2" in prompt:
            raise RuntimeError("simulated judge failure")
        return _judge_stub(prompt)

    summary = run_judge(input_path=inp, output_dir=tmp_path, judge_fn=flaky)
    df = pd.read_excel(tmp_path / "judged_strict.xlsx").fillna("")
    row2 = df[df["benchmark_id"] == "ans-002"].iloc[0]
    assert row2["label"] == ""
    assert "[JUDGE_ERROR]" in row2["judge_rationale"]
    assert "simulated judge failure" in row2["judge_rationale"]
    assert summary["errored"] >= 1


def test_run_judge_only_writes_requested_configs(tmp_path):
    inp = _make_responses_xlsx(tmp_path / "responses.xlsx")
    run_judge(input_path=inp, output_dir=tmp_path, configs=("strict",),
              judge_fn=_judge_stub)
    assert (tmp_path / "judged_strict.xlsx").exists()
    assert not (tmp_path / "judged_neutral.xlsx").exists()
    assert not (tmp_path / "judged_forced.xlsx").exists()


def test_run_judge_rejects_unknown_config(tmp_path):
    inp = _make_responses_xlsx(tmp_path / "responses.xlsx")
    with pytest.raises(ValueError, match="unknown config"):
        run_judge(input_path=inp, output_dir=tmp_path, configs=("bogus",),
                  judge_fn=_judge_stub)


def test_run_judge_limit_caps_rows(tmp_path):
    inp = _make_responses_xlsx(tmp_path / "responses.xlsx", n=3)
    run_judge(input_path=inp, output_dir=tmp_path, limit=2, judge_fn=_judge_stub)
    df = pd.read_excel(tmp_path / "judged_strict.xlsx")
    assert len(df) == 2


def test_run_judge_rebuild_starts_fresh(tmp_path):
    inp = _make_responses_xlsx(tmp_path / "responses.xlsx")
    run_judge(input_path=inp, output_dir=tmp_path, judge_fn=_judge_stub)
    summary = run_judge(input_path=inp, output_dir=tmp_path,
                        judge_fn=_judge_stub, rebuild=True)
    assert summary["processed"] == 9
    assert summary["skipped"] == 0