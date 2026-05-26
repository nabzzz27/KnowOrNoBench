"""Orchestrate the RAG-on-benchmark run.

For each of the 60 benchmark questions, call `src.rag.answer.answer` three times (one per
config: neutral / forced / strict), and accumulate results into a single-sheet Excel file
at `results/responses.xlsx`. The xlsx is both the deliverable and the resume marker — an
interrupted run can be resumed without burning API calls again because already-filled
response cells are skipped on restart.

The 16-column schema is described in the Phase 5 plan.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from src import config
from src.eval.benchmark_loader import load_benchmark


DEFAULT_OUTPUT_PATH = config.REPO_ROOT / "results" / "responses.xlsx"

VALID_CONFIGS = ("neutral", "forced", "strict")

# Columns persisted in `results/responses.xlsx` (order is the canonical column order).
RESPONSES_COLUMNS = [
    "benchmark_id",
    "category",
    "difficulty",
    "question",
    "answerable",
    "expected_behavior",
    "ground_truth_code",
    "ground_truth_title",
    "benchmark_rationale",
    "retrieved_context",
    "neutral_response",
    "forced_response",
    "strict_response",
    "neutral_error",
    "forced_error",
    "strict_error",
]


def _format_retrieved_context(chunks: list[dict]) -> str:
    """Reconstruct what the prompt's {context} substitution looked like.

    Mirrors `src.rag.prompts.format_context` — kept here as a small inline duplicate
    rather than importing, since this module records what the model saw, not what gets
    sent next time. If the prompt's context formatting changes, eval rows recorded
    earlier should not retroactively change shape.
    """
    if not chunks:
        return ""
    blocks = []
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        code = meta.get("code") or c.get("id") or ""
        title = meta.get("title") or ""
        blocks.append(f"[SSOC {code}] {title}\n{c['text']}")
    return "\n---\n".join(blocks)


def _empty_responses_df(benchmark_rows: list[dict]) -> pd.DataFrame:
    """Build the empty DataFrame with one row per benchmark question, all response and
    error cells blank, and the benchmark metadata pre-populated."""
    rows = []
    for q in benchmark_rows:
        rows.append({
            "benchmark_id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "answerable": q["answerable"],
            "expected_behavior": q["expected_behavior"],
            "ground_truth_code": q["ground_truth_code"] or "",
            "ground_truth_title": q["ground_truth_title"] or "",
            "benchmark_rationale": q["rationale"],
            "retrieved_context": "",
            "neutral_response": "",
            "forced_response": "",
            "strict_response": "",
            "neutral_error": "",
            "forced_error": "",
            "strict_error": "",
        })
    return pd.DataFrame(rows, columns=RESPONSES_COLUMNS)


def _cell_is_blank(value) -> bool:
    """A cell is 'not yet processed' iff blank or NaN."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _write(df: pd.DataFrame, path: Path) -> None:
    """Atomic-ish write: pandas writes the whole sheet in one shot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False, sheet_name="responses")


def run_eval(
    output_path: Path | None = None,
    *,
    configs: Sequence[str] = VALID_CONFIGS,
    rebuild: bool = False,
    limit: int | None = None,
    answer_fn: Callable[..., dict] | None = None,
) -> dict:
    """Run the benchmark through the RAG for the given configs and write results.xlsx.

    Args:
        output_path: where to write the xlsx. Defaults to results/responses.xlsx.
        configs: subset of VALID_CONFIGS to run. Other configs' columns stay blank
            (or keep their existing values if the xlsx is being resumed).
        rebuild: if True, ignore any existing xlsx and start fresh.
        limit: if set, only process the first N benchmark rows. For smoke testing.
        answer_fn: test seam. Defaults to `src.rag.answer.answer`.

    Returns:
        Summary dict: processed / skipped / errored / output_path / elapsed_seconds.
    """
    for c in configs:
        if c not in VALID_CONFIGS:
            raise ValueError(f"unknown config {c!r}; expected subset of {VALID_CONFIGS}")

    out = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH
    benchmark_rows = load_benchmark()
    if limit is not None:
        benchmark_rows = benchmark_rows[:limit]

    if rebuild or not out.exists():
        df = _empty_responses_df(benchmark_rows)
    else:
        df = pd.read_excel(out)
        # If the existing xlsx has fewer rows than the benchmark (e.g. limit grew),
        # extend with blank rows.
        existing_ids = set(df["benchmark_id"].astype(str))
        new_rows = [q for q in benchmark_rows if q["id"] not in existing_ids]
        if new_rows:
            df = pd.concat([df, _empty_responses_df(new_rows)], ignore_index=True)

    if answer_fn is None:
        from src.rag.answer import answer as default_answer
        answer_fn = default_answer

    processed = 0
    skipped = 0
    errored = 0
    started_at = time.time()

    # Iterate in benchmark order so progress is predictable.
    bench_ids = [q["id"] for q in benchmark_rows]
    bench_by_id = {q["id"]: q for q in benchmark_rows}

    for bid in bench_ids:
        q = bench_by_id[bid]
        row_idx = df.index[df["benchmark_id"].astype(str) == bid][0]

        for cfg in configs:
            resp_col = f"{cfg}_response"
            err_col = f"{cfg}_error"

            if not _cell_is_blank(df.at[row_idx, resp_col]):
                skipped += 1
                continue

            try:
                result = answer_fn(q["question"], config=cfg)
                df.at[row_idx, resp_col] = result.get("response", "") or ""
                df.at[row_idx, err_col] = ""
                # retrieved_context is the same across configs for a given question, so
                # set it once when it's still blank.
                if _cell_is_blank(df.at[row_idx, "retrieved_context"]):
                    df.at[row_idx, "retrieved_context"] = _format_retrieved_context(
                        result.get("retrieved_chunks", []))
                processed += 1
                print(f"[run_eval] {bid}/{cfg}: ok")
            except Exception as e:  # noqa: BLE001
                df.at[row_idx, resp_col] = ""
                df.at[row_idx, err_col] = repr(e)
                errored += 1
                print(f"[run_eval] {bid}/{cfg}: ERROR {type(e).__name__}: {e}")

            # Persist after every cell so an interrupt loses at most one call.
            _write(df, out)

    elapsed = time.time() - started_at

    return {
        "processed": processed,
        "skipped": skipped,
        "errored": errored,
        "output_path": str(out),
        "elapsed_seconds": round(elapsed, 2),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
