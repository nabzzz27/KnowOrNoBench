"""LLM-as-judge: per-call API client + benchmark orchestrator.

This module contains the entire judge surface in one file:
  - `judge(prompt) -> dict`      calls gemini-2.5-pro (Google AI Studio) with strict
                                 JSON mode and validates the schema.
  - `run_judge(...) -> dict`     iterates results/responses.xlsx, writes three
                                 per-config xlsx files (results/judged_{config}.xlsx).

Mirrors the structure of src/rag/answer.py — one file, two public functions.

Original plan was Llama 3.3 70B via Groq (non-Gemini family) for self-preference-bias
avoidance. Groq's free-tier daily-token quota exhausted mid-run, so we pivoted to
gemini-2.5-pro on the project's existing Gemini credit. The same-family caveat is
disclosed in PROCESS.md and README/Limitations.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
from dotenv import load_dotenv

from src import config
from src.eval.judge_prompt import VALID_LABELS, format_judge_prompt


load_dotenv(config.REPO_ROOT / ".env")


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


# ---------------------------------------------------------------------------
# Per-call client: judge()
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    """Lazy google-genai client. Reuses GEMINI_API_KEY from .env (same key the
    generator uses; Google AI Studio gives one key for embeddings + gen + judge)."""
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env at the repo root.")
        from google import genai
        _client = genai.Client(api_key=key)
    return _client


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "resource_exhausted" in s or "rate" in s and "limit" in s or "quota" in s


def _strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json fences; strip them defensively."""
    return _CODE_FENCE_RE.sub("", text.strip()).strip()


def _parse_and_validate(text: str) -> dict:
    """Parse JSON and validate the shape against the judge contract."""
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"judge returned non-JSON: {cleaned[:200]!r}") from e

    if not isinstance(data, dict):
        raise ValueError(f"judge returned non-object JSON: {type(data).__name__}")

    label = data.get("label")
    if label not in VALID_LABELS:
        raise ValueError(
            f"judge label {label!r} not in {VALID_LABELS}")

    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"judge rationale missing or empty: {rationale!r}")

    rpa = data.get("retrieval_provided_answer")
    if not isinstance(rpa, bool):
        raise ValueError(
            f"judge retrieval_provided_answer must be bool, got {type(rpa).__name__}")

    return {
        "label": label,
        "rationale": rationale.strip(),
        "retrieval_provided_answer": rpa,
    }


def _default_call(prompt: str, model: str) -> str:
    """Call Gemini via google-genai with strict-JSON mode (response_mime_type)."""
    from google.genai import types
    client = _get_client()
    res = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    return res.text or ""


def judge(
    prompt: str,
    *,
    model: str | None = None,
    max_tries: int = 3,
    _call: Callable[[str, str], str] | None = None,
) -> dict:
    """Call the judge LLM with `prompt`, parse JSON, return the validated dict.

    Retries up to `max_tries` total:
      - 429 / rate-limit / quota → wait 60s and retry
      - JSON-parse / validation errors → retry once with the same prompt
      - other exceptions → propagate after retries exhausted

    `_call` is the transport seam — tests inject a stub.
    """
    mdl = model or config.JUDGE_MODEL
    call = _call or _default_call
    last_exc: Exception | None = None

    for attempt in range(1, max_tries + 1):
        try:
            raw = call(prompt, mdl)
            return _parse_and_validate(raw)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt == max_tries:
                break
            wait = 60 if _is_rate_limit(e) else 2 ** attempt
            print(f"[judge retry] attempt {attempt} failed ({type(e).__name__}: {e}); "
                  f"sleeping {wait}s")
            time.sleep(wait)

    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Orchestrator: run_judge()
# ---------------------------------------------------------------------------
DEFAULT_INPUT_PATH = config.REPO_ROOT / "results" / "responses.xlsx"
DEFAULT_OUTPUT_DIR = config.REPO_ROOT / "results"

VALID_CONFIGS = ("neutral", "forced", "strict")

# 15 columns per per-config judged xlsx.
JUDGED_COLUMNS = [
    "benchmark_id", "category", "difficulty", "question",
    "answerable", "expected_behavior",
    "ground_truth_code", "ground_truth_title",
    "benchmark_rationale", "retrieved_context",
    "response", "response_error",
    "label", "judge_rationale", "retrieval_provided_answer",
]


def _empty_judged_df(responses_df: pd.DataFrame, cfg: str) -> pd.DataFrame:
    """Build the empty 15-col judged DataFrame for one config from responses.xlsx.

    Judge-output columns are forced to ``object`` dtype so later cell assignments can
    store mixed types (str for `label` / `judge_rationale`, bool for
    `retrieval_provided_answer`) without pandas 3.x dtype-strictness errors.
    """
    rows = []
    for _, r in responses_df.iterrows():
        rows.append({
            "benchmark_id": r["benchmark_id"],
            "category": r["category"],
            "difficulty": r["difficulty"],
            "question": r["question"],
            "answerable": r["answerable"],
            "expected_behavior": r["expected_behavior"],
            "ground_truth_code": r.get("ground_truth_code", "") or "",
            "ground_truth_title": r.get("ground_truth_title", "") or "",
            "benchmark_rationale": r["benchmark_rationale"],
            "retrieved_context": r["retrieved_context"],
            "response": r.get(f"{cfg}_response", "") or "",
            "response_error": r.get(f"{cfg}_error", "") or "",
            "label": "",
            "judge_rationale": "",
            "retrieval_provided_answer": None,
        })
    df = pd.DataFrame(rows, columns=JUDGED_COLUMNS)
    for col in ("label", "judge_rationale", "retrieval_provided_answer"):
        df[col] = df[col].astype(object)
    return df


def _cell_is_blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return str(v).strip() == ""


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False, sheet_name="judged")


def run_judge(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    *,
    configs: Sequence[str] = VALID_CONFIGS,
    rebuild: bool = False,
    limit: int | None = None,
    judge_fn: Callable[[str], dict] | None = None,
) -> dict:
    """Iterate results/responses.xlsx, write results/judged_{config}.xlsx per config.

    Args:
        input_path:   responses.xlsx path. Defaults to results/responses.xlsx.
        output_dir:   directory for the per-config xlsx files. Defaults to results/.
        configs:      subset of VALID_CONFIGS to process.
        rebuild:      if True, ignore any existing judged_{config}.xlsx and start fresh.
        limit:        if set, only judge the first N benchmark rows.
        judge_fn:     test seam; defaults to `judge` in this module.

    Returns:
        Summary dict: processed / skipped / errored / output_paths / elapsed_seconds.
    """
    for c in configs:
        if c not in VALID_CONFIGS:
            raise ValueError(f"unknown config {c!r}; expected subset of {VALID_CONFIGS}")

    inp = Path(input_path) if input_path else DEFAULT_INPUT_PATH
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    if not inp.exists():
        raise FileNotFoundError(f"responses xlsx not found: {inp}")

    responses_df = pd.read_excel(inp).fillna("")
    if limit is not None:
        responses_df = responses_df.head(limit).reset_index(drop=True)

    jfn = judge_fn or judge

    processed = 0
    skipped = 0
    errored = 0
    output_paths: list[str] = []
    started_at = time.time()

    for cfg in configs:
        out_path = out_dir / f"judged_{cfg}.xlsx"
        output_paths.append(str(out_path))

        if rebuild or not out_path.exists():
            df = _empty_judged_df(responses_df, cfg)
        else:
            df = pd.read_excel(out_path).fillna("")
            # Extend if the existing xlsx is shorter than the (possibly grown) benchmark.
            existing_ids = set(df["benchmark_id"].astype(str))
            new_ids = [str(r["benchmark_id"]) for _, r in responses_df.iterrows()
                       if str(r["benchmark_id"]) not in existing_ids]
            if new_ids:
                extra = _empty_judged_df(
                    responses_df[responses_df["benchmark_id"].astype(str).isin(new_ids)],
                    cfg,
                )
                df = pd.concat([df, extra], ignore_index=True)

        for idx, row in df.iterrows():
            # Skip cells already judged.
            if not _cell_is_blank(row["label"]):
                skipped += 1
                continue

            # Skip rows where the RAG response itself is missing (no response → nothing
            # for the judge to classify; record blank).
            if _cell_is_blank(row["response"]):
                df.at[idx, "judge_rationale"] = "[JUDGE_SKIP] no RAG response to judge"
                _write(df, out_path)
                skipped += 1
                continue

            prompt = format_judge_prompt(
                question=str(row["question"]),
                expected_behavior=str(row["expected_behavior"]),
                ground_truth=(
                    f"{row['ground_truth_code']} {row['ground_truth_title']}".strip()
                    if str(row["ground_truth_code"]).strip()
                    else None
                ),
                retrieved_context=str(row["retrieved_context"]),
                response=str(row["response"]),
            )

            try:
                result = jfn(prompt)
                df.at[idx, "label"] = result["label"]
                df.at[idx, "judge_rationale"] = result["rationale"]
                df.at[idx, "retrieval_provided_answer"] = bool(
                    result["retrieval_provided_answer"])
                processed += 1
                print(f"[run_judge] {row['benchmark_id']}/{cfg}: {result['label']}")
            except Exception as e:  # noqa: BLE001
                df.at[idx, "judge_rationale"] = f"[JUDGE_ERROR] {type(e).__name__}: {e}"
                df.at[idx, "label"] = ""
                df.at[idx, "retrieval_provided_answer"] = None
                errored += 1
                print(f"[run_judge] {row['benchmark_id']}/{cfg}: ERROR {type(e).__name__}")

            _write(df, out_path)

    elapsed = time.time() - started_at
    return {
        "processed": processed,
        "skipped": skipped,
        "errored": errored,
        "output_paths": output_paths,
        "elapsed_seconds": round(elapsed, 2),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }