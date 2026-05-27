"""Run the LLM-as-judge over results/responses.xlsx, write per-config judged xlsx files.

Usage:
    python -m scripts.run_judge                  # all 3 configs, resume from existing files
    python -m scripts.run_judge --config strict  # judge only strict's responses
    python -m scripts.run_judge --rebuild        # discard existing judged_*.xlsx and re-judge
    python -m scripts.run_judge --limit 5        # smoke test: first 5 benchmark rows only
"""

from __future__ import annotations

import argparse
import json

from src.eval.judge import VALID_CONFIGS, run_judge


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the LLM-as-judge on RAG responses.")
    ap.add_argument(
        "--config", default="all",
        choices=list(VALID_CONFIGS) + ["all"],
        help="Which config's responses to judge (default: all).",
    )
    ap.add_argument(
        "--rebuild", action="store_true",
        help="Discard any existing results/judged_{config}.xlsx and start fresh.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Only judge the first N benchmark rows (for smoke testing).",
    )
    args = ap.parse_args()

    configs = VALID_CONFIGS if args.config == "all" else (args.config,)
    summary = run_judge(configs=configs, rebuild=args.rebuild, limit=args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()