"""Run the RAG over the benchmark for all (or one) configs.

Usage:
    python -m scripts.run_full_eval                 # all 3 configs, resume from existing xlsx
    python -m scripts.run_full_eval --config strict # only strict
    python -m scripts.run_full_eval --rebuild       # start fresh
    python -m scripts.run_full_eval --limit 3       # smoke test on first 3 benchmark rows
"""

from __future__ import annotations

import argparse
import json

from src.eval.run_eval import VALID_CONFIGS, run_eval


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the benchmark through the RAG.")
    ap.add_argument(
        "--config", default="all",
        choices=list(VALID_CONFIGS) + ["all"],
        help="Which RAG config to run (default: all = neutral + forced + strict).",
    )
    ap.add_argument(
        "--rebuild", action="store_true",
        help="Discard any existing results/responses.xlsx and start fresh.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N benchmark rows (for smoke testing).",
    )
    args = ap.parse_args()

    configs = VALID_CONFIGS if args.config == "all" else (args.config,)
    summary = run_eval(configs=configs, rebuild=args.rebuild, limit=args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()