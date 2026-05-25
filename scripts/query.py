"""Ad-hoc query CLI for feeling the RAG before the formal eval runs.

Usage:
    python -m scripts.query "What is SSOC 25121?" --config strict
    python -m scripts.query "What is SSOC 25121?" --config forced
    python -m scripts.query "What is SSOC 25121?" --config neutral

This is NOT what the eval calls. The eval imports `answer` directly.
"""

from __future__ import annotations

import argparse
import json

from src.rag import answer


_PREVIEW_CHARS = 300


def _shrink_for_display(record: dict) -> dict:
    out = dict(record)
    out["retrieved_chunks"] = [
        {
            "id": c["id"],
            "distance": round(c["distance"], 4),
            "title": c["metadata"].get("title", ""),
            "source": c["metadata"].get("source", ""),
            "text_preview": (
                c["text"][:_PREVIEW_CHARS] + "…"
                if len(c["text"]) > _PREVIEW_CHARS else c["text"]
            ),
        }
        for c in record["retrieved_chunks"]
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one RAG query end-to-end.")
    ap.add_argument("question", help="The question to ask.")
    ap.add_argument(
        "--config", default="strict",
        choices=["neutral", "forced", "strict"],
        help="Which RAG configuration to use (default: strict).",
    )
    ap.add_argument(
        "--full", action="store_true",
        help="Print full chunk texts (default: truncated previews).",
    )
    args = ap.parse_args()

    record = answer(args.question, config=args.config)
    display = record if args.full else _shrink_for_display(record)
    print(json.dumps(display, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
