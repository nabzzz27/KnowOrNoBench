"""Headless one-time build of the persistent Chroma index from the chunk JSONL.

Used by Docker / cold-start automation. For interactive build with progress + sanity-query
inspection, prefer `notebooks/build_index.ipynb` (which calls the same `src.index.build_index`).

Usage:
    python -m scripts.build_index               # idempotent: fills any missing chunks
    python -m scripts.build_index --rebuild     # wipe and rebuild from scratch
"""

from __future__ import annotations

import argparse
import json
import time

from src import config, ingest
from src.index import build_index


def _load_or_build_chunks() -> list[dict]:
    if not config.CHUNKS_PATH.exists():
        print(f"[build_index] {config.CHUNKS_PATH.relative_to(config.REPO_ROOT)} missing; "
              f"running ingest first...")
        chunks = ingest.ingest_all()
        ingest.write_jsonl(chunks)
        ingest.write_preview(chunks)
        return chunks
    with config.CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser(description="Build the SSOC Chroma index (one-time).")
    ap.add_argument("--rebuild", action="store_true",
                    help="delete the existing collection and rebuild from scratch")
    args = ap.parse_args()

    chunks = _load_or_build_chunks()
    print(f"[build_index] loaded {len(chunks)} chunks; building at {config.CHROMA_PATH}")
    t0 = time.time()
    col = build_index(chunks, rebuild=args.rebuild)
    elapsed = time.time() - t0
    print(f"[build_index] done: collection.count() = {col.count()} ; elapsed {elapsed:.0f}s")


if __name__ == "__main__":
    main()
