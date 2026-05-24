# Process

A short narrative of how this was built — what was tried, what was dropped, which tools made which decisions, and where I exercised judgment.

## Time spent

- Total focused hours: <placeholder>
- Rough split (exploration / build / eval / writeup): <placeholder>

## Problem selection

- Problems considered: <placeholder>
- Why I picked this one: <placeholder>
- What I deliberately ruled out and why: <placeholder>

## What I tried

### De-risking spike (first thing built, before any production code)

Wrote a single throwaway script (`scripts/spike.py`, run via `--dry-run` then live) with six
probes, each retiring one risky assumption before investing in the real pipeline. A `--dry-run`
mode prints the exact 100-chunk sample manifest with **zero API calls**, so the data going into
the embedder is inspectable before anything is spent.

Findings (full detail in `spike_log.md`):
- **Stack installs on Python 3.14.3** — `google-genai 2.6.0`, `chromadb 1.5.9`, `pandas 3.0.3`,
  `openpyxl 3.1.5` all import. (Biggest unknown, since 3.14 is bleeding-edge.)
- **Excel parses with no cleanup** — header at sheet row 5; 1,006 five-digit detailed
  occupations; `Examples Classified Elsewhere` populated on 59% of them.
- **Retrieval is strong** — cosine recall@1 = recall@5 = **100%** on a 100-chunk stratified
  sample with 10 hand-written queries. Confirms `TOP_K=4` is ample; no reranking/hybrid needed.
- **Embedding is fast but rate-limited** — batch of 100 in ~2.6s, but the free tier counts quota
  **per text at ~100/min**; back-to-back calls 429 immediately.

- What I kept: the architecture as specified (structure-aware chunking, 768-dim asymmetric
  Gemini embeddings, Chroma cosine, Flash generator). All validated by the spike.
- What I dropped/changed: short exponential backoff (2s/4s) — useless against a per-minute quota
  window. Replaced with proactive pacing (<=95 texts/60s) + a 60s wait on 429. This is the
  mitigation the production `embed.py` will inherit.

## Tools and models

- Coding agents used and for what: <placeholder>
- Models tried and the final choice: <placeholder>
- Where I trusted the agent vs. overrode it: <placeholder>

## Judgment calls

- Key design decisions and the tradeoffs: <placeholder>
- Things I chose NOT to build, with reasons: <placeholder>

## Data decisions

- Source choices and why: <placeholder>
- Any hand-curation or labelling effort: <placeholder>

## Evaluation decisions

- Why this eval methodology: <placeholder>
- Sample-size reasoning: <placeholder>
- What I'd change with more time: <placeholder>

## What surprised me

<placeholder>

## What I'd do differently

<placeholder>
