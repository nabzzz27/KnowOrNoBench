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

### Chunking (Phase 1 of the production pipeline)

- **Two sources, one corpus (1,475 chunks):**
  - **Excel → 1,420 chunks** — both **4-digit unit groups (414)** and **5-digit occupations
    (1,006)**. 4-digit rows turned out to carry full definitions *and* a Tasks duty-list (97%
    populated), so they're substantive, not thin. (An early profiling pass wrongly reported
    Tasks as empty; caught by dumping a raw row before trusting the summary.)
  - **Report PDF pages 8–23 → 55 chunks** — pages are clean digital text (no OCR). Chunked
    **one chunk per numbered paragraph** (1.1, 2.5, …) rather than per page/section, because the
    user wanted smaller units; each is ~100–225 tokens and carries its section heading.
- **Excel column selection (user call):** include Definition + Tasks + Groups(child-list) +
  Examples-under + Examples-elsewhere; **drop Notes** and the 1/2/3-digit summary rows. Kept
  `Examples Classified Elsewhere` deliberately — it is the load-bearing signal for the
  false-premise abstention category (flagged this when the user initially wanted to drop it).
- **Division of labour:** Excel = per-code definitions; report = the conceptual/structural layer
  (what SSOC is, 2024-vs-2020 comparison, principles) — grounding for the abstention categories.
- **Report section labels:** derived each paragraph's section from its own number (2.13 → §2) and
  named sections from the genuine heading nearest before each section's first paragraph; this was
  needed because the major-groups table rows ("9 Cleaners… 5 12 24 65") otherwise masquerade as
  headings. Every chunk is dumped to `data/processed/chunks_preview.txt` for inspection before
  embedding. `tests/test_ingest.py` (15 tests) guards counts, unique ids, ex_else survival,
  token limits, and idempotency.
- **Chunk audit (post-build):** reviewing the full dump caught two report-text defects — the
  running page header/footer ("Singapore Department of Statistics … 2024" + a private-use glyph)
  leaking into 15/55 chunks, and de-hyphenation that joined real compounds ("five-digit" →
  "fivedigit"). Fixed: strip the footer signature line + PUA glyphs before splitting, and keep the
  hyphen when joining wrapped lines (9/11 line-end hyphens are genuine compounds). Excel chunks
  audited clean. Tables (e.g. §2.12 counts) left as linearised prose — accepted for a basic RAG.

- Any hand-curation or labelling effort: <placeholder>

## Evaluation decisions

- Why this eval methodology: <placeholder>
- Sample-size reasoning: <placeholder>
- What I'd change with more time: <placeholder>

## What surprised me

<placeholder>

## What I'd do differently

<placeholder>
