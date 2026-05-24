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

### Phase 2 — Embed & Index (the real build vs. the spike)

Lifted the spike's pacer + retry into `src/embed.py` (`embed_documents`, `embed_query` — the
function names enforce the RETRIEVAL_DOCUMENT vs. RETRIEVAL_QUERY asymmetry at the call site).
Built `src/index.py` around a persistent Chroma collection with `hnsw:space=cosine` set
explicitly; `build_index` is idempotent + crash-resumable (queries existing ids first, embeds
only the missing set). That property turned out to be load-bearing — see below.

**Live finding during the real build: the *daily* RPD is the binding constraint, not RPM.** The
spike measured the per-minute quota (~100 texts/min, text-counted) and our pacer handled it
perfectly. But on the production build of 1,475 chunks, we ran out at batch 10/15 (~1,000
chunks indexed) and started hitting persistent 429s the pacer could not resolve — because the
**free-tier `gemini-embedding-001` daily cap is 1,000 text-requests / day** (also text-counted),
and RPD only resets at midnight Pacific. The pacer guards against the per-minute window; the
daily window is a separate, harder ceiling.

This was a real lesson: when validating a paced system, measure both *per-minute throughput*
and *expected daily volume*. The spike covered the former; the latter only surfaced live.

**Mitigation:** the idempotent resume in `build_index` made recovery a no-op — the 1,000
already-indexed chunks persisted across sessions; re-running tomorrow would embed only the
remaining ~475 (well under the next day's cap).

**Decision: enable Google AI Studio billing.** The brief explicitly permits this ("works on
free tier and would scale cleanly with a budget — that's fine"); the architecture remains
free-tier-compatible (the system itself doesn't require paid — billing is only for development
speed and the upcoming eval). Estimated total project cost is well under $1 (embedding ~$0.02
one-time; full eval ~$0.30–$0.50). The bigger reason: Phase 3's eval would otherwise need
~13 days at the free-tier Flash RPD of 20/day (85 questions × 3 configs = 255 calls).
Documented in the README "Cost & free-tier behavior" section (Phase 4) per the brief's "state
your limitations clearly" requirement. After billing was enabled, the build finished in ~5 min
via the resumability path, `count == 1475`, sanity queries returned the gold codes at rank 1.

### Phase 2.5 — Retrieval-quality probe (verifying the embeddings before generation)

Before building the answer-generation pipeline, I wanted a defensible answer to a basic
question: are the embeddings good enough that the right chunk is reachable in the first place?
Generation-side problems can be debugged by tweaking prompts; retrieval-side problems cannot.
So I separated the two failure modes with a dedicated diagnostic notebook
(`notebooks/retrieval_quality.ipynb`).

25 hand-curated queries: 15 answerable (known gold codes), 6 unanswerable (SSIC confusion,
salary, fabricated code, obsolete version, off-topic), 4 edge (typo, ambiguous, niche, broad
category). For each: embed once, retrieve top-5 from Chroma, capture rank + distance + full
chunk text.

**Results.** recall@4 = 15/15 = **100%**; recall@1 = 73% — the 4 misses are all parent ↔ child
or sibling confusions (e.g. "Member of Parliament" → unit-group `1111` above 5-digit `11110`),
inherent to chunking both 4- and 5-digit codes. With `TOP_K = 4` the gold is always in the
context window, so the generator will still see it. MRR = 0.844.

**Distance separation is clean** between answerable and unanswerable medians (0.218 vs 0.356) —
a usable signal for threshold-based abstention later if needed, though prompt-based abstention
remains the primary lever. Retrieval is **deterministic** (embedding model has no sampling,
HNSW is deterministic once built, chunks are static); re-running the notebook yields the same
results.

**Notable finding.** The obsolete-version query ("SSOC code for IT support in SSOC 2010")
landed inside the answerable/unanswerable distance overlap zone — the system semantically
surfaced a current-day IT-support code. This is exactly where prompt-based abstention (using
the report chunks about SSOC-2024-vs-2020) will need to do work that retrieval alone cannot.

### Robustness pass — considered, deferred

A code audit surfaced 8 defensive-fix candidates (header-rename robustness, hash-based vector
freshness, NA handling, etc.). Verified that none break the current build; the only actively
noisy one (`report-6.1` swallowing a Skills-Framework footnote and the page-23 "Part II:"
divider) is reasonable encyclopedic text attached to the wrong paragraph, not garbage. Skipped
the pass: this is a prototype where the **eval methodology is the deliverable**, not a
production RAG. The brief explicitly says the RAG-under-test is meant to be deliberately
simple. Hardening would be appropriate for a production system; here it would trade
eval-rigour time for marginal RAG robustness. Recorded as a deliberate trade-off.

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
