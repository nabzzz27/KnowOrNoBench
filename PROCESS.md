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

### Phase 3 — Retrieve & Generate

Built `src/rag/` as a small sub-package the eval contracts against:
`retrieve.py` (wraps `embed_query` + Chroma), `prompts.py` (frozen templates +
`format_context`), `generate.py` (Gemini 2.5 Flash at temp 0 with inline retry),
`answer.py` (orchestrator). Public surface is a single `answer(question, config) → dict`
returning `{question, config, retrieved_chunks, prompt, response}` — fully JSON-serialisable
and drops straight into per-question eval records. `scripts/query.py` is a CLI feel-tester,
not used by the eval.

- **Two configs, not three.** Initially scoped NAIVE / ABSTENTION / ZEROSHOT, then dropped
  ZEROSHOT after thinking through what the benchmark actually measures. KnowOrNoBench
  measures the *RAG's* hallucination behaviour, so the relevant counterfactual is
  "same retrieval, no abstention rule" (NAIVE), not "no retrieval at all" (ZEROSHOT).
  ZEROSHOT answers a different research question ("does retrieval help?") that is downstream
  of the headline result. Cutting it tightened scope by one prompt to freeze, one branch in
  the orchestrator, and 3 tests. The PRD's contrast surface is now NAIVE vs ABSTENTION,
  isolating exactly the abstention rule's contribution.

- **Prompts frozen.** NAIVE = "use this context, answer the question" with no abstention
  rule. ABSTENTION = same context block plus four explicit rules: (1) only use the context,
  (2) reply "I don't know" + state what is missing when context is insufficient, (3) refuse
  false-premise questions and explain why the SSOC 2024 source does not support the premise,
  (4) cite the SSOC code(s) when answering. Frozen because any wording change invalidates
  any κ already measured against the previous wording — iteration belongs on a dev split.

- **Test seams.** Both `retrieve()` and `generate()` accept optional injected functions
  (`embed_fn`/`col` and `gen_fn`), matching the existing pattern in `src/index.py`. Tests
  pass deterministic stubs; production uses defaults. 21 new tests, zero API spend.

- **`__init__.py` naming gotcha.** Re-exporting the function `answer` from `__init__.py`
  shadows the `src.rag.answer` submodule, so `from src.rag import answer as answer_mod`
  binds the function, not the module. Fixed by switching tests to
  `from src.rag.answer import answer` (which sidesteps the shadow). The eval can still
  import the public surface as planned: `from src.rag import answer`.

- **Inline retry, no shared helper.** Generate has its own 3-line rate-limit detector
  duplicated from `src/embed.py` rather than a shared `_retry.py`. Trivial duplication
  beats a premature abstraction — the two modules' retry policies will likely diverge
  (per-text minute window for embeddings vs. per-request quota for generation), and the
  abstraction would have to be re-thought when that happens.

- **Verification checkpoint.** Full pytest green (48 total, 21 new, 0 API calls).
  Two real queries against the live Chroma + Gemini Flash:
  - Answerable: `"What is SSOC 25121?"` → top-1 = 25121 (Software developer, distance 0.27),
    response cleanly quotes the definition from the chunk.
  - Unanswerable (false premise): `"What is the SSOC code for a unicorn trainer?"` and
    `"What was SSOC code 25121 in the 2010 version?"` across both configs.

- **Unexpected finding at the checkpoint.** Both NAIVE and ABSTENTION refused both
  unanswerable questions — NAIVE without being told the rule. ABSTENTION used the verbatim
  "I don't know" phrase as instructed; NAIVE produced longer-form refusals explaining what
  was missing. Read: Gemini 2.5 Flash is honest-by-default when context plainly does not
  contain the answer. The benchmark's job is therefore *not* "does the abstention rule
  catch trivial misses" — it is to find the borderline cases where retrieval surfaces
  plausible-but-wrong chunks (parent ↔ child code confusion, sibling occupations,
  obsolete-version queries that semantically match a current code) and NAIVE confabulates a
  citation while ABSTENTION holds the line. The spike already flagged the obsolete-version
  case as the highest-risk overlap zone (top-1 distance 0.27, identical to the answerable
  median); benchmark curation should oversample these borderline categories rather than
  obvious false premises.

- **Cost.** ~3 API calls at the checkpoint, < $0.001.

### Phase 3.5 — Prompt-config refactor (NAIVE/ABSTENTION → NEUTRAL/FORCED/STRICT)

The Phase 3 checkpoint exposed a methodology gap: the NAIVE-vs-ABSTENTION contrast
collapsed because Flash 2.5 is honest-by-default at temp 0. Without an *explicit*
counterfactual that removes the model's natural caution, the benchmark could only measure
"does the abstention rule polish behaviour the model already has?" rather than "does the
rule prevent hallucination the model would otherwise produce?". A weaker question.

The fix was a 3-config contrast surface:
- **`neutral`** — natural baseline (no rule). Same wording as the retired NAIVE_PROMPT.
- **`forced`** — explicitly forbids refusal. Generates known-positive hallucinations the
  judge must catch; together with the known-negative `strict/answerable` cells, this gives
  the eval ground-truth labels without manual annotation on those rows. NEW.
- **`strict`** — must use context, must refuse if unsupported. Same wording as the retired
  ABSTENTION_PROMPT, just renamed. The production-intended behaviour and the default for
  `answer(question)` calls with no config argument.

Temperature stayed at 0.0 (CLAUDE.md constraint, reproducible eval). The new `forced`
prompt does the work that raising temperature would have done — it breaks Flash 2.5 out
of safe-mode by explicitly forbidding refusal language ("I don't know", "I cannot answer",
"the context does not contain that information").

- **Re-freeze decision.** The prior prompts were committed as frozen but no judge κ had
  been measured against them, so re-freezing is cheap right now. After Phase 4 (judge
  validation), prompt edits will be expensive — κ would need to be re-measured.

- **Re-verification on the SAME unanswerable questions that previously collapsed:**

  Question: *"What is the SSOC code for a unicorn trainer?"*
  - `strict`: "I don't know. The occupation 'unicorn trainer' is not supported by the SSOC 2024 source provided." ✓
  - `forced`: **"The SSOC code for a unicorn trainer is 51943."** ← confabulated; grabbed
    the top-retrieved unrelated code and committed
  - `neutral`: explanatory refusal — "Unicorns are mythical creatures..." (natural behaviour)

  Question: *"What was SSOC code 25121 in the 2010 version?"*
  - `strict`: "I don't know - The context does not contain information about SSOC codes from the 2010 version." ✓
  - `forced`: **"The SSOC 2010 code for what is now SSOC 25121 (Software developer) was 2512."** ← *confidently invented an SSOC 2010 mapping*. Exactly the failure mode the benchmark needs to detect.
  - `neutral`: refusal with explanation that the context only covers 2024.

  Both questions now produce a sharp three-way contrast where the previous two-config
  design produced two near-identical refusals.

- **What `forced` is for (corrected framing).** It is NOT a hallucination ceiling that the
  RAG dials down from — nobody would ship the `forced` prompt in production, so "ceiling
  of a continuous knob" is a contrived story. The sharper framing: `forced` is a
  *positive control for the judge*. On an unanswerable question we know by construction
  that `forced` MUST confabulate, so if the judge labels a `forced/unanswerable` response
  as "correct refusal", the judge is broken. Likewise `strict/answerable` is a negative
  control. These two cell-types give the eval ground truth without manual labelling; the
  remaining hand labels go on `strict/unanswerable`, which is where the headline κ is
  measured. `neutral` is no longer "the contrast condition" — it is a covariate for
  whether the model already abstains without the rule.

- **Methodology gaps still present (flagged for the README, not closed by this refactor):**
  1. `neutral` is "our baseline wording", not "the typical RAG baseline" — every team's
     default prompt is different. Frame as such in the writeup.
  2. `forced`-induced hallucinations may not be representative of real-world ones. A
     judge that aces the `forced/unanswerable` rows can still miss subtler hallucinations
     the model would produce naturally. Mitigation: the κ-validation set must include
     hand-labelled real hallucination examples, not rely on `forced` rows alone.
  3. The judge labels each (question, config, response) row uniformly as
     abstained / hallucinated / correct_answer. The rubric stays the same; what differs is
     the *expected* label per cell-type: `forced/unanswerable` → hallucinated,
     `strict/answerable` → correct_answer, `strict/unanswerable` → abstained.
     Construction-level ground truth on the first two; hand labels on the third.
  4. Single-shot at temp 0 → no variance bars on the metric. Acceptable for v1 (the
     contrast is the headline, not the distribution); future work could re-eval at temp
     0.7 with N samples.
  5. Three configs cannot save a weak question set — discrimination only emerges if
     benchmark questions span false-premise, missing-data, paraphrase, parent/child
     confusion, etc. That work happens in benchmark curation (CATEGORY_RULES.md).

- **Test count:** 48 → 50 (added third-config tests in test_rag_prompts.py and
  test_rag_answer.py). All green, no API spent.

- **Cost of the refactor:** ~7 API calls across the two-question 3-way contrast,
  < $0.002.

### Phase 3.7 — forced/answerable verification

The Phase 3.6 framing implicitly assumed `forced/answerable` is a clean known-negative
ground-truth row (the model should answer correctly because both the context AND the
forced rule push that way). All previous checkpoints only ran `forced` against
*unanswerable* questions, so this assumption was untested. Ran an N=1 empirical check
to either back it or flag it.

Question: *"What is SSOC 25121?"* (answerable; top-1 = 25121, distance 0.27)

- `strict` response: full verbatim definition pulled from the chunk —
  *"SSOC 25121 is 'Software developer'. Definition: Software developer researches,
  designs and develops computer and network software... [continues, verbatim]."*
- `forced` response: paraphrased but semantically faithful —
  *"SSOC 25121 is for a Software developer. A Software developer researches, designs,
  and develops computer and network software or specialised utility programmes. They
  analyse user needs, develop intuitive and responsive software solutions, update
  software, enhance existing capabilities, and direct software testing and validation
  procedures..."*

Compared on three axes:
- **Code accuracy:** both cite 25121 ✓
- **Definition fidelity:** strict = verbatim; forced = paraphrased, no facts added or
  removed
- **Outside-knowledge contamination:** none visible in either — no invented salary
  ranges, alternate codes, year-of-creation, or career path additions

**Verdict (N=1):** `forced/answerable` is clean in this sample. Rule 3's "use outside
knowledge" clause did NOT fire because the context was complete. The Phase 3.6 framing
stands: this cell is a known-negative.

**Caveats worth flagging for the eval:**
1. N=1. Topics where retrieved context is thinner (e.g. report chunks) could behave
   differently — rule 3 is more likely to fire there. The full eval will surface this
   if it happens; do not over-claim cleanliness from one sample.
2. Forced paraphrases rather than quoting verbatim. The judge prompt MUST recognise
   "same facts, different wording" as a correct answer — penalising paraphrase would
   systematically downscore `forced/answerable` and corrupt the negative-control
   labels.

### Phase 4 — Benchmark curation (60 hand-curated questions, Excel-canonical)

Built `benchmark/questions.xlsx` (the single source of truth) and `src/eval/benchmark_loader.py`
(typed-schema loader with row-level validation). 60 questions across 5 categories: 20
answerable + 4 × 10 unanswerable (SSIC confusion, obsolete version, beyond-corpus
attribute, false/fabricated premise). Difficulty target 1/3 easy / 1/3 medium / 1/3 hard
in each category.

- **Singlish category dropped** (PRD had 6 cats; we kept 5). Reason: curation scope. The
  SG-specific narrative still rests on SSIC + obsolete-version.
- **Taxonomy citation:** Niu et al., "RAGTruth" (ACL 2024) — the four unanswerable
  categories map onto RAGTruth's four documented RAG failure types (evidence-conflicting,
  subjective/unsupported, out-of-context, baseless information). Justification recorded
  in `benchmark/README.md`.
- **Obsolete-version disclaimer:** category 3 uses plausible-pattern invented pre-2024
  codes (not validated against historical SingStat publications). Tests temporal
  grounding behaviour, not historical mapping accuracy. Disclosed in README.
- **Drafting discipline:** category-by-category interactive drafting with the user
  reviewing every question before it landed in the xlsx. Every answerable code + chunk
  excerpt verified verbatim against `data/processed/ssoc_chunks.jsonl`.
- **Cross-version answerable check:** during category-3 drafting, found that the report's
  Chapter 4 ("Comparison with SSOC 2020") DOES contain partial cross-version info — the
  2020/2024 count table (Section 4.2), explicit lists of new 2024 codes (4.5, 4.6), one
  reclassification example (4.10). Redrafted obsolete-version questions to avoid items
  the corpus could partially answer.
- **Hard-tier paraphrase calibration:** initial pass deeply paraphrased answerable hard
  questions (dropping the title keyword entirely). User pulled back to lighter paraphrases
  that retain one title keyword. Trade-off documented: more natural-sounding questions
  but less discriminative pressure on retrieval. Defensible for a prototype-scale eval.
- **Tests:** `tests/test_benchmark.py` (11 tests) guards schema, prefix-category
  consistency, answerable→ground-truth presence, type coercion (Excel writes "25121" as
  float `25121.0`; loader normalises back to string).

### Phase 5 — Run RAG on benchmark (180 responses, single self-contained xlsx)

`src/eval/run_eval.py` orchestrates 60 questions × 3 configs (neutral / forced / strict)
= 180 calls to `src.rag.answer.answer`. Results land in `results/responses.xlsx`, a
single 60-row × 16-column sheet with the question, ground truth, retrieved context, and
all three model responses side-by-side. xlsx is the working file AND the resume marker —
already-filled response cells are skipped on restart so an interrupted run does not burn
API calls.

- **Output schema deliberately flat.** Originally drafted with three sheets (responses,
  run_metadata, prompts). User trimmed to one sheet — model versions, temperature, and
  prompt provenance live in PROCESS.md and `src/rag/prompts.py`; no need to duplicate
  into the xlsx.
- **Temperature 0.0** retained throughout (CLAUDE.md constraint; the FORCED prompt does
  the work raising temp would otherwise do — verified in Phase 3.5/3.6).
- **Prompts left as-is** for this run (no optimisation before the benchmark eval). The
  Phase 3.5 checkpoint already showed sharp 3-way contrast; iterating prompts after
  seeing eval results is tuning on the test set. The escape valve if results look broken
  is a 30 dev / 30 test split with prompt iteration on dev only — Phase 6+ decision, not
  taken pre-emptively.
- **Naming-collision repeat.** Same Python footgun as `src.rag.answer`: re-exporting a
  function with the same name as its submodule shadows the submodule. Resolved by not
  re-exporting `run_eval` from `src/eval/__init__.py`; callers use
  `from src.eval.run_eval import run_eval`. Comment in `__init__.py` documents why.

**Run summary:**
- 60 questions × 3 configs = 180 cells (177 processed in the full run; 3 strict cells
  pre-filled by the smoke test were skipped)
- **0 errors**
- Elapsed: 14.5 min (the rate-limit pacer hit the 95-text/min embedding window a few
  times)
- Cost: ~$0.02 on Gemini 2.5 Flash

**Spot-check findings:**

1. **fab-008 (Tell me about SSOC 25144 — nonexistent code):** textbook 3-way contrast.
   `neutral` refused naturally; `strict` refused per the rule; `forced` confidently
   confabulated *"SSOC 25144 refers to a Data Engineer"* with a fabricated definition.
   This is exactly the failure mode the benchmark was designed to elicit.

2. **ans-008 — benchmark quality finding (ambiguous ground truth).** Question: *"What's
   the code for someone who operates trains?"* with GT 31596 (Train operations officer).
   All three configs retrieved BOTH 31596 AND 83110 (Train operator). Strict cited both
   codes; neutral and forced cited 83110 (Train operator) as the primary answer.

   Root cause: the corpus has TWO codes whose definitions match "operates trains":
   - 31596 *Train operations officer*: definition opens with "operates trains and
     provides passenger transport services..."
   - 83110 *Train operator*: title is literally "Train operator"

   Both are defensible answers to the question. The benchmark's GT of 31596 is not
   uniquely correct.

   **Decision:** leave the benchmark unchanged (do not retrofit the GT after seeing
   model output — discipline). Flag for the judge: the judge prompt should treat
   semantically-equivalent codes from the corpus as CORRECT_ANSWER even if not the exact
   GT, with `retrieval_provided_answer=true`. This is exactly the kind of edge case the
   κ-validation hand-labelling set should include.

   Lesson for future benchmark curation: when picking answerable codes, grep the chunks
   for the question's *task description* (not just the title) to surface sibling codes
   that could match.

**Tests:** 8 new tests in `tests/test_run_eval.py` (resume, error handling, rebuild,
limit, config subset). All green. Total suite: 69 passing.

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
