# Benchmark — 60 hand-curated questions

This directory holds the evaluation benchmark for KnowOrNoBench: **60 questions across
5 categories**, designed to measure how reliably a RAG over the Singapore Standard
Occupational Classification (SSOC) 2024 abstains when asked questions its corpus cannot
answer.

`questions.xlsx` is the **single source of truth**. The pipeline reads it via pandas
(`src/eval/benchmark_loader.py`) at eval time. There is no parallel JSONL — the Excel
is canonical so a human can edit it directly during curation without touching code.

## Counts

| Category                    | Tag  | n   |
|---                          |---   |---  |
| Answerable (control)        | —    | 20  |
| SSIC confusion              | SG   | 10  |
| Obsolete version            | SG   | 10  |
| Beyond-corpus attribute     | —    | 10  |
| False / fabricated premise  | —    | 10  |
| **Total**                   |      | **60** |

Difficulty target per category: roughly 1/3 easy + 1/3 medium + 1/3 hard.

## Why this taxonomy

The four unanswerable categories each correspond to a documented RAG failure mode. The
anchor citation is:

> Niu, Cheng, Yuanhao Wu, et al. **"RAGTruth: A Hallucination Corpus for Developing
> Trustworthy Retrieval-Augmented Language Models."** ACL 2024.

RAGTruth labelled 18,000+ RAG responses and identified four primary hallucination
types: *evidence-conflicting*, *subjective / unsupported*, *out-of-context*, and
*baseless information*. Our categories map directly to these failure types, plus an
answerable control to keep the abstention metric honest.

| Category | RAGTruth failure mode | Why we include it |
|---|---|---|
| **SSIC confusion** | Out-of-context | Adjacent-taxonomy queries (wrong-knowledge-base questions) are documented as one of the highest-frequency RAG failures when corpora are domain-narrow. SSIC↔SSOC is a natural example. |
| **Obsolete version** | Out-of-context (temporal) | Temporal-staleness failures are flagged in RAGTruth and central to FreshQA (Vu et al. 2023) — RAG corpora are version-locked, queries aren't. |
| **Beyond-corpus attribute** | Subjective / unsupported | RAGTruth's most-frequent hallucination class: model adds plausible-sounding facts not present in retrieval. We mirror this with attribute queries the corpus has no schema for. |
| **False / fabricated premise** | Baseless info + evidence-conflicting | RAGTruth shows specificity in the question primes the model to play along — a fabricated 5-digit code or false claim about a real code is precisely this trigger. |
| **Answerable (control)** | — | Over-refusal anchor; required to keep the abstention metric honest. |

## Per-category rules

### 1. Answerable (n=20) — `id` prefix `ans-`

**Tests:** over-refusal rate (the honesty counterweight on abstention metrics).

**Rule:** the question has a direct, defensible answer in the SSOC 2024 corpus. Either
- a 4- or 5-digit code can be cited from the Excel/PDF source, OR
- the answer is structural information present in the report (SSOC structure, hierarchy
  rules, 2020-vs-2024 differences).

**Difficulty bands:**

- **Easy:** common occupations with unambiguous direct lookup.
- **Medium:** less common occupations but clearly in corpus; question phrased as a lookup.
- **Hard:** paraphrased descriptions; 4-digit unit-group questions; questions phrased by
  task description rather than title.

**Edge case:** if a question *could* be answered from outside knowledge but the corpus
itself supports the answer, it counts as answerable. The judge will decide whether the
*response* used the context or not.

---

### 2. SSIC confusion (n=10) — `id` prefix `ssic-`

**Tests:** out-of-context handling when the query references an adjacent classification
system (SSIC = industries; SSOC = occupations). Both are 5-digit Singapore-specific codes
and easily confused.

**Rule:** the question asks about SSIC, asks for an industry-classification code, or asks
a question whose answer would naturally come from SSIC. All are unanswerable from the
SSOC 2024 corpus. Real SSIC codes (verifiable via SingStat) are used where applicable.

**Difficulty bands:**

- **Easy:** explicit "SSIC" keyword in an occupational-sounding query.
- **Medium:** ambiguous wording that implies industry classification.
- **Hard:** the question doesn't mention SSIC but is clearly asking about an industry,
  not an occupation.

**Edge case:** if a question mentions both SSIC and SSOC and asks about the relationship,
it is still unanswerable from this corpus (which carries neither correspondence table).

---

### 3. Obsolete version (n=10) — `id` prefix `obs-`

**Tests:** temporal grounding. The corpus is SSOC 2024 only.

**Rule:** the question asks about a pre-2024 SSOC version (2010, 2015, 2020), version
comparison, or pre-2024 code lookup. All are unanswerable.

**Provenance disclaimer:** Category 3 questions reference plausible-pattern obsolete
codes that LOOK like real pre-2024 entries but are not validated against historical
SingStat publications. This is a deliberate choice — the goal is to test *temporal
grounding behaviour*, not to verify historical mappings. A response that confidently
asserts a pre-2024 code mapping is a hallucination regardless of whether the cited
number happened to be a real pre-2024 code, because the corpus contains only SSOC 2024
and the model cannot legitimately know.

**Difficulty bands:**

- **Easy:** explicit pre-2024 year mentioned.
- **Medium:** cross-version comparison questions.
- **Hard:** asks about a recent occupation in an older system, where the model might
  naively answer with a plausible-looking number.

---

### 4. Beyond-corpus attribute (n=10) — `id` prefix `bca-`

**Tests:** scope awareness. The corpus has SSOC codes, titles, definitions, tasks, and
examples. It has NO salary data, demographics, geography, employer info, headcount, or
historical metadata.

**Rule:** the question asks about an attribute that is real and well-defined but is not
in the corpus schema. All are unanswerable.

**Difficulty bands:**

- **Easy:** salary or headcount queries on real codes.
- **Medium:** demographic or industry-employer queries.
- **Hard:** scope questions that look like they could be answerable from the corpus
  structure but aren't (e.g. code creation date).

---

### 5. False / fabricated premise (n=10) — `id` prefix `fab-`

**Tests:** confabulation under specificity. Specificity in the question primes the model
to play along.

**Rule:** the question contains a false premise. Either
- **Subtype A — nonexistent code:** the cited code does not exist in SSOC 2024.
- **Subtype B — false claim about a real code:** the code is real but the claim attached
  to it is wrong.

All are unanswerable; the expected behaviour is to identify the false premise and abstain.

**Difficulty bands:**

- **Easy:** obvious nonexistent codes or wildly false claims.
- **Medium:** subtle false claims about real codes.
- **Hard:** nonexistent codes that look like real codes (similar number patterns), or
  false claims where the false detail is plausible-sounding.

## Read-only discipline

Once curation is signed off, `questions.xlsx` is treated as **read-only**. Eval runs
read from it but never write back. The defence against "did you cherry-pick?" is that
the benchmark was finalised before any RAG-on-benchmark output existed in this repo.

## Schema

See `src/eval/benchmark_loader.py` for the column schema enforced at load time. Tests
in `tests/test_benchmark.py` guard counts, prefix-category consistency, and
answerable→ground-truth presence.
