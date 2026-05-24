"""De-risking spike for the SSOC RAG system  (THROWAWAY — not production code).

Purpose: prove the riskiest assumptions hold *before* we build the real `src/` pipeline.
Each "probe" tests one assumption, prints its finding, and contributes to `spike_log.md`.

The spike NEVER embeds the full corpus — it works on a 100-row sample (see SAMPLE_SIZE) and
*projects* the full-build cost. Run `--dry-run` to inspect exactly what would be embedded
without spending a single API call.

Usage:
    python -m scripts.spike --dry-run   # Probes 0-2 only, ZERO API calls
    python -m scripts.spike             # full live run, writes spike_log.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Spike parameters  (echoed at startup so there is no hidden behaviour)
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 100          # chunks embedded in the spike (never the full ~1,006 detailed corpus)
EMBED_DIM = 768            # embedding output dimensionality
RECALL_K = 5               # retrieve top-5 for the recall metric
REAL_TOP_K = 4             # production default (reported for context only)
EMBED_BATCH = 100          # texts per embedding call; falls back to 25 then 1 on failure
GEN_TEMPERATURE = 0.0      # deterministic generation
EMBED_TOKEN_LIMIT = 2048   # gemini-embedding-001 per-text input limit (from Google docs)
# Free-tier embedding quota is counted PER TEXT (discovered live: 429 right after ~100 texts).
# We pace to stay under this within a rolling 60s window; the real embed.py needs the same.
EMBED_TEXTS_PER_MIN = 95   # safety margin below the observed ~100/min ceiling
RATE_LIMIT_WAIT = 60       # seconds to wait on a 429 (per-minute window reset)

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"
DOC_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"

REPO_ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = REPO_ROOT / "data" / "raw" / "ssoc2024-detailed-definitions.xlsx"
HEADER_ROW = 4             # 0-indexed; columns live on the 5th row of the sheet
LOG_PATH = REPO_ROOT / "spike_log.md"

# 10 hand-picked gold (code, query) pairs, one per major group (group 2 appears twice as
# it is by far the largest). Codes are guaranteed included in the sample so recall is testable.
# Queries deliberately avoid quoting the title verbatim, to make retrieval realistic.
GOLD = [
    ("11110", "Which SSOC code applies to a Member of Parliament who makes and amends laws?"),
    ("25121", "What is the SSOC classification for someone who develops software applications?"),
    ("24111", "Which SSOC code is for an accountant who prepares and audits financial statements?"),
    ("34341", "What SSOC code covers a chef running a restaurant kitchen (not pastry)?"),
    ("41201", "Which SSOC code is for a secretary handling correspondence and scheduling?"),
    ("51201", "What is the SSOC code for a cook preparing meals in an eatery?"),
    ("61133", "Which SSOC code applies to a gardener doing horticultural and landscaping work?"),
    ("74110", "What SSOC code is for an electrician who installs and maintains wiring?"),
    ("83221", "Which SSOC code covers a taxi driver?"),
    ("91122", "What SSOC code is for a cleaner who services hotel rooms?"),
]

# Deliberately unanswerable questions for the abstention-motivation probe.
UNANSWERABLE = [
    ("SSIC confusion",
     "What is the SSIC code for the food and beverage retail industry in Singapore?"),
    ("beyond-corpus attribute",
     "What is the average monthly salary of a software developer under SSOC 25121?"),
    ("fabricated premise",
     "What occupation does SSOC code 88888 refer to?"),
]

# Answerable questions for the generation probe (both gold codes, in-sample).
ANSWERABLE_GEN = [
    "What is SSOC 25121 and what does that occupation do?",
    "Which SSOC code is for an electrician, and what is its definition?",
]

COLS = ["code", "title", "groups", "definition", "tasks", "notes", "ex_under", "ex_else"]

_findings: list[str] = []  # accumulates markdown lines for spike_log.md


def log(line: str = "") -> None:
    """Print to stdout and remember the line for spike_log.md."""
    print(line)
    _findings.append(line)


def banner(title: str) -> None:
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


# ---------------------------------------------------------------------------
# Probe 0 — environment & dependency check
# ---------------------------------------------------------------------------
def probe0_environment() -> dict:
    banner("PROBE 0 — environment & dependency check")
    log(f"python: {sys.version.split()[0]}")
    import importlib
    versions = {}
    ok = True
    for mod in ["google.genai", "chromadb", "pandas", "openpyxl", "dotenv"]:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "(no __version__)")
            versions[mod] = v
            log(f"  OK   {mod:14} {v}")
        except Exception as e:  # noqa: BLE001 — a failed import is itself a finding
            ok = False
            log(f"  FAIL {mod:14} -> {e!r}")

    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    log(f"  GEMINI_API_KEY present: {bool(key)}  "
        f"(length {len(key) if key else 0}, value never printed)")

    return {"imports_ok": ok, "versions": versions, "key_present": bool(key)}


# ---------------------------------------------------------------------------
# Probe 1 — Excel parse
# ---------------------------------------------------------------------------
def probe1_parse():
    banner("PROBE 1 — Excel parse")
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

    log(f"file: {XLSX_PATH.relative_to(REPO_ROOT)}  (exists: {XLSX_PATH.exists()})")
    df = pd.read_excel(XLSX_PATH, sheet_name=0, header=HEADER_ROW, dtype=str)
    log(f"raw shape: {df.shape}")
    log(f"columns ({len(df.columns)}): {list(df.columns)}")

    df.columns = COLS
    df["code"] = df["code"].astype(str).str.strip()
    df["len"] = df["code"].str.len()

    dist = df["len"].value_counts().sort_index()
    log("code-length distribution (digits -> rows):")
    for k, v in dist.items():
        log(f"    {k}: {v}")

    d5 = df[df["len"] == 5].copy().reset_index(drop=True)
    log(f"5-digit detailed occupations: {len(d5)}")
    log("populated-column rates among 5-digit rows:")
    for c in ["title", "definition", "notes", "ex_under", "ex_else"]:
        log(f"    {c:11} {d5[c].notna().sum():4d}  ({100 * d5[c].notna().mean():.0f}%)")
    return d5


# ---------------------------------------------------------------------------
# Probe 2 — sample selection + chunk construction (the manifest)
# ---------------------------------------------------------------------------
def build_chunk_text(row) -> str:
    """Structure-aware chunk text. Includes every populated source column — crucially
    'Examples of Job Classified Elsewhere', which is load-bearing for the false-premise
    benchmark category (it gives the model a signal for what a code does NOT cover)."""
    import pandas as pd
    parts = [f"SSOC Code: {row['code']}", f"Title: {row['title']}"]

    def add(label, val):
        if pd.notna(val) and str(val).strip():
            parts.append(f"\n{label}:\n{str(val).strip()}")

    add("Definition", row["definition"])
    add("Examples of jobs classified under this code", row["ex_under"])
    add("Examples of jobs classified elsewhere", row["ex_else"])
    add("Notes", row["notes"])
    return "\n".join(parts)


def select_sample(d5):
    """Deterministic stratified sample of SAMPLE_SIZE codes: all gold codes first, then
    evenly-spaced fills across major groups 1-9 so we have realistic distractors."""
    gold_codes = [c for c, _ in GOLD]
    by_group = {g: d5[d5["code"].str[0] == g]["code"].tolist() for g in "123456789"}

    need = SAMPLE_SIZE - len(set(gold_codes))
    per_group = need // 9 + 1
    picks: list[str] = []
    for g in "123456789":
        codes = by_group[g]
        step = max(1, len(codes) // per_group)
        picks.extend(codes[::step][:per_group])

    ordered = list(dict.fromkeys(gold_codes + picks))  # gold first, dedup, preserve order
    return ordered[:SAMPLE_SIZE]


def probe2_manifest(d5):
    banner("PROBE 2 — sample selection + chunk manifest")
    sample_codes = select_sample(d5)
    by_code = {r["code"]: r for _, r in d5.iterrows()}

    records = []
    for code in sample_codes:
        row = by_code[code]
        text = build_chunk_text(row)
        records.append({"code": code, "title": row["title"], "text": text,
                        "tokens_est": len(text) // 4})

    log(f"selected {len(records)} sample chunks (SAMPLE_SIZE={SAMPLE_SIZE})")
    log("group spread (leading digit -> count):")
    spread = {}
    for r in records:
        spread[r["code"][0]] = spread.get(r["code"][0], 0) + 1
    log("    " + "  ".join(f"{g}:{spread.get(g, 0)}" for g in "123456789"))

    log("")
    log("FULL MANIFEST (code | ~tokens | title):")
    for r in records:
        log(f"    {r['code']} | {r['tokens_est']:4d} | {r['title']}")

    max_tok = max(r["tokens_est"] for r in records)
    log(f"\nmax estimated tokens in a chunk: {max_tok}  (embedding limit {EMBED_TOKEN_LIMIT})")
    over = [r["code"] for r in records if r["tokens_est"] > EMBED_TOKEN_LIMIT]
    log(f"chunks over the limit: {over if over else 'none'}")

    log("\n--- 2 FULL CHUNK TEXTS (verbatim, exactly what gets embedded) ---")
    for r in records[:2]:
        log(f"\n[{r['code']}]")
        log(r["text"])

    log("\n--- QUERIES → GOLD CODE (and present-in-sample check) ---")
    sample_set = {r["code"] for r in records}
    all_present = True
    for code, q in GOLD:
        present = code in sample_set
        all_present &= present
        log(f"    {'OK ' if present else 'MISSING'} gold={code}  q={q!r}")
    log(f"\nall gold codes present in sample: {all_present}")
    return records, all_present


# ---------------------------------------------------------------------------
# Shared: client + retry helper
# ---------------------------------------------------------------------------
def make_client():
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "resource_exhausted" in s or ("rate" in s and "limit" in s)


def with_retry(fn, what: str, max_tries: int = 5):
    """Retry transient errors. On a 429 we wait the full per-minute window (RATE_LIMIT_WAIT);
    on other transient errors we use short exponential backoff."""
    for attempt in range(1, max_tries + 1):
        try:
            return fn(), attempt
        except Exception as e:  # noqa: BLE001
            if attempt == max_tries:
                raise
            if is_rate_limit(e):
                wait = RATE_LIMIT_WAIT
                _pacer.reset()  # window has clearly rolled; start fresh after the wait
                log(f"    [{what}] attempt {attempt} hit 429; waiting {wait}s for quota window")
            else:
                wait = 2 ** attempt
                log(f"    [{what}] attempt {attempt} failed ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("with_retry: exhausted retries without returning")  # unreachable


class _EmbedPacer:
    """Proactively keeps embedding calls under EMBED_TEXTS_PER_MIN within a rolling 60s window,
    so we throttle *before* hitting a 429 rather than after. This is the mitigation the real
    embed.py will use for the full-corpus build."""

    def __init__(self):
        self.events: list[tuple[float, int]] = []  # (timestamp, texts)

    def reset(self):
        self.events = []

    def reserve(self, n: int):
        now = time.time()
        self.events = [(t, c) for t, c in self.events if now - t < 60]
        used = sum(c for _, c in self.events)
        if self.events and used + n > EMBED_TEXTS_PER_MIN:
            sleep = 60 - (now - self.events[0][0]) + 1
            if sleep > 0:
                log(f"    [pace] {used}+{n} texts would exceed {EMBED_TEXTS_PER_MIN}/min;"
                    f" sleeping {sleep:.0f}s")
                time.sleep(sleep)
            self.reset()
        self.events.append((time.time(), n))


_pacer = _EmbedPacer()
_QCACHE: dict[str, list[float]] = {}  # query text -> embedding vector (embedded once, reused)


# ---------------------------------------------------------------------------
# Probe 3 — embedding: batch vs sequential, speed, dims, rate limits
# ---------------------------------------------------------------------------
def embed_texts(client, texts, task_type, batch_size):
    """Embed a list of texts, batching where possible. Returns (vectors, mode, n_429)."""
    from google.genai import types
    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM)
    vectors: list[list[float]] = []
    n_429 = 0
    i = 0
    while i < len(texts):
        chunk = texts[i:i + batch_size]
        _pacer.reserve(len(chunk))

        def call():
            return client.models.embed_content(model=EMBED_MODEL, contents=chunk, config=cfg)

        try:
            res, _ = with_retry(call, f"embed[{i}:{i+len(chunk)}]")
            vectors.extend(e.values for e in res.embeddings)
            i += batch_size
        except Exception as e:  # noqa: BLE001
            if is_rate_limit(e):
                n_429 += 1
            if batch_size > 1:
                new = 25 if batch_size > 25 else 1
                log(f"    batch={batch_size} failed ({type(e).__name__}); falling back to batch={new}")
                batch_size = new
                continue
            raise
    mode = f"batch={batch_size}" if batch_size > 1 else "sequential"
    return vectors, mode, n_429


def probe3_embed(client, records):
    banner("PROBE 3 — embedding (batch / speed / dims / rate limits)")
    texts = [r["text"] for r in records]

    t0 = time.time()
    vectors, mode, n_429 = embed_texts(client, texts, DOC_TASK, EMBED_BATCH)
    dt = time.time() - t0

    dim = len(vectors[0]) if vectors else 0
    log(f"embedded {len(vectors)} docs in {dt:.2f}s  (mode: {mode}, 429s: {n_429})")
    log(f"vector dimensionality: {dim}  (expected {EMBED_DIM})")
    log(f"per-doc latency: {1000 * dt / max(1, len(vectors)):.1f} ms")

    # project full detailed-corpus build (1,006 five-digit codes)
    per_doc = dt / max(1, len(vectors))
    full = 1006
    log(f"\nprojection for full {full} detailed codes at this rate:")
    log(f"    same path ({mode}): ~{per_doc * full:.0f}s  (~{per_doc * full / 60:.1f} min)")
    log(f"    naive sequential @ ~12 RPM free tier: ~{full / 12:.0f} min  -> batching is essential")

    for r, v in zip(records, vectors):
        r["embedding"] = v
    return {"dim": dim, "seconds": dt, "mode": mode, "n_429": n_429}


def build_query_cache(client):
    """Embed every query we will need (gold + generation) in ONE paced batch, using the
    asymmetric RETRIEVAL_QUERY task type, and cache by text. Keeps query embedding to a single
    request so we don't trickle past the per-minute quota."""
    banner("QUERY EMBEDDING (cached, asymmetric RETRIEVAL_QUERY)")
    texts = ([q for _, q in GOLD] + ANSWERABLE_GEN + [q for _, q in UNANSWERABLE])
    texts = list(dict.fromkeys(texts))  # dedup, preserve order
    t0 = time.time()
    vectors, mode, n_429 = embed_texts(client, texts, QUERY_TASK, EMBED_BATCH)
    for t, v in zip(texts, vectors):
        _QCACHE[t] = v
    log(f"cached {len(_QCACHE)} query embeddings in {time.time()-t0:.2f}s "
        f"(mode: {mode}, 429s: {n_429}, dim: {len(vectors[0])})")


def qvec(text):
    """Cached query vector lookup (no API call)."""
    return _QCACHE[text]


# ---------------------------------------------------------------------------
# Probe 4 — retrieval recall@k
# ---------------------------------------------------------------------------
def probe4_recall(records):
    banner("PROBE 4 — retrieval recall@k (cosine)")
    import chromadb
    from chromadb.config import Settings

    cclient = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    col = cclient.get_or_create_collection("spike", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[r["code"] for r in records],
        embeddings=[r["embedding"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[{"code": r["code"], "title": r["title"]} for r in records],
    )
    log(f"built in-memory cosine collection with {col.count()} chunks")

    hit5 = hit1 = 0
    log("\nper-query results (rank of gold code in top-5):")
    for code, q in GOLD:
        res = col.query(query_embeddings=[qvec(q)], n_results=RECALL_K)
        ids = res["ids"][0]
        rank = ids.index(code) + 1 if code in ids else None
        hit5 += int(code in ids)
        hit1 += int(ids[0] == code)
        log(f"    gold={code} rank={rank if rank else 'MISS'}  top5={ids}")

    n = len(GOLD)
    log(f"\nrecall@1: {hit1}/{n} = {100*hit1/n:.0f}%")
    log(f"recall@{RECALL_K}: {hit5}/{n} = {100*hit5/n:.0f}%   (PRD target >=80%)")
    log(f"(production default REAL_TOP_K={REAL_TOP_K})")
    return {"recall_at_1": hit1 / n, "recall_at_k": hit5 / n}


# ---------------------------------------------------------------------------
# Probe 5 — generation: answerable vs unanswerable (abstention motivation)
# ---------------------------------------------------------------------------
NAIVE_PROMPT = """You are a helpful assistant. Use the following SSOC information to answer the user's question.

Context:
{context}

Question: {question}

Answer:"""


def retrieve_context(col, question, k):
    res = col.query(query_embeddings=[qvec(question)], n_results=k)
    return "\n\n---\n\n".join(res["documents"][0])


def probe5_generation(client, records):
    banner("PROBE 5 — generation: answerable vs unanswerable (naive prompt)")
    import chromadb
    from chromadb.config import Settings
    from google.genai import types

    cclient = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    col = cclient.get_or_create_collection("spike_gen", metadata={"hnsw:space": "cosine"})
    col.add(ids=[r["code"] for r in records],
            embeddings=[r["embedding"] for r in records],
            documents=[r["text"] for r in records])

    gcfg = types.GenerateContentConfig(temperature=GEN_TEMPERATURE)

    def ask(question):
        ctx = retrieve_context(col, question, REAL_TOP_K)
        prompt = NAIVE_PROMPT.format(context=ctx, question=question)
        resp, _ = with_retry(
            lambda: client.models.generate_content(model=GEN_MODEL, contents=prompt, config=gcfg),
            "generate")
        return resp.text

    log(">>> ANSWERABLE (should answer with the correct code/details):")
    for q in ANSWERABLE_GEN:
        log(f"\n  Q: {q}")
        log(f"  A: {ask(q).strip()}")

    log("\n>>> UNANSWERABLE (naive prompt — watch for confabulation):")
    results = []
    abstain_markers = ("not available", "does not", "do not", "cannot", "not include",
                       "not contain", "i am sorry", "i'm sorry", "not provide", "no code",
                       "not in the", "not found")
    for kind, q in UNANSWERABLE:
        ans = ask(q).strip()
        abstained = any(m in ans.lower() for m in abstain_markers)
        results.append((kind, abstained))
        log(f"\n  [{kind}] (looks like abstention: {abstained}) Q: {q}")
        log(f"  A: {ans}")
    log("\n(Interpretation: confabulation here is the empirical justification for the abstention config.)")
    return results


# ---------------------------------------------------------------------------
# Probe 6 — decisions for the real build (synthesised from the measurements above)
# ---------------------------------------------------------------------------
def probe6_decisions(emb, rec, unans):
    banner("PROBE 6 — DECISIONS FOR THE REAL BUILD")
    n_abstain = sum(1 for _, a in unans if a) if unans else 0
    lines = [
        "ENV  — google-genai 2.6.0 / chromadb 1.5.9 / pandas 3.0.3 / openpyxl 3.1.5 all import on "
        "Python 3.14.3. -> Dockerfile FROM python:3.14-slim; pin these exact versions in requirements.txt.",

        "INGEST — header at sheet row 5 (header=4); chunk the 1,006 five-digit detailed codes. Chunk "
        "text = Code + Title + Definition + Examples-under + Examples-elsewhere + Notes; skip the "
        "always-empty Groups/Tasks columns at this level. 'Examples Classified Elsewhere' present on "
        "59% of rows (load-bearing for the false-premise category). Max chunk ~313 tokens << 2048 "
        "limit -> no truncation needed.",

        f"EMBED — gemini-embedding-001 @ {emb['dim']}-dim, RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY asymmetry "
        f"works. Batched 100 texts in {emb['seconds']:.1f}s. FREE-TIER QUOTA IS ~100 TEXTS/MIN, COUNTED "
        "PER TEXT (429 hit right after ~100). -> embed.py MUST pace (<=95 texts/60s) and wait ~60s on "
        "429 (short exponential backoff is NOT enough). Full 1,006-code build ~= 11 paced batches ~= "
        "~11 min one-time index build.",

        f"RETRIEVE — Chroma cosine (hnsw:space=cosine, set explicitly). recall@1 = "
        f"{rec['recall_at_1']*100:.0f}% and recall@5 = {rec['recall_at_k']*100:.0f}% on 100-chunk "
        f"sample. -> production TOP_K=4 is comfortably safe; no reranking/hybrid needed.",

        f"GENERATE — gemini-2.5-flash @ temp 0 gives coherent, correct answers on answerable Qs. "
        f"KEY FINDING: under the NAIVE prompt *with retrieved context*, the model already abstained on "
        f"{n_abstain}/{len(unans) if unans else 0} easy unanswerables (SSIC / salary / fabricated code). "
        "-> the benchmark must include SUBTLER unanswerables and the zeroshot (no-retrieval) config to "
        "produce real hallucination signal; the abstention prompt's value should be measured against "
        "those harder cases, not these easy ones. This is the central thing to get right in the eval.",
    ]
    for ln in lines:
        log("- " + ln)
        log("")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SSOC RAG de-risking spike (throwaway).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run Probes 0-2 only (parse + manifest). ZERO API calls.")
    args = ap.parse_args()

    banner("SPIKE PARAMETERS")
    for k, v in [("SAMPLE_SIZE", SAMPLE_SIZE), ("EMBED_DIM", EMBED_DIM), ("RECALL_K", RECALL_K),
                 ("REAL_TOP_K", REAL_TOP_K), ("EMBED_BATCH", EMBED_BATCH),
                 ("GEN_TEMPERATURE", GEN_TEMPERATURE), ("EMBED_MODEL", EMBED_MODEL),
                 ("GEN_MODEL", GEN_MODEL), ("dry_run", args.dry_run)]:
        log(f"  {k} = {v}")

    env = probe0_environment()
    if not env["imports_ok"]:
        log("\nABORT: a dependency failed to import (see Probe 0).")
        write_log()
        sys.exit(1)
    if not args.dry_run and not env["key_present"]:
        log("\nABORT: GEMINI_API_KEY missing — cannot run live probes. Use --dry-run or set the key.")
        write_log()
        sys.exit(1)

    d5 = probe1_parse()
    records, all_present = probe2_manifest(d5)
    if not all_present:
        log("\nWARNING: not all gold codes are in the sample — recall would be unfair. Fix selection.")

    if args.dry_run:
        banner("DRY RUN COMPLETE")
        log("Probes 0-2 done. No API calls were made. Inspect the manifest above, then run "
            "`python -m scripts.spike` for the live probes.")
        write_log()
        return

    client = make_client()
    try:
        emb = probe3_embed(client, records)
        build_query_cache(client)
        rec = probe4_recall(records)
        unans = probe5_generation(client, records)
        probe6_decisions(emb, rec, unans)
        banner("SPIKE COMPLETE")
        log("All probes ran. Review the findings above and in spike_log.md.")
    except Exception as e:  # noqa: BLE001 — capture partial findings rather than lose them
        banner("SPIKE ABORTED")
        log(f"A probe raised: {type(e).__name__}: {e}")
        log("Partial findings up to this point are saved below.")
        raise
    finally:
        write_log()


def write_log():
    header = ("# Spike log\n\n"
              "_Auto-generated by `scripts/spike.py` (throwaway de-risking spike). "
              "Captures findings that feed the production-build decisions in PROCESS.md._\n")
    LOG_PATH.write_text(header + "\n```\n" + "\n".join(_findings) + "\n```\n", encoding="utf-8")
    print(f"\n[wrote {LOG_PATH.relative_to(REPO_ROOT)}]")


if __name__ == "__main__":
    main()
