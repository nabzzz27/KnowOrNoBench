"""Ingest the two SSOC sources into a single list of chunks (one JSONL artifact).

Two sources, two chunking strategies:
  * Excel  -> one chunk per 4-digit unit group and 5-digit occupation. Chunk text carries
             Definition + Tasks + Groups(child-list) + Examples-under + Examples-elsewhere
             (whichever are populated). `Examples Classified Elsewhere` is kept deliberately:
             it is the load-bearing signal for the false-premise abstention category.
  * Report -> report PDF pages 8-23, one chunk per numbered paragraph (1.1, 2.5, ...), each
             tagged with its section heading. Pages are clean digital text (no OCR).

This phase makes NO API calls — chunking is purely local. Run as a module to write the JSONL
plus a human-readable preview of every chunk:

    python -m src.ingest              # write JSONL + chunks_preview.txt, print stats
    python -m src.ingest --sample 5   # also print 5 chunks to the terminal
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd
import pdfplumber

from src import config

# Raw Excel columns, in sheet order, mapped to short keys.
EXCEL_COLS = ["code", "title", "groups", "definition", "tasks", "notes", "ex_under", "ex_else"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _clean(val) -> str | None:
    """Return a stripped string, or None for empty / NaN cells."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def _est_tokens(text: str) -> int:
    """Rough token estimate (chars / 4) — only used for the 2,048-limit sanity check."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Excel chunking
# ---------------------------------------------------------------------------
def build_excel_chunk_text(row: dict) -> str:
    """Structure-aware chunk text. Each labelled section is emitted only when its column is
    populated, so 4-digit rows naturally render Tasks/Groups and 5-digit rows render the
    Examples columns."""
    parts = [f"SSOC Code: {row['code']}", f"Title: {row['title']}"]

    def add(label: str, key: str):
        val = row.get(key)
        if val:
            parts.append(f"\n{label}:\n{val}")

    add("Definition", "definition")
    add("Tasks", "tasks")
    add("Occupations under this group", "groups")
    add("Examples of jobs classified under this code", "ex_under")
    add("Examples of jobs classified elsewhere", "ex_else")
    return "\n".join(parts)


def parse_excel() -> list[dict]:
    """One chunk per 4-digit and 5-digit SSOC code, sorted by code (idempotent)."""
    df = pd.read_excel(config.EXCEL_PATH, sheet_name=0, header=config.EXCEL_HEADER_ROW, dtype=str)
    if len(df.columns) != len(EXCEL_COLS):
        raise ValueError(f"expected {len(EXCEL_COLS)} columns, got {list(df.columns)}")
    df.columns = EXCEL_COLS

    chunks: list[dict] = []
    for _, raw in df.iterrows():
        code = _clean(raw["code"])
        if code is None or len(code) not in config.EXCEL_CODE_LENGTHS:
            continue
        row = {k: _clean(raw[k]) for k in EXCEL_COLS}
        row["code"] = code
        text = build_excel_chunk_text(row)
        chunks.append({
            "id": code,
            "source": "ssoc_excel",
            "code": code,
            "title": row["title"] or "",
            "text": text,
            "meta": {"level": len(code)},
        })
    chunks.sort(key=lambda c: c["code"])
    return chunks


# ---------------------------------------------------------------------------
# Report (PDF) chunking
# ---------------------------------------------------------------------------
_PARA_RE = re.compile(r"(?m)^(\d{1,2}\.\d{1,2})\s")          # numbered paragraph, e.g. "2.5 "
_HEAD_RE = re.compile(r"(?m)^([1-9])\s+([A-Z][^\n]{2,80})$")   # section heading, e.g. "1 Introduction"
_TABLE_TAIL = re.compile(r"(?:\s+\d+){2,}\s*$")               # rejects table rows ("... 5 12 24 65")

# Running header/footer that pdfplumber pulls into the body text. Matched as the combined
# phrase so the legitimate "Singapore Department of Statistics (DOS)..." in section 1.2 is kept.
_FOOTER_RE = re.compile(
    r"Singapore Department of Statistics\s+Singapore Standard Occupational Classification 2024")
_GLYPH_RE = re.compile(r"[-]")                   # Unicode private-use glyphs (logo char)


def _extract_report_text() -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate the report's pages 8-23 into one stream so cross-page paragraphs stay
    intact. Strips the running header/footer line and private-use glyphs per page first.
    Returns the full text and per-page (start, end, human_page) char spans for locating which
    page a paragraph starts on."""
    first, last = config.REPORT_PAGES
    full = ""
    spans: list[tuple[int, int, int]] = []
    with pdfplumber.open(config.PDF_PATH) as pdf:
        for human_page in range(first, last + 1):
            text = _GLYPH_RE.sub("", pdf.pages[human_page - 1].extract_text() or "")
            text = "\n".join(ln for ln in text.split("\n") if not _FOOTER_RE.search(ln))
            start = len(full)
            full += text + "\n"
            spans.append((start, len(full), human_page))
    return full, spans


def _page_at(offset: int, spans: list[tuple[int, int, int]]) -> int:
    for start, end, page in spans:
        if start <= offset < end:
            return page
    return spans[-1][2]


def _normalise(text: str) -> str:
    """Join wrapped lines and collapse whitespace. Line-end hyphens are kept (they are real
    compounds in this document, e.g. 'five-\\ndigit' -> 'five-digit', not soft wraps)."""
    text = re.sub(r"-\s*\n\s*", "-", text)   # keep the hyphen when joining a wrapped line
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_report() -> list[dict]:
    """One chunk per numbered paragraph across report pages 8-23, tagged with its section.

    Each paragraph's section is derived from its own number (e.g. "2.13" -> section 2), which
    is robust against table rows that look like headings. Section *names* come from genuine
    heading lines, keyed by their leading integer; candidates ending in a run of numbers (i.e.
    rows of the major-groups table) are rejected."""
    full, spans = _extract_report_text()
    paras = list(_PARA_RE.finditer(full))

    # First paragraph position for each section number (paras are in document order).
    first_para_pos: dict[int, int] = {}
    for m in paras:
        first_para_pos.setdefault(int(m.group(1).split(".")[0]), m.start())

    # A section's name is the genuine heading "N Title" closest *before* its first paragraph
    # (major-group table rows elsewhere are thereby excluded). Table-row look-alikes that end
    # in a run of numbers are also rejected.
    heads = [(m.start(), int(m.group(1)), m.group(2).strip()) for m in _HEAD_RE.finditer(full)]
    section_names: dict[int, str] = {}
    for num, fp in first_para_pos.items():
        cands = [(pos, title) for pos, n, title in heads
                 if n == num and pos <= fp and not _TABLE_TAIL.search(title)]
        if cands:
            section_names[num] = max(cands, key=lambda c: c[0])[1]

    chunks: list[dict] = []
    for i, m in enumerate(paras):
        para = m.group(1)                       # e.g. "2.13"
        sec_num = int(para.split(".")[0])
        section = section_names.get(sec_num, f"Section {sec_num}")
        start = m.start()
        end = paras[i + 1].start() if i + 1 < len(paras) else len(full)
        body = _normalise(full[start:end])
        if len(body) < 20:                      # skip stray marker matches with no real content
            continue
        chunks.append({
            "id": f"report-{para}",
            "source": "ssoc_report",
            "code": f"report-{para}",
            "title": f"SSOC 2024 Report — §{section}",
            "text": f"SSOC 2024 Report — §{section}\n{body}",
            "meta": {"page": _page_at(start, spans), "para": para, "section": section},
        })
    return chunks


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def ingest_all() -> list[dict]:
    """Build all chunks from both sources (Excel first, then report, in document order)."""
    return parse_excel() + parse_report()


def write_jsonl(chunks: list[dict]) -> None:
    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def write_preview(chunks: list[dict]) -> None:
    """Human-readable dump of every chunk, for inspection before embedding."""
    lines = [f"# Chunk preview — {len(chunks)} chunks\n"]
    for c in chunks:
        lines.append("=" * 90)
        lines.append(f"id={c['id']}  source={c['source']}  ~tokens={_est_tokens(c['text'])}  "
                     f"meta={c['meta']}")
        lines.append("-" * 90)
        lines.append(c["text"])
        lines.append("")
    config.CHUNKS_PREVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def _print_stats(chunks: list[dict]) -> None:
    excel = [c for c in chunks if c["source"] == "ssoc_excel"]
    report = [c for c in chunks if c["source"] == "ssoc_report"]
    n4 = sum(1 for c in excel if c["meta"]["level"] == 4)
    n5 = sum(1 for c in excel if c["meta"]["level"] == 5)
    toks = sorted(_est_tokens(c["text"]) for c in chunks)
    over = [c["id"] for c in chunks if _est_tokens(c["text"]) > config.EMBED_TOKEN_LIMIT]
    print(f"total chunks:        {len(chunks)}")
    print(f"  excel:             {len(excel)}  (4-digit={n4}, 5-digit={n5})")
    print(f"  report:            {len(report)}")
    print(f"est. tokens min/median/max: {toks[0]} / {toks[len(toks)//2]} / {toks[-1]}")
    print(f"chunks over {config.EMBED_TOKEN_LIMIT}-token limit: {over if over else 'none'}")
    print(f"wrote {config.CHUNKS_PATH.relative_to(config.REPO_ROOT)}")
    print(f"wrote {config.CHUNKS_PREVIEW_PATH.relative_to(config.REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="Build SSOC chunks from Excel + report PDF.")
    ap.add_argument("--sample", type=int, default=0, help="also print N chunks to the terminal")
    args = ap.parse_args()

    chunks = ingest_all()
    write_jsonl(chunks)
    write_preview(chunks)
    _print_stats(chunks)

    if args.sample:
        print("\n" + "#" * 90 + f"\n# SAMPLE OF {args.sample} CHUNKS\n" + "#" * 90)
        step = max(1, len(chunks) // args.sample)
        for c in chunks[::step][:args.sample]:
            print("\n" + "=" * 90)
            print(f"id={c['id']}  source={c['source']}  meta={c['meta']}")
            print("-" * 90)
            print(c["text"])


if __name__ == "__main__":
    main()
