"""Integrity tests for the chunking stage (src/ingest.py).

These guard the properties the rest of the pipeline and the eval rely on: the right rows are
chunked, ids are unique, the load-bearing 'Examples Classified Elsewhere' text survives, chunks
fit the embedding limit, and ingest is deterministic.
"""

from __future__ import annotations

import json

import pytest

from src import config, ingest


@pytest.fixture(scope="module")
def excel_chunks():
    return ingest.parse_excel()


@pytest.fixture(scope="module")
def report_chunks():
    return ingest.parse_report()


@pytest.fixture(scope="module")
def all_chunks():
    return ingest.ingest_all()


# --- Excel ------------------------------------------------------------------
def test_excel_counts(excel_chunks):
    n4 = sum(1 for c in excel_chunks if c["meta"]["level"] == 4)
    n5 = sum(1 for c in excel_chunks if c["meta"]["level"] == 5)
    assert n4 == 414
    assert n5 == 1006
    assert len(excel_chunks) == 1420


def test_only_four_and_five_digit(excel_chunks):
    assert all(len(c["code"]) in config.EXCEL_CODE_LENGTHS for c in excel_chunks)


def test_known_codes_present(excel_chunks):
    codes = {c["code"] for c in excel_chunks}
    for code in ("11110", "25121", "74110", "1111", "7212"):
        assert code in codes


def test_definition_always_present(excel_chunks):
    assert all("Definition:" in c["text"] for c in excel_chunks)


def test_ex_else_preserved(excel_chunks):
    """The load-bearing false-premise signal must survive into the chunk text."""
    sw = next(c for c in excel_chunks if c["code"] == "25121")
    assert "Examples of jobs classified elsewhere:" in sw["text"]
    assert "Computer engineer" in sw["text"]


def test_tasks_present_for_four_digit(excel_chunks):
    leg = next(c for c in excel_chunks if c["code"] == "1111")
    assert "Tasks:" in leg["text"]
    assert "Occupations under this group:" in leg["text"]


def test_notes_dropped(excel_chunks):
    """Notes were intentionally excluded; code 1111 has a Notes cell but it must not appear."""
    leg = next(c for c in excel_chunks if c["code"] == "1111")
    assert "Notes:" not in leg["text"]


# --- Report -----------------------------------------------------------------
def test_report_nonempty(report_chunks):
    assert len(report_chunks) > 30
    assert all(c["id"].startswith("report-") for c in report_chunks)
    assert all(c["source"] == "ssoc_report" for c in report_chunks)


def test_report_section_names_clean(report_chunks):
    """Section labels must be real headings, not polluted by table rows (no trailing
    run of numbers)."""
    import re
    sections = {c["meta"]["section"] for c in report_chunks}
    assert sections <= {
        "Introduction", "Structure of SSOC", "Classification Principles",
        "Comparison with SSOC 2020", "Linking SSOC with SFw2",
        "How to Determine the Appropriate Occupational Code for an Occupation",
    }
    assert all(not re.search(r"(?:\s+\d+){2,}\s*$", s) for s in sections)


def test_report_first_paragraph(report_chunks):
    first = next(c for c in report_chunks if c["meta"]["para"] == "1.1")
    assert "Singapore Standard Occupational Classification" in first["text"]


def test_no_header_footer_leak(report_chunks):
    """The running header/footer and private-use glyphs must be stripped."""
    import re
    footer = re.compile(
        r"Singapore Department of Statistics\s+Singapore Standard Occupational Classification 2024")
    glyph = re.compile(r"[-]")
    assert all(not footer.search(c["text"]) for c in report_chunks)
    assert all(not glyph.search(c["text"]) for c in report_chunks)


def test_compound_hyphens_preserved(report_chunks):
    """De-hyphenation must keep real compounds intact (not 'fivedigit')."""
    assert all("fivedigit" not in c["text"] for c in report_chunks)
    r210 = next(c for c in report_chunks if c["meta"]["para"] == "2.10")
    assert "five-digit" in r210["text"]


# --- Whole corpus -----------------------------------------------------------
def test_unique_ids_and_nonempty_text(all_chunks):
    ids = [c["id"] for c in all_chunks]
    assert len(ids) == len(set(ids))
    assert all(c["text"].strip() for c in all_chunks)
    assert all({"id", "source", "code", "title", "text", "meta"} <= c.keys() for c in all_chunks)


def test_within_token_limit(all_chunks):
    assert all(ingest._est_tokens(c["text"]) <= config.EMBED_TOKEN_LIMIT for c in all_chunks)


def test_idempotent():
    """Two runs produce byte-identical JSONL ordering/content."""
    a = ingest.ingest_all()
    b = ingest.ingest_all()
    assert [json.dumps(x, sort_keys=True) for x in a] == [json.dumps(x, sort_keys=True) for x in b]