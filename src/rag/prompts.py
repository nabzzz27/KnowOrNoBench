"""Frozen prompt templates and context formatter for the three RAG configs.

Each prompt serves a distinct role in the eval methodology:
  - NEUTRAL_PROMPT  — natural baseline behaviour (no abstention rule, no force-answer rule);
                      a covariate for whether the model already abstains without the rule
  - FORCED_PROMPT   — explicitly forbids refusal; generates known-positive hallucinations
                      on unanswerable questions for judge validation
  - STRICT_PROMPT   — must use context, must refuse if not supported (production behaviour);
                      the configuration the project ships and where the headline κ is measured

Once committed, the strings MUST NOT be edited — any change invalidates judge κ measured
against the previous wording. Iterate on a dev split only.
"""

from __future__ import annotations


NEUTRAL_PROMPT = """\
You are an assistant answering questions about the Singapore Standard Occupational Classification 2024 (SSOC 2024).

Use the context below to answer the user's question. Cite the relevant SSOC code(s) when applicable.

Context:
{context}

Question: {question}

Answer:"""


FORCED_PROMPT = """\
You are an assistant answering questions about the Singapore Standard Occupational Classification 2024 (SSOC 2024).

Rules:
1. You MUST provide a concrete answer to the user's question.
2. You are NOT permitted to refuse, abstain, or say "I don't know", "I cannot answer", "the context does not contain that information", or any equivalent.
3. If the context is incomplete, irrelevant, or contradicts the question, use your best judgement and any outside knowledge to commit to a specific answer (e.g. a specific SSOC code) anyway.
4. Always cite a specific SSOC code in your answer.

Context:
{context}

Question: {question}

Answer:"""


STRICT_PROMPT = """\
You are an assistant answering questions about the Singapore Standard Occupational Classification 2024 (SSOC 2024).

Rules:
1. Answer using ONLY the information in the context below. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, reply exactly: "I don't know" — and briefly state what is missing.
3. If the question is based on a false premise (e.g., refers to an SSOC code, occupation, or category that does not appear in the context), reply: "I don't know" and explain that the premise is not supported by the SSOC 2024 source.
4. When you can answer, cite the relevant SSOC code(s) from the context.

Context:
{context}

Question: {question}

Answer:"""


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as the {context} substitution for the three prompts.

    Each chunk becomes one block: a header `[SSOC {code}] {title}` followed by the chunk
    text, separated by `---`. Empty list → empty string (the prompt template handles a
    blank context fine; we don't add hedging text).
    """
    if not chunks:
        return ""
    blocks = []
    for c in chunks:
        meta = c.get("metadata", {})
        code = meta.get("code", c.get("id", ""))
        title = meta.get("title", "")
        blocks.append(f"[SSOC {code}] {title}\n{c['text']}")
    return "\n---\n".join(blocks)
