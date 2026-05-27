"""RAG orchestrator: the single function the eval contracts against.

`answer(question, config)` returns a fully JSON-serialisable record of what happened:
{question, config, retrieved_chunks, prompt, response}. The eval drops this dict straight
into its per-question records.

The three configs serve distinct roles in the eval methodology:
  - neutral: natural baseline — a covariate for whether the model already abstains
             without an explicit rule
  - forced:  explicitly forbids refusal — a known-positive control. On an unanswerable
             question, forced MUST confabulate by construction, giving the judge a
             ground-truth hallucinated row without manual labelling
  - strict:  must use context, must refuse if not supported — the configuration the
             project ships and where the headline judge-κ is measured

All three use identical retrieval, so behavioural differences across configs are caused
by the prompt alone. `strict` is the default for `answer(question)` calls without a
config argument.
"""

from __future__ import annotations

from typing import Callable

from src.rag import prompts
from src.rag.generate import generate as default_generate
from src.rag.retrieve import retrieve as default_retrieve


_VALID_CONFIGS = ("neutral", "forced", "strict")

_TEMPLATES = {
    "neutral": prompts.NEUTRAL_PROMPT,
    "forced": prompts.FORCED_PROMPT,
    "strict": prompts.STRICT_PROMPT,
}


def answer(
    question: str,
    config: str = "strict",
    *,
    retrieve_fn: Callable[[str], list[dict]] | None = None,
    generate_fn: Callable[[str], str] | None = None,
) -> dict:
    """Run one RAG turn and return the record the eval expects.

    Args:
        question: the user/eval question.
        config: one of "neutral" | "forced" | "strict" (default "strict").
        retrieve_fn / generate_fn: test seams; default to the real implementations.

    Returns:
        {"question", "config", "retrieved_chunks", "prompt", "response"}.
    """
    if config not in _VALID_CONFIGS:
        raise ValueError(
            f"unknown config {config!r}; expected one of {_VALID_CONFIGS}")

    r = retrieve_fn if retrieve_fn is not None else default_retrieve
    g = generate_fn if generate_fn is not None else default_generate

    retrieved = r(question)
    prompt = _TEMPLATES[config].format(
        context=prompts.format_context(retrieved),
        question=question,
    )

    response = g(prompt)
    return {
        "question": question,
        "config": config,
        "retrieved_chunks": retrieved,
        "prompt": prompt,
        "response": response,
    }
