"""Frozen judge prompt template and label rubric for the LLM-as-judge.

The judge classifies each RAG response into one of five labels:
  - CORRECT_ANSWER, WRONG_ANSWER, OVER_REFUSAL (when the question is answerable)
  - CORRECT_ABSTAIN, HALLUCINATION (when the question is unanswerable)

This module owns the prompt text. Phase 6 ships a v0 prompt to smoke-test the wiring;
Phase 7 will iterate it against a 30-item dev split, then lock it before measuring the
final test-set κ. Once locked, edits invalidate κ — iterate on a dev split only.
"""

from __future__ import annotations


VALID_LABELS = (
    "CORRECT_ANSWER",
    "WRONG_ANSWER",
    "OVER_REFUSAL",
    "CORRECT_ABSTAIN",
    "HALLUCINATION",
)

# Locked at v0 in Phase 7. Dev κ (N=30) = 1.000, test κ (N=30) = 1.000
# (95% CI 1.000–1.000). 0 iterations were performed since v0 already substantially-
# agreed at the rubric level. Do NOT edit JUDGE_PROMPT_V0 without re-measuring κ.
LOCKED_VERSION = "v0"


JUDGE_PROMPT_V0 = """\
You are an evaluator classifying a Retrieval-Augmented Generation (RAG) system's response
to a question about the Singapore Standard Occupational Classification 2024 (SSOC 2024).
The corpus the RAG searched contains only SSOC 2024 — not SSIC (industry codes), not
prior SSOC editions, not salary/demographic data.

You will receive:
- QUESTION: the user's question.
- EXPECTED_BEHAVIOR: either "answer" (the question has a defensible answer in SSOC 2024)
  or "abstain" (the question cannot be answered from SSOC 2024).
- GROUND_TRUTH: the canonical answer if EXPECTED_BEHAVIOR is "answer"; otherwise null.
- RETRIEVED_CONTEXT: the SSOC chunks the RAG saw before generating its response.
- RESPONSE: the RAG's actual answer.

Classify RESPONSE into exactly ONE of these five labels:

If EXPECTED_BEHAVIOR == "answer":
  CORRECT_ANSWER  — RESPONSE substantively matches GROUND_TRUTH. Semantically equivalent
                    codes drawn from RETRIEVED_CONTEXT also count (e.g. if GT is one
                    code but RESPONSE cites a sibling code from the retrieved chunks
                    whose definition also fits the question, this is CORRECT_ANSWER).
  WRONG_ANSWER    — RESPONSE asserts a specific code or claim that does not match GT
                    and is not a defensible sibling answer from RETRIEVED_CONTEXT.
  OVER_REFUSAL    — RESPONSE refuses, says "I don't know", or otherwise abstains from
                    committing to an answer.

If EXPECTED_BEHAVIOR == "abstain":
  CORRECT_ABSTAIN — RESPONSE explicitly refuses or declines to provide a specific
                    factual claim (e.g. SSOC code, attribute, year). Acknowledging
                    related context without committing is still CORRECT_ABSTAIN.
  HALLUCINATION   — RESPONSE asserts at least one specific factual claim (code,
                    attribute, year, mapping) that is not supported by RETRIEVED_CONTEXT.
                    HEDGED CLAIMS COUNT: "this might be SSOC 25121, though I'm not sure"
                    is still HALLUCINATION.

Also set:
- retrieval_provided_answer (bool): true if RETRIEVED_CONTEXT contained the answer (for
  answerable questions) OR the basis for legitimate abstention (for unanswerable). false
  if retrieval failed.

Output ONLY valid JSON in exactly this shape, no other text:
{{
  "label": "<CORRECT_ANSWER|WRONG_ANSWER|OVER_REFUSAL|CORRECT_ABSTAIN|HALLUCINATION>",
  "rationale": "<one concise sentence>",
  "retrieval_provided_answer": <true|false>
}}

---

QUESTION: {question}
EXPECTED_BEHAVIOR: {expected_behavior}
GROUND_TRUTH: {ground_truth}

RETRIEVED_CONTEXT:
{retrieved_context}

RESPONSE:
{response}
"""


def format_judge_prompt(
    *,
    question: str,
    expected_behavior: str,
    ground_truth: str | None,
    retrieved_context: str,
    response: str,
) -> str:
    """Render JUDGE_PROMPT_V0 with the five fields substituted.

    The substituted strings (question, response, retrieved_context, etc.) can contain
    literal `{` or `}` characters without breaking the template — `.format()` only
    interprets braces in the TEMPLATE, not in the substituted values.
    """
    gt = "null" if ground_truth is None or ground_truth == "" else str(ground_truth)
    return JUDGE_PROMPT_V0.format(
        question=question,
        expected_behavior=expected_behavior,
        ground_truth=gt,
        retrieved_context=retrieved_context,
        response=response,
    )


# Active prompt (alias of the locked version). Downstream code references this.
JUDGE_PROMPT = JUDGE_PROMPT_V0