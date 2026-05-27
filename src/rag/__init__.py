"""RAG orchestration: retrieve, prompt, generate, answer.

The public surface is `answer(question, config) -> dict` — the contract the eval calls.
"""

from src.rag.answer import answer

__all__ = ["answer"]
