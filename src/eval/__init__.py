"""Eval surface: benchmark loader, judge, metrics.

This package owns everything that reads/writes eval artifacts: it reads benchmark questions,
calls the RAG (via `src.rag.answer`), calls the judge, computes metrics, writes reports.
"""

from src.eval.benchmark_loader import load_benchmark

__all__ = ["load_benchmark"]

# Note: src.eval.run_eval (the module) is intentionally NOT re-exported from this
# package — re-exporting the same-named function would shadow the submodule and break
# `import src.eval.run_eval` / monkeypatching by full path. Callers should
# `from src.eval.run_eval import run_eval` directly.
