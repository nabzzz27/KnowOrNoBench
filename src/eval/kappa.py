"""Cohen's κ + bootstrap 95% CI for judge-vs-hand agreement.

The eval reports κ on the 30 test-set hand-labels (computed once, after the judge prompt
is locked). κ on the 30 dev-set labels is used iteratively during prompt tuning.

Why Cohen's κ (not raw accuracy):
  κ corrects for chance agreement. On 5 labels with skewed distributions, raw accuracy
  can look great while the judge actually agrees by guessing the majority class. κ
  rewards agreement above chance only.

Implementation:
  - Cohen's κ on 5 categorical labels with equal weighting (standard).
  - Wilson-style 95% CI via 1000-resample percentile bootstrap (paired rows).
  - Returns a structured dict with κ, CI bounds, 5×5 confusion matrix, n.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Sequence


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Cohen's κ on two equally-long label sequences over the same items.

    κ = (p_o − p_e) / (1 − p_e)
      where p_o is the observed agreement and p_e is the expected agreement under
      independent marginal distributions.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"length mismatch: {len(labels_a)} vs {len(labels_b)}")
    n = len(labels_a)
    if n == 0:
        raise ValueError("cohen_kappa: empty input")

    # Observed agreement
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n

    # Expected agreement under independence: sum over labels of P_a(l) * P_b(l)
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    p_e = sum((count_a[l] / n) * (count_b[l] / n)
              for l in set(count_a) | set(count_b))

    if p_e >= 1.0:
        # Both annotators always pick the same single label — agreement undefined.
        # Treat as perfect agreement iff that's the observed case.
        return 1.0 if p_o == 1.0 else 0.0

    return (p_o - p_e) / (1 - p_e)


def confusion_matrix(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    label_universe: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Build a {row_label: {col_label: count}} matrix.

    Convention: rows = labels_a (hand), cols = labels_b (judge).
    Universe ensures all 5 labels appear even if some have count 0.
    """
    matrix = {a: {b: 0 for b in label_universe} for a in label_universe}
    for a, b in zip(labels_a, labels_b):
        if a not in matrix or b not in matrix[a]:
            raise ValueError(f"label outside universe: a={a!r} b={b!r}")
        matrix[a][b] += 1
    return matrix


def bootstrap_kappa_ci(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap percentile CI on Cohen's κ. Returns (low, high)."""
    if len(labels_a) != len(labels_b):
        raise ValueError("length mismatch")
    n = len(labels_a)
    if n == 0:
        return (0.0, 0.0)

    rng = random.Random(seed)
    kappas: list[float] = []
    pairs = list(zip(labels_a, labels_b))
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        a_b = list(zip(*sample))
        try:
            kappas.append(cohen_kappa(a_b[0], a_b[1]))
        except ValueError:
            continue

    if not kappas:
        return (0.0, 0.0)
    kappas.sort()
    lo_idx = int((1 - confidence) / 2 * len(kappas))
    hi_idx = int((1 + confidence) / 2 * len(kappas)) - 1
    return (kappas[lo_idx], kappas[hi_idx])


def compute_kappa_block(
    hand_labels: Sequence[str],
    judge_labels: Sequence[str],
    label_universe: Sequence[str],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Bundle κ + CI + confusion matrix + N for one split (dev or test).

    `hand_labels` and `judge_labels` must be paired by row — i.e. labels_a[i] and
    labels_b[i] refer to the same RAG response cell.
    """
    n = len(hand_labels)
    kappa = cohen_kappa(hand_labels, judge_labels)
    ci_low, ci_high = bootstrap_kappa_ci(
        hand_labels, judge_labels, n_resamples=n_resamples, seed=seed)
    cm = confusion_matrix(hand_labels, judge_labels, label_universe)
    return {
        "kappa": round(kappa, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "n": n,
        "confusion_matrix": cm,
    }
