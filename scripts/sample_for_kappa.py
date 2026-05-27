"""Write `results/hand_labels.xlsx` for the κ-validation hand-labelling step.

Usage:
    python -m scripts.sample_for_kappa            # writes the file (refuses to overwrite)
    python -m scripts.sample_for_kappa --rebuild  # overwrite existing
"""

from __future__ import annotations

import argparse
import sys

from src.eval.sample_for_kappa import (
    DEFAULT_HAND_LABELS_PATH, sample_for_kappa, write_hand_labels_xlsx,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample 60 cells for κ-validation hand labelling.")
    ap.add_argument(
        "--rebuild", action="store_true",
        help="Overwrite results/hand_labels.xlsx if it already exists.",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the stratified sample (default 42; recorded in xlsx).",
    )
    args = ap.parse_args()

    if DEFAULT_HAND_LABELS_PATH.exists() and not args.rebuild:
        print(f"refusing to overwrite {DEFAULT_HAND_LABELS_PATH} — pass --rebuild to force",
              file=sys.stderr)
        sys.exit(1)

    df = sample_for_kappa(seed=args.seed)
    path = write_hand_labels_xlsx(df, seed=args.seed)
    print(f"wrote {path}")
    print(f"  total rows: {len(df)} (30 dev + 30 test)")
    print(f"  per category:")
    for cat in df["category"].unique():
        dev = ((df["category"] == cat) & (df["split"] == "dev")).sum()
        test = ((df["category"] == cat) & (df["split"] == "test")).sum()
        print(f"    {cat:<26}  dev={dev}  test={test}")


if __name__ == "__main__":
    main()
