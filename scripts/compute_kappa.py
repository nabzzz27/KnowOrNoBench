"""Compute Cohen's κ + bootstrap CI for hand-labels vs judge-labels.

Reads:
  - results/hand_labels.xlsx (the user-filled file)
  - results/judged_{neutral,forced,strict}.xlsx

Joins on (benchmark_id, config). Splits by hand_labels.split ("dev" / "test").

Writes:
  - results/kappa_dev.json
  - results/kappa_test.json

Prints a summary table to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import config
from src.eval.judge_prompt import VALID_LABELS
from src.eval.kappa import compute_kappa_block
from src.eval.sample_for_kappa import DEFAULT_HAND_LABELS_PATH


HAND_LABELS_DEFAULT = DEFAULT_HAND_LABELS_PATH
JUDGED_PATHS_DEFAULT = {
    cfg: config.REPO_ROOT / "results" / f"judged_{cfg}.xlsx"
    for cfg in ("neutral", "forced", "strict")
}
KAPPA_DEV_PATH = config.REPO_ROOT / "results" / "kappa_dev.json"
KAPPA_TEST_PATH = config.REPO_ROOT / "results" / "kappa_test.json"


def _join_hand_and_judge(
    hand_xlsx: Path,
    judged_paths: dict[str, Path],
) -> pd.DataFrame:
    """Return a DataFrame with columns [benchmark_id, config, split, hand_label,
    judge_label] joined from hand_labels.xlsx and the three judged xlsx files."""
    hand = pd.read_excel(hand_xlsx).fillna("")
    if hand["hand_label"].astype(str).str.strip().eq("").any():
        n_blank = (hand["hand_label"].astype(str).str.strip() == "").sum()
        raise ValueError(
            f"{n_blank} rows in {hand_xlsx.name} are missing hand_label — "
            "finish the hand-labelling first.")

    judge_frames = []
    for cfg, p in judged_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"judged file missing: {p}")
        df = pd.read_excel(p).fillna("")[["benchmark_id", "label"]]
        df = df.rename(columns={"label": "judge_label"})
        df["config"] = cfg
        judge_frames.append(df)
    judge_all = pd.concat(judge_frames, ignore_index=True)

    merged = hand.merge(judge_all, on=["benchmark_id", "config"], how="left")

    # Validate
    if merged["judge_label"].astype(str).str.strip().eq("").any():
        n_blank = (merged["judge_label"].astype(str).str.strip() == "").sum()
        raise ValueError(
            f"{n_blank} rows could not be joined to a judge label — "
            "did you run `python -m scripts.run_judge` on all 180 cells?")

    return merged[["benchmark_id", "config", "split", "hand_label", "judge_label"]]


def compute_kappa(
    hand_xlsx: Path | None = None,
    judged_paths: dict[str, Path] | None = None,
) -> dict:
    """Return {"dev": <block>, "test": <block>} for the two splits."""
    h = Path(hand_xlsx) if hand_xlsx else HAND_LABELS_DEFAULT
    j = judged_paths or JUDGED_PATHS_DEFAULT
    merged = _join_hand_and_judge(h, j)

    out: dict = {}
    for split in ("dev", "test"):
        rows = merged[merged["split"] == split]
        if len(rows) == 0:
            out[split] = {"kappa": None, "ci_low": None, "ci_high": None,
                          "n": 0, "confusion_matrix": {}}
            continue
        out[split] = compute_kappa_block(
            list(rows["hand_label"]),
            list(rows["judge_label"]),
            VALID_LABELS,
        )
    return out


def _print_block(name: str, block: dict) -> None:
    print(f"\n=== {name} (N={block['n']}) ===")
    if block["n"] == 0:
        print("  (no rows in this split)")
        return
    print(f"  κ = {block['kappa']:.3f}  "
          f"(95% CI: {block['ci_low']:.3f} – {block['ci_high']:.3f})")
    print("  confusion matrix (rows = hand, cols = judge):")
    header = "    " + " " * 18 + "  ".join(f"{l[:6]:>6}" for l in VALID_LABELS)
    print(header)
    for l in VALID_LABELS:
        row = block["confusion_matrix"].get(l, {})
        cells = "  ".join(f"{row.get(c, 0):>6}" for c in VALID_LABELS)
        print(f"    {l:<18} {cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Cohen's κ for hand vs judge labels.")
    ap.add_argument("--hand", type=Path, default=None,
                    help="Path to hand_labels.xlsx (default: results/hand_labels.xlsx)")
    args = ap.parse_args()

    result = compute_kappa(hand_xlsx=args.hand)

    KAPPA_DEV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(KAPPA_DEV_PATH, "w") as f:
        json.dump(result["dev"], f, indent=2)
    with open(KAPPA_TEST_PATH, "w") as f:
        json.dump(result["test"], f, indent=2)

    _print_block("DEV", result["dev"])
    _print_block("TEST", result["test"])
    print(f"\nwrote {KAPPA_DEV_PATH}")
    print(f"wrote {KAPPA_TEST_PATH}")


if __name__ == "__main__":
    main()
