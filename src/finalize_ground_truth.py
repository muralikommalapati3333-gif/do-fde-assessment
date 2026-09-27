"""
Step 3 of 3: freeze the verified labels into the answer key.

Run these in order to build the answer key:
    1. build_gold_set.py           pick which issues to label
    2. prelabel_gold_set.py        AI drafts a label for each + flags hard ones
    -. [human verifies the CSV]    you check/fix every label (the real truth)
    3. finalize_ground_truth.py <-- THIS FILE: validate + freeze the answer key

What it does:
    - reads data/gold_set_to_review.csv (the sheet you verified)
    - checks every row has a valid label from our schema
    - writes data/ground_truth.json: {issue_number: label} for the scored subset

Why a separate frozen file?
    The review CSV is a working document. Once you sign off, we freeze the
    answer key to its own file so scoring can never accidentally depend on the
    editable sheet. Every model is scored against this exact file - same ruler
    for everyone.

Why keep the full issue text OUT of this file?
    ground_truth.json is only the answer key (number -> correct label). The issue
    text already lives in data/issues.json. Keeping them separate means the
    scorer joins "what the issue says" (issues.json) with "the right answer"
    (ground_truth.json) by issue number - clean and hard to get wrong.
"""

import csv
import json
from pathlib import Path

from labels import LABELS

ROOT = Path(__file__).resolve().parent.parent
REVIEW_FILE = ROOT / "data" / "gold_set_to_review.csv"
GROUND_TRUTH_FILE = ROOT / "data" / "ground_truth.json"


def main():
    with open(REVIEW_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ground_truth = {}
    problems = []

    for row in rows:
        number = row["number"].strip()
        label = row["final_label"].strip().lower()

        if not label:
            problems.append(f"#{number}: no label")
            continue
        if label not in LABELS:
            problems.append(f"#{number}: '{label}' is not a valid label")
            continue
        ground_truth[number] = label

    if problems:
        print("STOPPING - fix these rows in the CSV first:")
        for p in problems:
            print(f"  {p}")
        return

    # Save the frozen answer key, sorted by issue number for a stable file.
    ordered = {n: ground_truth[n] for n in sorted(ground_truth, key=int)}
    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)

    # Summary.
    counts = {}
    for label in ground_truth.values():
        counts[label] = counts.get(label, 0) + 1

    print(f"Froze {len(ground_truth)} verified labels -> {GROUND_TRUTH_FILE}")
    print("\nGround-truth label distribution:")
    for label in LABELS:
        print(f"  {label:14s}: {counts.get(label, 0)}")


if __name__ == "__main__":
    main()
