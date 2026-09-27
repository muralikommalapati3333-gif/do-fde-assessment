"""
Step 1 of 3: build a balanced sample for human review.

Run these in order to build the answer key:
    1. build_gold_set.py       <-- THIS FILE: pick which issues to label
    2. prelabel_gold_set.py        AI drafts a label for each + flags hard ones
    -. [human verifies the CSV]    you check/fix every label (the real truth)
    3. finalize_ground_truth.py    validate + freeze the answer key

What it does:
    - reads the frozen issues (data/issues.json)
    - guesses a starting label for each using the maintainer-label HINT
    - picks a balanced sample that makes sure rare classes (security,
      documentation, question) are well represented, not drowned out by bugs
    - writes data/gold_set_to_review.csv for you to confirm/correct by hand

Why "balanced" (stratified) sampling?
    A purely random sample would be mostly bugs and enhancements, with almost no
    security or documentation examples. Then per-class precision/recall for those
    rare classes would be meaningless. So we deliberately over-sample the rare
    classes.

Why a fixed random seed?
    So the sample is identical every run - reproducible.
"""

import csv
import json
import random
from pathlib import Path

from labels import hint_from_maintainer_labels

ROOT = Path(__file__).resolve().parent.parent
ISSUES_FILE = ROOT / "data" / "issues.json"
REVIEW_FILE = ROOT / "data" / "gold_set_to_review.csv"

SEED = 42  # fixed seed -> same sample every time

# How many of each hint bucket to include in the review sample.
# Rare classes: take (almost) all. Common classes: take a sample.
# "unlabeled" issues are sampled too, so we can find 'other' and catch cases the
# maintainers missed.
SAMPLE_PLAN = {
    "security": 26,        # take all we have
    "documentation": 20,   # take all (there are very few)
    "question": 20,        # take all we have
    "bug": 30,             # sample
    "enhancement": 30,     # sample
    "unlabeled": 30,       # sample issues with no usable maintainer label
}


def main():
    with open(ISSUES_FILE, encoding="utf-8") as f:
        issues = json.load(f)

    # Bucket every issue by its hint label ("" means no usable hint = unlabeled).
    buckets = {key: [] for key in SAMPLE_PLAN}
    for issue in issues:
        hint = hint_from_maintainer_labels(issue["labels"])
        bucket = hint if hint else "unlabeled"
        if bucket in buckets:
            buckets[bucket].append(issue)

    # Sample from each bucket using the fixed seed.
    rng = random.Random(SEED)
    selected = []
    for bucket, want in SAMPLE_PLAN.items():
        pool = buckets[bucket]
        take = min(want, len(pool))
        selected.extend(rng.sample(pool, take))
        print(f"  {bucket:14s}: have {len(pool):3d}, taking {take}")

    # De-duplicate (an issue can only appear once) and sort by number.
    seen = set()
    unique = []
    for issue in selected:
        if issue["number"] not in seen:
            seen.add(issue["number"])
            unique.append(issue)
    unique.sort(key=lambda i: i["number"])

    # Write the review CSV. You will edit the 'final_label' column.
    with open(REVIEW_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["number", "state", "maintainer_labels", "hint_label",
             "final_label", "title", "body_preview", "url"]
        )
        for issue in unique:
            hint = hint_from_maintainer_labels(issue["labels"])
            body_preview = (issue["body"] or "").replace("\n", " ").replace("\r", " ")[:300]
            writer.writerow([
                issue["number"],
                issue["state"],
                "|".join(issue["labels"]),
                hint,
                hint,                       # pre-fill final_label with the hint
                issue["title"],
                body_preview,
                issue["url"],
            ])

    print(f"\nWrote {len(unique)} issues to review:")
    print(f"  {REVIEW_FILE}")
    print("\nNext: open that CSV, check the 'final_label' column for each row,")
    print("and fix any that are wrong. Allowed labels:")
    print("  bug, enhancement, question, documentation, security, other")


if __name__ == "__main__":
    main()
