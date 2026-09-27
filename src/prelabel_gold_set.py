"""
Step 2 of 3: apply AI-suggested labels to the review sheet.

Run these in order to build the answer key:
    1. build_gold_set.py           pick which issues to label
    2. prelabel_gold_set.py    <-- THIS FILE: AI drafts a label + flags hard ones
    -. [human verifies the CSV]    you check/fix every label (the real truth)
    3. finalize_ground_truth.py    validate + freeze the answer key

IMPORTANT (methodology / honesty):
    These labels were suggested by an AI assistant (Claude) after reading each
    issue, using the SAME rulebook in labels.py. They are a STARTING POINT.
    A human (you) then verifies them - that human judgment is the real ground
    truth. This is standard "AI-assisted labeling with human review".

    Note: the assisting model (Claude) is NOT one of the models being evaluated
    (those are DigitalOcean open-weight models), so there is no "model grading
    itself" problem.

The 'check' column flags issues where the call was genuinely debatable - often
where the maintainer's label disagreed with the rulebook. Review those first.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEW_FILE = ROOT / "data" / "gold_set_to_review.csv"

# issue number -> AI-suggested label (from reading each issue against the rulebook)
AI_LABELS = {
    18: "enhancement", 48: "question", 75: "bug", 90: "enhancement",
    95: "enhancement", 130: "enhancement", 167: "enhancement", 189: "bug",
    200: "enhancement", 219: "enhancement", 238: "enhancement", 246: "bug",
    252: "bug", 261: "bug", 275: "enhancement", 293: "enhancement", 294: "bug",
    298: "question", 308: "question", 309: "question", 314: "enhancement",
    318: "question", 328: "bug", 329: "bug", 336: "question", 340: "enhancement",
    394: "enhancement", 400: "enhancement", 409: "question", 417: "question",
    420: "enhancement", 425: "bug", 448: "bug", 457: "bug", 480: "bug",
    493: "question", 535: "question", 547: "enhancement", 567: "bug", 604: "bug",
    630: "enhancement", 640: "question", 654: "other", 670: "question", 671: "bug",
    692: "question", 710: "question", 720: "bug", 722: "question", 729: "security",
    735: "security", 771: "question", 791: "enhancement", 802: "enhancement",
    803: "enhancement", 811: "security", 812: "security", 814: "enhancement",
    817: "enhancement", 821: "enhancement", 830: "security", 833: "bug",
    840: "enhancement", 843: "security", 849: "security", 859: "security",
    880: "enhancement", 927: "security", 932: "security", 936: "enhancement",
    937: "enhancement", 958: "security", 968: "bug", 973: "enhancement",
    974: "enhancement", 975: "bug", 985: "enhancement", 993: "security",
    1061: "security", 1064: "bug", 1066: "security", 1071: "bug",
    1083: "enhancement", 1085: "bug", 1086: "bug", 1087: "bug", 1090: "security",
    1091: "security", 1092: "security", 1094: "security", 1095: "security",
    1096: "security", 1105: "enhancement", 1106: "other", 1130: "security",
    1131: "security", 1139: "security", 1156: "documentation", 1164: "security",
    1175: "security", 1190: "documentation", 1196: "security", 1268: "enhancement",
    1274: "enhancement", 1281: "bug", 1284: "bug", 1306: "bug", 1326: "enhancement",
    1329: "bug", 1335: "enhancement", 1354: "enhancement", 1358: "enhancement",
    1359: "enhancement", 1364: "bug", 1380: "bug", 1382: "bug", 1417: "bug",
    1460: "enhancement", 1461: "enhancement", 1464: "bug", 1505: "enhancement",
    1527: "bug", 1534: "documentation", 1614: "bug", 1624: "enhancement",
    1647: "enhancement", 1656: "bug", 1710: "bug", 1733: "bug", 1737: "enhancement",
    1749: "bug", 1763: "bug", 1882: "bug",
}

# Issues where the call was debatable - verify these FIRST. The note explains why.
CHECK_NOTES = {
    130: "brew delivers old version - could be bug (broken pkg) vs enhancement (update pkg)",
    293: "asks to see LB logs - question vs enhancement (feature request)",
    318: "trivial 'what does doctl mean' - question vs other (off-topic)",
    425: "maintainer said 'question', but command freezes = defect -> I chose bug",
    567: "maintainer said 'question', but wrong droplet size created = defect -> I chose bug",
    654: "'geth cannot execute' - unrelated to doctl -> I chose other (off-topic)",
    771: "maintainer said 'bug', but title is [QUESTION] and body asks how-to -> I chose question",
    814: "wants list of sizes 'for reference on docs' - enhancement vs documentation",
    817: "maintainer said 'bug', but asks for ability to remove tokens = feature -> I chose enhancement",
    821: "maintainer said 'bug', but asks to set token expiry = feature -> I chose enhancement",
    1106: "maintainer said 'bug|duplicate'; schema says duplicates -> other -> I chose other",
    1534: "maintainer said 'bug', but complaint is 'attributes undocumented' -> I chose documentation",
    1763: "'non working example' in help - bug (broken cmd) vs documentation (wrong example)",
}


def main():
    with open(REVIEW_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    updated = 0
    for row in rows:
        number = int(row["number"])
        if number in AI_LABELS:
            row["final_label"] = AI_LABELS[number]
            row["check"] = "REVIEW" if number in CHECK_NOTES else ""
            row["note"] = CHECK_NOTES.get(number, "")
            updated += 1

    # Rewrite with the new columns in a sensible order for reviewing.
    fieldnames = ["number", "check", "final_label", "hint_label",
                  "maintainer_labels", "title", "body_preview", "note",
                  "state", "url"]
    with open(REVIEW_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    # Summary of the label distribution.
    counts = {}
    for row in rows:
        counts[row["final_label"]] = counts.get(row["final_label"], 0) + 1

    print(f"Applied AI labels to {updated} issues.")
    print("\nLabel distribution in the gold set:")
    for label in ["bug", "enhancement", "question", "security", "documentation", "other"]:
        print(f"  {label:14s}: {counts.get(label, 0)}")
    print(f"\nFlagged for priority review (check=REVIEW): {len(CHECK_NOTES)}")
    print(f"\nUpdated file: {REVIEW_FILE}")


if __name__ == "__main__":
    main()
