"""
Smoke test: classify a HANDFUL of issues with ONE model and print the result.

This is not the full run - it's a "does the pipeline actually work?" check before
we spend credits on the whole corpus. It shows, per issue: the label the model
picked, the correct answer, tokens used, and how long the call took.

Run:  python src/smoke_test.py                 (uses default model)
      python src/smoke_test.py openai-gpt-oss-120b   (pick a model)
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from inference import make_client, classify_issue

ROOT = Path(__file__).resolve().parent.parent
ISSUES_FILE = ROOT / "data" / "issues.json"
GROUND_TRUTH_FILE = ROOT / "data" / "ground_truth.json"

DEFAULT_MODEL = "openai-gpt-oss-20b"   # small + cheap, good for a first check
NUM_ISSUES = 5


def main():
    load_dotenv()
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    issues = json.loads(ISSUES_FILE.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))

    # Pick a few issues that we KNOW the answer to, so we can eyeball correctness.
    labeled = [i for i in issues if str(i["number"]) in truth][:NUM_ISSUES]

    client = make_client()
    print(f"Model: {model}\n" + "=" * 60)

    for issue in labeled:
        result = classify_issue(client, model, issue)
        correct = truth[str(issue["number"])]
        mark = "OK " if result.label == correct else "XX "
        if result.error:
            mark = "ERR"

        print(f"\n[{mark}] #{issue['number']}  {issue['title'][:55]}")
        print(f"      model said : {result.label}   (correct: {correct})")
        print(f"      tokens     : {result.prompt_tokens} in / "
              f"{result.completion_tokens} out")
        print(f"      latency    : {result.latency_s:.2f}s   attempts: {result.attempts}")
        if result.error:
            print(f"      ERROR ({result.error_type}): {result.error[:100]}")


if __name__ == "__main__":
    main()
