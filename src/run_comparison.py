"""
Run the comparison: classify the corpus with each candidate model and save the raw
per-call results to disk.

We save RAW results (labels, tokens, latency, errors) - not dollar figures. Money
is computed later in report.py from these raw files. That means we can fix a price
or change the pricing table WITHOUT paying to re-run the models.

CANDIDATES below is the set that produced the comparison in the README: the
DigitalOcean-hosted open-weight chat models, chosen to span size, price, and
architecture (dense / MoE / reasoning / vision) - not minor variants. A few models
we smoke-tested were dropped before a full run and are NOT listed here: reasoning /
very large models pacing at tens of seconds per issue (qwen3.5-397b-a17b,
arcee-trinity-large-thinking, mimo-v2.5-pro), kimi-k2.6 (forces temperature=1,
breaking determinism), and minimax-m2.5 / qwen3.8-max (unavailable at test time).

Run all:      python src/run_comparison.py
Run some:     python src/run_comparison.py mistral-3-14B gemma-4-31B-it
Scored only:  python src/run_comparison.py --scored-only   (cheaper; accuracy only)
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from inference import make_client
from runner import classify_corpus, get_concurrency

ROOT = Path(__file__).resolve().parent.parent
ISSUES_FILE = ROOT / "data" / "issues.json"
GROUND_TRUTH_FILE = ROOT / "data" / "ground_truth.json"
RUNS_DIR = ROOT / "data" / "runs"

CANDIDATES = [
    "openai-gpt-oss-20b",       # small, cheap baseline
    "openai-gpt-oss-120b",      # bigger sibling - does size buy accuracy here?
    "mistral-3-14B",            # small dense model, flat cheap pricing
    "gemma-4-31B-it",           # mid-size instruct model (recommended)
    "deepseek-4-flash",         # cheap flash model (recommended workhorse)
    "deepseek-v4.1-flash",      # newer flash variant
    "deepseek-3.2",             # larger deepseek - highest accuracy in the field
    "glm-5.3-flash",            # cheap GLM flash
    "glm-5.3",                  # premium GLM - does the price buy accuracy?
    "glm-5.2",                  # premium GLM, prior gen
    "llama-4-maverick",         # Meta MoE
    "nemotron-nano-12b-v2-vl",  # NVIDIA vision-language model (slow: ~0.4 req/s)
]


def main():
    load_dotenv()
    args = sys.argv[1:]
    scored_only = "--scored-only" in args
    models = [a for a in args if not a.startswith("--")] or CANDIDATES

    issues = json.loads(ISSUES_FILE.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))

    if scored_only:
        issues = [i for i in issues if str(i["number"]) in truth]

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    concurrency = get_concurrency()

    client = make_client()
    print(f"Corpus: {len(issues)} issues | concurrency: {concurrency} | "
          f"models: {len(models)}\n")

    for model in models:
        print(f"Running {model} ...")
        results, wall_clock_s = classify_corpus(client, model, issues, concurrency)

        # Save raw results + the run conditions (concurrency, wall-clock) so the
        # report can reproduce every number, including at-what-concurrency latency.
        payload = {
            "model": model,
            "concurrency": concurrency,
            "wall_clock_s": wall_clock_s,
            "scored_only": scored_only,
            "results": [r.as_dict() for r in results],
        }
        out_file = RUNS_DIR / f"{model.replace('/', '_')}.json"
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        ok = sum(1 for r in results if r.label is not None)
        errs = sum(1 for r in results if r.error is not None)
        print(f"  saved {out_file.name}: {ok} classified, {errs} errors, "
              f"{wall_clock_s:.1f}s wall-clock\n")


if __name__ == "__main__":
    main()
