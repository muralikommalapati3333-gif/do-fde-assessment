"""
Read the saved raw runs, apply prices, and print a comparison table.

This is where dollars appear. It re-loads the raw per-call results from
data/runs/, joins them to the answer key and the price table, and prints one row
per model plus per-class detail. Nothing here calls the API - so you can re-run it
freely, and re-run it after fixing a price, for $0.

Run:  python src/report.py
"""

import json
from pathlib import Path

from dataclasses import dataclass

from labels import LABELS
from metrics import summarize
from pricing import price_for, PRICING_SOURCE_DATE

ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_FILE = ROOT / "data" / "ground_truth.json"
RUNS_DIR = ROOT / "data" / "runs"


@dataclass
class Row:
    """Minimal stand-in for CallResult when loading from JSON (metrics needs .attrs)."""
    number: int
    label: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    error: str | None
    error_type: str | None
    model: str = ""
    raw_output: str = ""
    attempts: int = 1


def load_run(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    results = [
        Row(
            number=r["number"], label=r["label"],
            prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
            latency_s=r["latency_s"], error=r["error"], error_type=r["error_type"],
            raw_output=r.get("raw_output", ""),
        )
        for r in data["results"]
    ]
    return data, results


def main():
    truth = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))
    run_files = sorted(RUNS_DIR.glob("*.json"))
    if not run_files:
        print("No runs found. Run:  python src/run_comparison.py")
        return

    summaries = []
    for path in run_files:
        data, results = load_run(path)
        model = data["model"]
        try:
            price_in, price_out = price_for(model)
            priced = True
        except (KeyError, ValueError):
            # No confirmed price yet - still report accuracy/latency, blank cost.
            price_in, price_out, priced = 0.0, 0.0, False
        summary = summarize(
            model, results, truth, price_in, price_out,
            wall_clock_s=data["wall_clock_s"], concurrency=data["concurrency"],
        )
        summary["priced"] = priced
        summaries.append(summary)

    if not summaries:
        return

    # Sort best-accuracy first.
    summaries.sort(key=lambda s: s["accuracy_over_all"], reverse=True)

    print(f"Prices as of {PRICING_SOURCE_DATE}. "
          f"Concurrency shown per model.\n")
    header = (f"{'model':<30} {'acc':>6} {'acc*':>6} {'$/1k':>7} "
              f"{'$/correct':>10} {'p50':>6} {'p95':>6} {'rps':>6} {'err%':>6}")
    print(header)
    print("-" * len(header))
    for s in summaries:
        cost_per_1k = s["scored_cost_usd"] / s["scored_issues"] * 1000 if s["scored_issues"] else 0
        print(
            f"{s['model']:<30} "
            f"{s['accuracy_over_all']*100:>5.1f}% "
            f"{s['accuracy_over_classified']*100:>5.1f}% "
            f"{cost_per_1k:>7.3f} "
            f"{s['cost_per_correct_usd']*1000:>9.4f}m "
            f"{s['latency_p50_s']:>5.2f}s "
            f"{s['latency_p95_s']:>5.2f}s "
            f"{s['throughput_rps']:>5.2f} "
            f"{s['error_rate']*100:>5.1f}%"
        )

    print("\nLegend: acc = correct/all scored | acc* = correct/classified "
          "(ignores failures) | $/1k = USD per 1000 issues | "
          "$/correct in milli-dollars | rps = requests/sec\n")

    # Per-class F1 table (where each model is weak).
    print("Per-class F1 (support in parentheses):")
    label_hdr = "  ".join(f"{l[:5]:>7}" for l in LABELS)
    print(f"{'model':<30} {label_hdr}")
    for s in summaries:
        cells = []
        for label in LABELS:
            pc = s["per_class"][label]
            cells.append(f"{pc['f1']:>4.2f}({pc['support']:>2})")
        print(f"{s['model']:<30} " + "  ".join(f"{c:>7}" for c in cells))


if __name__ == "__main__":
    main()
