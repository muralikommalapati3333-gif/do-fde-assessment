"""
Scoring. Turn a pile of per-call results into the numbers we report. This file does
MATH ONLY - no API calls - so it's fast and testable.

This produces all the numbers we report:
  - accuracy, and per-class precision / recall / F1
  - a confusion matrix
  - cost per call, total cost, and cost per CORRECT classification
    (token counts x per-token price -> dollars, fully traceable)
  - p50 / p95 latency, wall-clock, throughput (requests/sec)
  - error rate broken down by type

How we treat failures:
  A call that never returned a usable label (an API error, or an unparseable
  reply) is NOT quietly dropped. We report two accuracies:
    - accuracy_over_classified: correct / (calls that produced a label)
    - accuracy_over_all:        correct / (every scored issue)
  The gap between them is the price of the model's failures. Hiding failures
  would flatter a flaky model, so we surface both.
"""

from labels import LABELS


def percentile(values, pct):
    """Simple percentile (linear interpolation). pct is 0-100."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def cost_for_call(prompt_tokens, completion_tokens, price_in_per_m, price_out_per_m):
    """Dollars for one call. Prices are USD per 1,000,000 tokens.

    The whole money calculation lives in one place on purpose, so the dollar
    figure stays traceable: token counts x per-token rates -> dollars.
    """
    return (prompt_tokens / 1_000_000) * price_in_per_m + \
           (completion_tokens / 1_000_000) * price_out_per_m


def confusion_matrix(results, truth):
    """counts[true_label][predicted_label] = how many. Only classified calls."""
    matrix = {t: {p: 0 for p in LABELS} for t in LABELS}
    for r in results:
        if r.label is None:
            continue
        true_label = truth.get(str(r.number))
        if true_label is None:
            continue
        matrix[true_label][r.label] += 1
    return matrix


def per_class_prf(results, truth):
    """Precision, recall, F1, and support for each label."""
    # Count true-positives, false-positives, false-negatives per label.
    tp = {label: 0 for label in LABELS}
    fp = {label: 0 for label in LABELS}
    fn = {label: 0 for label in LABELS}
    support = {label: 0 for label in LABELS}

    for r in results:
        true_label = truth.get(str(r.number))
        if true_label is None:
            continue
        support[true_label] += 1
        if r.label is None:
            # No prediction -> a miss for the true class, but not a false-positive
            # for any predicted class.
            fn[true_label] += 1
            continue
        if r.label == true_label:
            tp[true_label] += 1
        else:
            fp[r.label] += 1
            fn[true_label] += 1

    out = {}
    for label in LABELS:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support[label],
        }
    return out


def summarize(model, results, truth, price_in_per_m, price_out_per_m,
              wall_clock_s, concurrency):
    """Roll a list of CallResults for ONE model into a full metrics dict."""
    scored = [r for r in results if str(r.number) in truth]
    classified = [r for r in scored if r.label is not None]

    correct = sum(
        1 for r in classified if r.label == truth[str(r.number)]
    )

    # --- accuracy (two versions, see module docstring) ---
    acc_over_classified = correct / len(classified) if classified else 0.0
    acc_over_all = correct / len(scored) if scored else 0.0

    # --- cost (over the WHOLE corpus run, scored + unscored) ---
    total_cost = sum(
        cost_for_call(r.prompt_tokens, r.completion_tokens,
                      price_in_per_m, price_out_per_m)
        for r in results
    )
    # Cost of just the scored issues, used for "cost per correct answer".
    total_cost_scored = sum(
        cost_for_call(r.prompt_tokens, r.completion_tokens,
                      price_in_per_m, price_out_per_m)
        for r in scored
    )
    cost_per_correct = total_cost_scored / correct if correct else float("inf")

    # --- latency / throughput ---
    # A call counts toward latency if it actually completed a network round-trip
    # (latency_s > 0). That includes calls whose reply we couldn't parse into a
    # label - they still measured a real API response time. Hard failures
    # (timeout / rate_limit) return latency_s = 0.0, so they're excluded here.
    latencies = [r.latency_s for r in results if r.latency_s > 0]
    total_calls = len(results)
    throughput = total_calls / wall_clock_s if wall_clock_s > 0 else 0.0

    # --- errors, broken down by type ---
    errors = [r for r in results if r.error is not None]
    error_breakdown = {}
    for r in errors:
        error_breakdown[r.error_type] = error_breakdown.get(r.error_type, 0) + 1
    error_rate = len(errors) / total_calls if total_calls else 0.0

    # --- token totals ---
    total_prompt_tokens = sum(r.prompt_tokens for r in results)
    total_completion_tokens = sum(r.completion_tokens for r in results)

    return {
        "model": model,
        "concurrency": concurrency,
        # counts
        "total_issues": total_calls,
        "scored_issues": len(scored),
        "classified": len(classified),
        "correct": correct,
        # accuracy
        "accuracy_over_classified": acc_over_classified,
        "accuracy_over_all": acc_over_all,
        "per_class": per_class_prf(results, truth),
        "confusion_matrix": confusion_matrix(results, truth),
        # cost
        "total_cost_usd": total_cost,
        "scored_cost_usd": total_cost_scored,
        "cost_per_correct_usd": cost_per_correct,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "price_in_per_m": price_in_per_m,
        "price_out_per_m": price_out_per_m,
        # latency / throughput
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "wall_clock_s": wall_clock_s,
        "throughput_rps": throughput,
        # errors
        "error_rate": error_rate,
        "error_breakdown": error_breakdown,
    }
