"""
Run ONE model over MANY issues, in parallel, and time the whole thing.

Design:
  - A thread pool fires several calls at once to process the corpus in parallel.
  - The worker count comes from the INFERENCE_CONCURRENCY env var, so concurrency
    is configurable without rebuilding the container.
  - We time the whole batch with one clock, for wall-clock time and throughput.

Why threads and not async?
  Each call is a simple blocking HTTP request that spends its time WAITING on the
  network. A thread pool overlaps that waiting perfectly, needs no async plumbing,
  and keeps the classify function dead simple. Good enough, easy to explain.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from inference import classify_issue

DEFAULT_CONCURRENCY = 8


def get_concurrency():
    """Read the parallel-request count from the environment (default 8)."""
    raw = os.environ.get("INFERENCE_CONCURRENCY", str(DEFAULT_CONCURRENCY))
    try:
        value = int(raw)
        return value if value > 0 else DEFAULT_CONCURRENCY
    except ValueError:
        return DEFAULT_CONCURRENCY


def classify_corpus(client, model, issues, concurrency=None, progress=True):
    """Classify every issue with one model. Returns (results, wall_clock_s).

    Results come back in issue-number order so runs are comparable and stable.
    """
    if concurrency is None:
        concurrency = get_concurrency()

    results = []
    done = 0
    total = len(issues)
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(classify_issue, client, model, issue): issue
            for issue in issues
        }
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if progress and (done % 25 == 0 or done == total):
                print(f"    {model}: {done}/{total}")

    wall_clock_s = time.perf_counter() - start
    results.sort(key=lambda r: r.number)
    return results, wall_clock_s
