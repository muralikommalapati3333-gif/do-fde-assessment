"""
Fetch the issues
================

Goal: download the doctl GitHub issues ONCE and freeze them to a file, so that
every model we test later is scored on the exact same pile of tickets.

Why freeze to a file?
    The corpus must stay stable across runs. If we re-fetched from GitHub every
    time, new issues could appear between runs and the model comparison would be
    unfair. Fetch once -> save -> reuse.

Why filter out pull requests?
    GitHub's "issues" API quietly includes pull requests in the same list.
    A PR is not an issue, so we drop anything that has a "pull_request" field.

This is intentionally small - ingestion is just plumbing, so it stays lean and
the effort goes into the parts that matter (labeling and scoring).
"""

import json
import os
import time
from pathlib import Path

import requests

# --- Settings -------------------------------------------------------------

REPO = "digitalocean/doctl"          # the repository we pull issues from
PER_PAGE = 100                       # GitHub returns at most 100 items per page
API_URL = f"https://api.github.com/repos/{REPO}/issues"

# Where to save the frozen corpus. We compute the path relative to THIS file,
# so the script works no matter which folder you run it from.
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "data" / "issues.json"


def fetch_all_issues():
    """Loop through every page of issues (open + closed) and return them all."""

    # A GitHub token is OPTIONAL. Without one, GitHub allows 60 requests/hour,
    # which is plenty for ~500 issues. If you set GITHUB_TOKEN it raises the
    # limit to 5000/hour - useful if you re-run a lot.
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    all_items = []

    # We use CURSOR-BASED pagination. GitHub no longer supports simple page
    # numbers (page=1,2,3...) for large lists - it returns a 422 error and tells
    # you to follow the "next" link instead. So we start with the first URL, then
    # keep following the "next" link GitHub gives us in the response headers,
    # until there is no next link (that means we've reached the end).
    url = API_URL
    params = {
        "state": "all",       # "all" = both open AND closed issues
        "per_page": PER_PAGE,
    }
    page = 1

    while url:
        response = requests.get(url, headers=headers, params=params, timeout=30)

        # If GitHub says no (e.g. rate limit), stop and show why.
        if response.status_code != 200:
            raise RuntimeError(
                f"GitHub returned {response.status_code}: {response.text[:200]}"
            )

        page_items = response.json()
        all_items.extend(page_items)
        print(f"  fetched page {page} ({len(page_items)} items)")

        # GitHub puts the link to the next page in the response's "Link" header.
        # The requests library parses it into response.links for us.
        next_link = response.links.get("next")
        url = next_link["url"] if next_link else None
        params = None            # the next URL already contains all parameters
        page += 1
        time.sleep(0.5)          # be polite to the API

    return all_items


def clean_issues(raw_items):
    """Drop pull requests and keep only the fields we actually need."""
    issues = []
    pull_requests_dropped = 0

    for item in raw_items:
        # GitHub marks pull requests with a "pull_request" field. Skip those.
        if "pull_request" in item:
            pull_requests_dropped += 1
            continue

        issues.append({
            "number": item["number"],
            "title": item.get("title") or "",
            "body": item.get("body") or "",
            "state": item.get("state"),                       # open / closed
            "labels": [lbl["name"] for lbl in item.get("labels", [])],
            "url": item.get("html_url"),
        })

    return issues, pull_requests_dropped


def main():
    print(f"Downloading issues from {REPO} ...")
    raw_items = fetch_all_issues()

    issues, pr_dropped = clean_issues(raw_items)

    # Save the frozen corpus.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)

    # ---- Summary so you can SEE the data (Prerequisite 1) ----
    print("\n" + "=" * 55)
    print(f"Total items GitHub returned : {len(raw_items)}")
    print(f"Pull requests dropped       : {pr_dropped}")
    print(f"Real issues kept            : {len(issues)}")
    print(f"Saved to                    : {OUTPUT_FILE}")

    # How many issues already have a maintainer label vs none?
    labeled = sum(1 for i in issues if i["labels"])
    print(f"\nIssues WITH maintainer labels : {labeled}")
    print(f"Issues WITHOUT any label      : {len(issues) - labeled}")

    # Show the raw maintainer labels so we can see how messy they are before we
    # start building our own answer key.
    label_counts = {}
    for issue in issues:
        for name in issue["labels"]:
            label_counts[name] = label_counts.get(name, 0) + 1
    print("\nMaintainer label frequencies (top 20):")
    for name, count in sorted(label_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {count:4d}  {name}")

    # Peek at 3 real issues.
    print("\n--- 3 sample issues ---")
    for issue in issues[:3]:
        preview = issue["body"][:200].replace("\n", " ")
        print(f"\n#{issue['number']} [{issue['state']}] labels={issue['labels']}")
        print(f"  title: {issue['title']}")
        print(f"  body : {preview}...")


if __name__ == "__main__":
    main()
