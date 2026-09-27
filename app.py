"""
The Streamlit app. This is the eval harness itself, not a separate viewer: it runs
the same metrics.py math as the command-line report, so the numbers on screen are
the exact numbers from the run, not a prettier copy.

What it shows:
  Sidebar   - pick Model A and Model B; optionally run a fresh comparison live.
  Top       - operational metrics per model: cost, p50/p95 latency (with the
              concurrency they were measured at), wall-clock, throughput, errors.
  Scored    - accuracy plus per-class precision/recall/F1, confusion matrices side
              by side, and a drill-down into the issues where the two models
              disagreed (with the ground-truth label shown).
  Unscored  - each model's label per issue, the raw model output, the agreement
              rate as a headline number, the per-class spread, and a disagreement view.

The data comes from the saved runs in data/runs/, so the deployed app loads instantly
and needs no credits to browse. The "Run fresh" button re-runs live if a key is set.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from labels import LABELS                       # noqa: E402
from metrics import summarize                   # noqa: E402
from pricing import price_for                   # noqa: E402
from report import load_run                     # noqa: E402 (reuse the raw loader)

DATA = ROOT / "data"
RUNS_DIR = DATA / "runs"

# The two models we recommend for production (see README section 6). Highlighted
# in the leaderboard and flagged in the comparison.
RECOMMENDED = {"deepseek-4-flash", "gemma-4-31B-it"}

st.set_page_config(page_title="doctl issue classifier - model comparison",
                   layout="wide")


# --------------------------------------------------------------------------- #
# Data loading (cached so switching models is instant)
# --------------------------------------------------------------------------- #
@st.cache_data
def load_truth():
    return json.loads((DATA / "ground_truth.json").read_text(encoding="utf-8"))


@st.cache_data
def load_issues():
    issues = json.loads((DATA / "issues.json").read_text(encoding="utf-8"))
    return {str(i["number"]): i for i in issues}


@st.cache_data
def available_runs():
    """model name -> its raw run file path."""
    out = {}
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out[data["model"]] = str(path)
        except (json.JSONDecodeError, KeyError):
            continue
    return out


@st.cache_data
def load_model_run(path):
    """Return (meta, {number -> result-dict}, [Row objects for metrics])."""
    data, rows = load_run(Path(path))
    by_number = {str(r.number): r for r in rows}
    return data, by_number, rows


def model_summary(model, rows, meta, truth):
    """Full metrics for one model, or None if it has no price on file."""
    try:
        price_in, price_out = price_for(model)
    except (KeyError, ValueError):
        return None
    return summarize(model, rows, truth, price_in, price_out,
                     meta["wall_clock_s"], meta["concurrency"])


def leaderboard_df(runs, truth):
    """One row per model: the whole field at a glance, ranked by accuracy."""
    rows = []
    for model, path in runs.items():
        meta, _, model_rows = load_model_run(path)
        try:
            price_in, price_out = price_for(model)
            priced = True
        except (KeyError, ValueError):
            price_in, price_out, priced = 0.0, 0.0, False
        s = summarize(model, model_rows, truth, price_in, price_out,
                      meta["wall_clock_s"], meta["concurrency"])
        cost_1k = (s["scored_cost_usd"] / s["scored_issues"] * 1000
                   if priced and s["scored_issues"] else None)
        rows.append({
            "": "★" if model in RECOMMENDED else "",
            "model": model,
            "accuracy %": round(s["accuracy_over_all"] * 100, 1),
            "cost / 1k ($)": round(cost_1k, 3) if cost_1k is not None else None,
            "p95 (s)": round(s["latency_p95_s"], 2),
            "error %": round(s["error_rate"] * 100, 1),
        })
    return (pd.DataFrame(rows)
            .sort_values("accuracy %", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Sidebar - model selection
# --------------------------------------------------------------------------- #
runs = available_runs()
truth = load_truth()
issues = load_issues()

st.sidebar.title("Model comparison")

if not runs:
    st.sidebar.error("No runs found. Run:  python src/run_comparison.py")
    st.stop()

model_names = list(runs.keys())
st.sidebar.caption(f"{len(model_names)} models with cached runs")

model_a = st.sidebar.selectbox("Model A", model_names, index=0)
model_b = st.sidebar.selectbox(
    "Model B", model_names, index=min(1, len(model_names) - 1)
)

meta_a, by_num_a, rows_a = load_model_run(runs[model_a])
meta_b, by_num_b, rows_b = load_model_run(runs[model_b])

summ_a = model_summary(model_a, rows_a, meta_a, truth)
summ_b = model_summary(model_b, rows_b, meta_b, truth)


# --------------------------------------------------------------------------- #
# Header + operational metrics
# --------------------------------------------------------------------------- #
st.title("doctl GitHub issues - language model comparison")
st.caption("Classify each issue into one of: " + ", ".join(LABELS) +
           ". One inference call per issue. Same corpus for every model.")

# --------------------------------------------------------------------------- #
# Overview - the whole field at a glance, before drilling into two models
# --------------------------------------------------------------------------- #
st.header("All models at a glance")
st.caption("Higher **accuracy** is better; lower **cost**, **p95 latency**, and "
           "**error %** are better. ★ = recommended for production. Ranked by accuracy.")

board = leaderboard_df(runs, truth)


def _highlight_reco(row):
    on = row["model"] in RECOMMENDED
    return ['background-color: #d1f0d9' if on else '' for _ in row]


st.dataframe(board.style.apply(_highlight_reco, axis=1),
             width='stretch', hide_index=True)
st.caption("Pick any two models in the sidebar to compare them in detail below.")
st.divider()


# --------------------------------------------------------------------------- #
# Head-to-head: Model A vs Model B (with win markers)
# --------------------------------------------------------------------------- #
def _mv(container, label, val, other, fmt, lower_better=False, help=None):
    """Show a metric plus the signed gap vs the other model as a green/red delta
    (green = this model is better on this metric)."""
    if val is None:
        container.metric(label, "n/a", help=help)
        return
    delta = None
    if other is not None and abs(val - other) > 1e-9:
        delta = fmt(val - other, signed=True)
    container.metric(label, fmt(val), delta=delta,
                     delta_color=("inverse" if lower_better else "normal"),
                     help=help)


def op_metrics_block(col, model, summ, other, meta):
    col.subheader(model + (" ★" if model in RECOMMENDED else ""))
    if summ is None:
        col.warning("No price on file - cost hidden. Add it in pricing.py.")

    pct = lambda x, signed=False: (f"{x*100:+.1f}%" if signed else f"{x*100:.1f}%")
    sec = lambda x, signed=False: (f"{x:+.2f}s" if signed else f"{x:.2f}s")
    rps = lambda x, signed=False: (f"{x:+.2f}/s" if signed else f"{x:.2f}/s")
    usd = lambda x, signed=False: ((f"+${x:.3f}" if x >= 0 else f"-${abs(x):.3f}")
                                   if signed else f"${x:.3f}")
    usdm = lambda x, signed=False: ((f"+${x*1000:.4f}m" if x >= 0 else f"-${abs(x)*1000:.4f}m")
                                    if signed else f"${x*1000:.4f}m")

    _mv(col, "Accuracy (all scored)",
        summ["accuracy_over_all"] if summ else None,
        other["accuracy_over_all"] if other else None, pct,
        help="Correct / all 133 scored issues. Higher is better.")

    c1, c2 = col.columns(2)
    if summ:
        cost1k = summ["scored_cost_usd"] / summ["scored_issues"] * 1000
        cost1k_o = (other["scored_cost_usd"] / other["scored_issues"] * 1000
                    if other else None)
        cost_call = summ["scored_cost_usd"] / summ["scored_issues"]
        cost_call_o = (other["scored_cost_usd"] / other["scored_issues"]
                       if other else None)
        # Left column = money, right column = speed.
        _mv(c1, "Cost / call", cost_call, cost_call_o, usdm, lower_better=True,
            help="Average USD for one issue, shown in milli-dollars "
                 "(tokens x per-token price). Lower is better.")
        _mv(c1, "Cost / 1k issues", cost1k, cost1k_o, usd, lower_better=True,
            help="USD to classify 1,000 issues. Lower is better.")
        _mv(c1, "Cost / correct", summ["cost_per_correct_usd"],
            other["cost_per_correct_usd"] if other else None, usdm,
            lower_better=True, help="USD per correct answer. Lower is better.")
        _mv(c2, "p50 latency", summ["latency_p50_s"],
            other["latency_p50_s"] if other else None, sec, lower_better=True,
            help="Typical (median) response time. Lower is better.")
        _mv(c2, "p95 latency", summ["latency_p95_s"],
            other["latency_p95_s"] if other else None, sec, lower_better=True,
            help="Near-worst-case response time. Lower is better.")
        _mv(c2, "Throughput", summ["throughput_rps"],
            other["throughput_rps"] if other else None, rps,
            help="Issues classified per second, at the concurrency below. "
                 "Higher is better.")
        _mv(col, "Error rate", summ["error_rate"],
            other["error_rate"] if other else None, pct, lower_better=True,
            help="Share of calls that failed. Lower is better. "
                 "See the breakdown by type below.")
    total = f" | total run cost ${summ['total_cost_usd']:.4f}" if summ else ""
    col.caption(f"Measured at concurrency {meta['concurrency']} | "
                f"wall-clock {meta['wall_clock_s']:.1f}s | "
                f"{len(meta['results'])} issues{total}")


st.header("Head-to-head: Model A vs Model B")
st.caption("Green ▲ / red ▼ marks which model wins each metric, versus the other.")
col_a, col_b = st.columns(2)
op_metrics_block(col_a, model_a, summ_a, summ_b, meta_a)
op_metrics_block(col_b, model_b, summ_b, summ_a, meta_b)

if summ_a and summ_b:
    err_a = summ_a["error_breakdown"] or {"none": 0}
    err_b = summ_b["error_breakdown"] or {"none": 0}
    with st.expander("Error breakdown by type (rate limit / timeout / parse / other)"):
        st.write({model_a: err_a, model_b: err_b})


# --------------------------------------------------------------------------- #
# Tabs: scored vs unscored
# --------------------------------------------------------------------------- #
scored_tab, unscored_tab = st.tabs(
    ["Scored view (vs ground truth)", "Unscored view (no ground truth)"]
)

# ---- Scored view --------------------------------------------------------- #
with scored_tab:
    if not (summ_a and summ_b):
        st.info("Both models need a price on file to show the scored view.")
    else:
        st.subheader("Per-class precision / recall / F1")

        def prf_df(summ):
            rows = []
            for label in LABELS:
                pc = summ["per_class"][label]
                rows.append({
                    "label": label,
                    "precision": round(pc["precision"], 3),
                    "recall": round(pc["recall"], 3),
                    "f1": round(pc["f1"], 3),
                    "support": pc["support"],
                })
            return pd.DataFrame(rows).set_index("label")

        c1, c2 = st.columns(2)
        c1.caption(model_a)
        c1.dataframe(prf_df(summ_a), width='stretch')
        c2.caption(model_b)
        c2.dataframe(prf_df(summ_b), width='stretch')

        st.subheader("Confusion matrices (rows = true label, columns = predicted by model)")
        gt_counts = {label: 0 for label in LABELS}
        for actual in truth.values():
            if actual in gt_counts:
                gt_counts[actual] += 1
        st.caption("Actual labels in the ground truth (each row totals to these): "
                   + ", ".join(f"{label} {gt_counts[label]}" for label in LABELS)
                   + f" — {sum(gt_counts.values())} issues in all.")

        def cm_df(summ):
            m = summ["confusion_matrix"]
            return pd.DataFrame(
                [[m[t][p] for p in LABELS] for t in LABELS],
                index=[f"true:{l}" for l in LABELS],
                columns=[f"pred:{l}" for l in LABELS],
            )

        c1, c2 = st.columns(2)
        c1.caption(model_a)
        c1.dataframe(cm_df(summ_a), width='stretch')
        c2.caption(model_b)
        c2.dataframe(cm_df(summ_b), width='stretch')

        st.subheader("Where the models disagreed (scored issues)")
        st.caption("Ground truth is shown so you can see which model was right.")
        disagreements = []
        for num, correct in truth.items():
            ra, rb = by_num_a.get(num), by_num_b.get(num)
            if not ra or not rb:
                continue
            if ra.label != rb.label:
                issue = issues.get(num, {})
                disagreements.append({
                    "issue": num,
                    "title": (issue.get("title") or "")[:70],
                    model_a: ra.label,
                    model_b: rb.label,
                    "ground_truth": correct,
                    "A_right": ra.label == correct,
                    "B_right": rb.label == correct,
                })
        if disagreements:
            st.caption(f"{len(disagreements)} scored issues where A and B differ")
            st.dataframe(pd.DataFrame(disagreements), width='stretch',
                         hide_index=True)
        else:
            st.success("The two models agreed on every scored issue.")


# ---- Unscored view ------------------------------------------------------- #
with unscored_tab:
    unscored_nums = [str(i) for i in issues if str(i) not in truth]
    both = [n for n in unscored_nums if n in by_num_a and n in by_num_b
            and by_num_a[n].label and by_num_b[n].label]

    # Headline: agreement rate.
    agree = sum(1 for n in both if by_num_a[n].label == by_num_b[n].label)
    rate = agree / len(both) if both else 0.0
    st.subheader("Agreement rate on unlabeled issues")
    st.metric(f"{model_a} vs {model_b} agree",
              f"{rate*100:.1f}%", help=f"{agree} of {len(both)} unlabeled issues")

    # Per-class distribution of each model's suggestions.
    st.subheader("What each model suggests (label distribution, unlabeled issues)")

    def dist(by_num):
        counts = {label: 0 for label in LABELS}
        for n in unscored_nums:
            r = by_num.get(n)
            if r and r.label:
                counts[r.label] += 1
        return counts

    dist_df = pd.DataFrame({model_a: dist(by_num_a), model_b: dist(by_num_b)})
    st.bar_chart(dist_df)

    # Disagreement filter + raw output.
    st.subheader("Browse issues")
    only_disagree = st.checkbox("Show only issues where the models disagree",
                                value=True)
    view_nums = [n for n in both
                 if (by_num_a[n].label != by_num_b[n].label) or not only_disagree]
    st.caption(f"Showing {len(view_nums)} issues")

    table = []
    for n in view_nums[:300]:   # cap the table for responsiveness
        issue = issues.get(n, {})
        table.append({
            "issue": n,
            "title": (issue.get("title") or "")[:70],
            model_a: by_num_a[n].label,
            model_b: by_num_b[n].label,
        })
    st.dataframe(pd.DataFrame(table), width='stretch', hide_index=True)

    # Raw model output for a chosen issue.
    st.subheader("Raw model output for one issue")
    if view_nums:
        pick = st.selectbox("Issue number", view_nums)
        issue = issues.get(pick, {})
        st.markdown(f"**{issue.get('title','')}**")
        st.caption(issue.get("url", ""))
        c1, c2 = st.columns(2)
        c1.caption(f"{model_a} -> {by_num_a[pick].label}")
        c1.code(by_num_a[pick].raw_output or "(empty)")
        c2.caption(f"{model_b} -> {by_num_b[pick].label}")
        c2.code(by_num_b[pick].raw_output or "(empty)")
