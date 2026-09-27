# Choosing a cheaper model to triage doctl issues

**Live app:** https://do-fde-assessment-app-ovafs.ondigitalocean.app

**Short version of my recommendation:** run **`deepseek-4-flash`** for everyday
triage, and keep **`gemma-4-31B-it`** on hand as a backup that runs on separate
infrastructure. The reasoning, and the numbers I used to get there, are below.

This README is a walk-through of how I thought about the problem and what I found.
It is not a setup manual. The build-and-run steps are near the bottom, kept short,
just enough that you could repeat my run if you wanted to.

## The situation

A customer sorts a lot of GitHub issues into categories, and right now they pay for
a top-tier model to do it. Sorting issues is a narrow job: each issue has one correct
bucket, and there are only six of them (`bug`, `enhancement`, `question`,
`documentation`, `security`, `other`). That is usually the kind of work a smaller,
cheaper model can handle well. So the question I set out to answer was a simple one:
can a cheaper model hit the accuracy they need, and can I show it with numbers instead
of a guess?

I used DigitalOcean's `doctl` repository as the test case, and DigitalOcean Serverless
Inference to call every model.

## How I set it up

**The issues.** I pulled every issue from the public `doctl` repo through the GitHub
API. That came to 536 issues once I dropped the pull requests the API quietly mixes in.
I saved that list to a file once and reused it for every model, so each model is judged
on the exact same set of issues.

**The answer key was the hard part.** The labels maintainers had already put on issues
looked like an easy answer key, but they are not reliable. Different people added them
over several years and they do not always agree, which the exercise itself points out.
So I built my own smaller answer key by hand:

- I used the maintainer labels only as a starting hint, mapping their messy tags onto
  my six buckets so I had something to sort by.
- I picked 133 issues on purpose instead of at random. A random grab from `doctl` would
  be almost all bugs and feature requests, with barely any security or documentation
  issues, and then I would have nothing to measure the rare buckets against. So I
  deliberately pulled in more of the rare ones.
- For each of the 133, an AI suggested a label using the same written rules I had set
  down, and then I checked every one myself. The AI doing the suggesting is not one of
  the models I was testing, so nothing grades its own work.
- 13 issues were genuinely tricky, or the maintainer's label disagreed with my rules
  (for example, an issue titled `[QUESTION]` that someone had tagged `bug`). I decided
  each of those myself and wrote down the reason.

One thing I want to be honest about: `doctl` barely has any documentation or "other"
issues (3 and 2 in my set). Those two buckets have too few examples for their scores
to mean much, so I lean on the four buckets that have enough data, and I say so rather
than dress up a number built on two examples.

The definitions and the tie-break rules live in one file and get used in two places:
once when I built the answer key, and again inside the prompt the model sees. That way
the model is graded against the exact rules I asked it to follow, not a moving target.

## How I judged each model

I kept the scoring plain on purpose, so the numbers are hard to argue with.

Every issue is one API call, never batched. A real triage system needs the cost and
the response time of each single call, needs to retry one failed issue on its own, and
needs the option to send a hard issue to a better model. Batching throws all of that
away.

I score by exact match against my answer key. I did not use an "AI that grades AI"
here, even though DigitalOcean offers one. That tool earns its keep when there is no
single right answer, like open-ended writing. Here I already have a correct answer for
every issue, so a judge would only add its own mistakes and its own cost on top.

I ran everything at temperature 0 so the same issue gives the same answer and the run
can be repeated. (One small caveat, since I would rather report it than hide it:
inference across a cluster is not perfectly repeatable. One borderline issue flipped
between two of my runs, which moved a top model by a fraction of a percent. It did not
change the recommendation.)

I also counted failures honestly. If a call errored out, or came back as something I
could not read, I did not quietly drop it. I report accuracy two ways: over every issue
(where failures count against the model) and over only the calls that returned a usable
label. The gap between those two numbers is a good measure of how flaky a model is.

Cost is just token count times the published per-token price, worked out in one place,
with the prices written down and dated in the repo so the dollar figures are easy to
check.

## Which models I tried

DigitalOcean's catalog lists around 78 model endpoints, but most are not a fit for this
job. Roughly 20 are for something else entirely (embeddings, images, audio, reranking).
Of the ~58 chat models, about 35 are the expensive top-tier ones, which is exactly what
the customer is trying to move away from. That left about 23 open-weight chat models
hosted by DigitalOcean as the realistic pool. I ran 12 of them all the way through,
chosen to cover a range of sizes, prices, and designs rather than near-duplicates of
each other.

| Model | Accuracy | acc (usable calls only) | Cost / 1,000 | Cost / correct | p50 | p95 | Err % |
|---|---|---|---|---|---|---|---|
| `deepseek-3.2` | **89.5%** | 89.5% | $0.365 | $0.000408 | 1.84s | 5.28s | 0.2% |
| `deepseek-4-flash` | 88.0% | 88.0% | **$0.101** | **$0.000115** | 1.94s | 4.87s | 0.0% |
| `gemma-4-31B-it` | 88.0% | 88.0% | $0.145 | $0.000165 | 2.39s | 5.01s | 0.0% |
| `glm-5.3-flash` | 88.0% | 88.0% | $0.108 | $0.000122 | 1.76s | 4.83s | 0.2% |
| `glm-5.3` | 88.0% | 88.0% | $1.007 | $0.001145 | 1.73s | 4.31s | 0.2% |
| `llama-4-maverick` | 88.0% | 88.0% | $0.193 | $0.000220 | 1.27s | 7.60s | 0.0% |
| `openai-gpt-oss-120b` | 87.2% | 87.2% | $0.141 | $0.000162 | 2.36s | 4.66s | 0.2% |
| `deepseek-v4.1-flash` | 85.7% | 91.2% | $0.349 | $0.000407 | 2.74s | 11.89s | 5.8% |
| `mistral-3-14B` | 85.7% | 85.7% | $0.150 | $0.000175 | 1.07s | 3.85s | 0.2% |
| `glm-5.2` | 85.0% | 91.1% | $1.535 | $0.001806 | 2.02s | 6.05s | 7.6% |
| `openai-gpt-oss-20b` | 85.0% | 85.0% | $0.091 | $0.000108 | 2.00s | 4.84s | 1.1% |
| `nemotron-nano-12b-v2-vl`\* | 81.2% | 87.1% | $0.360 | $0.000443 | 3.77s | 6.47s | 6.8% |

Accuracy is measured on the 133-issue answer key. Cost and latency are measured over
the run. `acc (usable calls only)` ignores failed or unreadable calls, so comparing it
to the plain accuracy column shows how much a model's failures are hurting it. `p50` and
`p95` are the typical and near-worst-case response times; `Err %` is the share of calls
that failed.

\* `nemotron-nano-12b-v2-vl` was scored on the same answer key but only ran that subset,
because it crawls at about 0.4 issues per second — far too slow for high volume. Every
other model ran the full set.

A few models I dropped before a full run, and why: `qwen3.5-397b-a17b`,
`arcee-trinity-large-thinking`, and `mimo-v2.5-pro` were reasoning or very large models
that took tens of seconds per issue, which rules them out for high volume; `kimi-k2.6`
forces its own temperature setting and breaks the repeatable run I wanted; and
`minimax-m2.5` and `qwen3.8-max` were simply unavailable when I tested.

## What the numbers showed

The most striking thing is that the top of the table is basically a tie. Six models
land between 88.0% and 89.5%, and on 133 issues that whole spread is about two answers.
That is inside the noise. Accuracy on its own cannot pick a winner here, so the real
decision comes down to cost and reliability.

The second column earns its place by catching models that only look good. `deepseek-v4.1-flash`
has the highest accuracy on the calls that worked (91.2%), but 5.8% of its calls failed
outright, which drags its real-world accuracy down to 85.7%. `glm-5.2` (7.6% failures)
and `nemotron` (6.8%) have the same problem. A big gap between those two accuracy columns
is a reliability warning sign.

The expensive models did not earn their price. `glm-5.2` and `glm-5.3` cost roughly 7 to
15 times more than the cheap models for the same accuracy or worse. And the one reasoning
model I could run to completion added latency and cost without adding accuracy, which fits
what you would expect: step-by-step "thinking" pays off on hard multi-step problems, not
on sorting an issue into one of six buckets.

## The two models I recommend

**Everyday model: `deepseek-4-flash`.** It tied for the best accuracy in the group at
88%, it was the cheapest of the accurate models at about $0.10 per 1,000 issues, and it
did not fail a single call. It is also fast. For the bulk of the traffic there is no good
reason to run anything more expensive.

**Backup model: `gemma-4-31B-it`.** Same 88% accuracy, also zero failed calls, still cheap
at about $0.15 per 1,000. The reason to keep it is not extra accuracy. It is that it comes
from a different company (Google) and runs independently of DeepSeek. If DeepSeek has a bad
day, an outage or a quiet drop in quality, triage keeps running on a model that is just as
good. For a system meant to run unattended, a second equally-good model on separate
infrastructure is worth the small extra cost.

If accuracy on the hardest issues ever matters more than cost, `deepseek-3.2` was the
single most accurate model at 89.5%. But it costs about 3.6 times the everyday model for
barely more than one extra correct answer out of 133. So I would send it only the genuinely
unclear issues, not the whole stream.

**The trade-off, plainly.** Because six models were tied on accuracy, I let cost and
reliability make the call. `deepseek-4-flash` is the cheapest of the accurate models and
the only one in that top group that never failed a call, so it wins the everyday slot on
merit. Choosing `gemma-4-31B-it` as the second model trades a few cents per thousand issues
for the safety of not depending on a single provider. That felt like the right trade for
something running in production.

## Rolling this out beyond doctl

This exercise proves the method on one repo. To take it wider I would:

1. Build a small hand-checked answer key for each product area. The stratified,
   human-verified approach carries over directly, and each area needs its own because
   the mix of issue types differs.
2. Route by confidence. Send the bulk to the cheap everyday model, and send the
   low-confidence or high-stakes cases (anything labeled `security`, say) to the more
   accurate model or to a person. The one-call-per-issue design already allows this.
3. Give the weak buckets a fallback. The eval shows exactly where each model is shaky;
   those buckets get a backup model or a human, not blind trust.
4. Watch for drift. Re-score a fresh sample now and then, because both the issues and
   the models change over time.

For that to hold up you need a maintained answer key per area, a confidence signal good
enough to route on, and an agreed error budget for each bucket.

## Build and run

The app reads results that are already saved in the repo, so you can browse it without
spending any inference credits. You only need an API key if you want to run the models
again yourself.

```bash
pip install -r requirements.txt

cp .env.example .env      # then put your key in DO_INFERENCE_KEY
python -m streamlit run app.py
```

To reproduce the data and the models run from scratch:

```bash
python src/ingest.py                                        # fetch and freeze the issues
python src/build_gold_set.py && python src/prelabel_gold_set.py
python src/finalize_ground_truth.py                         # the human-checked answer key
python src/run_comparison.py                                # classify with each model
python src/report.py                                        # print the comparison table
```

As a container:

```bash
docker build -t doctl-eval .
docker run -p 8080:8080 -e DO_INFERENCE_KEY=your_key_here doctl-eval
```

**Environment variables:**

- `DO_INFERENCE_KEY` (required only to call models live) — your DigitalOcean Serverless
  Inference API key. It is read at runtime and never written into the image or the repo.
- `DO_INFERENCE_BASE_URL` (optional) — defaults to `https://inference.do-ai.run/v1`.
- `INFERENCE_CONCURRENCY` (optional) — how many calls to run at once, default 8. Latency
  and throughput in the results are always reported at the concurrency they were measured at.

## What I chose not to do, and why

- **No AI-as-a-judge**, because the task has a real answer key and a judge would only add
  its own error and cost.
- **No batching**, because per-issue cost, retries, and routing all depend on one call per
  issue.
- **No fine-tuning.** The goal was to find an off-the-shelf model that is good enough.
  Fine-tuning is a lever for later, if the accuracy ceiling here ever turns out to be too low.
- **No heavy ingestion pipeline.** The exercise scopes it thin, so I kept it thin.
