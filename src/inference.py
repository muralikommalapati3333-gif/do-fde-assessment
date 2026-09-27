"""
Classify ONE issue with ONE model, and record everything we need about the call.

Key design rules:

  - One call per issue, never batched. That preserves per-call cost, per-call
    latency, and individually-retryable failures - so one bad issue never sinks
    the whole run.

  - Every call returns prompt/completion tokens, latency, and (on failure) an
    error type tagged rate_limit / timeout / parse / other.

The DigitalOcean Serverless Inference API is OpenAI-compatible, so we use the
openai library and just point it at DO's base URL. Nothing DO-specific leaks
into the logic - the same code works against any OpenAI-compatible provider.
"""

import json
import os
import re
import time
from dataclasses import dataclass, asdict

from openai import OpenAI, APITimeoutError, RateLimitError, APIError

from labels import LABELS
from prompt import build_messages

# Where DO Serverless Inference lives, and the key. Both come from the
# environment (.env), never hardcoded - the key must never touch the repo/zip.
BASE_URL = os.environ.get("DO_INFERENCE_BASE_URL", "https://inference.do-ai.run/v1")

# Retry settings for flaky calls (rate limits / timeouts).
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 1.5   # wait 1.5s, then 3s, then 4.5s ... between retries

# We only need a label back. Keep generous headroom so reasoning models that
# "think" before answering don't get cut off mid-thought.
MAX_OUTPUT_TOKENS = 512
TEMPERATURE = 0.0            # deterministic -> reproducible runs


@dataclass
class CallResult:
    """Everything we record about a single classification call."""
    number: int
    model: str
    label: str | None          # parsed label, or None if the call failed
    raw_output: str            # exactly what the model returned (for the UI)
    prompt_tokens: int
    completion_tokens: int
    latency_s: float           # wall-clock time for this one call
    attempts: int              # how many tries it took (1 = first try worked)
    error: str | None          # error message, or None on success
    error_type: str | None     # rate_limit / timeout / parse / other, or None

    def as_dict(self):
        return asdict(self)


def make_client():
    """Build the OpenAI-compatible client pointed at DO Serverless Inference."""
    api_key = os.environ.get("DO_INFERENCE_KEY")
    if not api_key:
        raise RuntimeError(
            "DO_INFERENCE_KEY is not set. Put it in a .env file (never in code)."
        )
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def _parse_label(text):
    """Pull a valid label out of the model's reply. Returns None if we can't."""
    if not text:
        return None

    # First choice: it obeyed us and returned JSON like {"label": "bug"}.
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            candidate = str(data.get("label", "")).strip().lower()
            if candidate in LABELS:
                return candidate
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: scan the text for a label word. Reasoning models often state the
    # answer LAST, so we take the last label mentioned.
    found = re.findall(r"\b(" + "|".join(LABELS) + r")\b", text.lower())
    if found:
        return found[-1]

    return None


def classify_issue(client, model, issue):
    """Classify one issue. Always returns a CallResult (never raises)."""
    messages = build_messages(issue)
    number = issue["number"]

    error = None
    error_type = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            latency = time.perf_counter() - start

            raw = (response.choices[0].message.content or "").strip()
            usage = response.usage
            label = _parse_label(raw)

            if label is None:
                # The call worked but the answer was unusable. We do NOT retry:
                # temperature is 0, so a retry would return the identical garbage
                # and just waste a call. Record it as a parse error and move on.
                return CallResult(
                    number=number, model=model, label=None, raw_output=raw,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    latency_s=latency, attempts=attempt,
                    error="could not parse a valid label", error_type="parse",
                )

            return CallResult(
                number=number, model=model, label=label, raw_output=raw,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                latency_s=latency, attempts=attempt,
                error=None, error_type=None,
            )

        except RateLimitError as e:
            error, error_type = str(e), "rate_limit"
        except APITimeoutError as e:
            error, error_type = str(e), "timeout"
        except APIError as e:
            error, error_type = str(e), "other"
        except Exception as e:  # noqa: BLE001 - never let one issue crash the run
            error, error_type = str(e), "other"

        # Wait before the next attempt (linear backoff: 1.5s, 3s, 4.5s). All
        # transient errors (rate_limit / timeout / other) are retried the same way.
        if attempt < MAX_ATTEMPTS:
            time.sleep(BACKOFF_BASE_SECONDS * attempt)

    # All attempts failed.
    return CallResult(
        number=number, model=model, label=None, raw_output="",
        prompt_tokens=0, completion_tokens=0, latency_s=0.0,
        attempts=MAX_ATTEMPTS, error=error, error_type=error_type,
    )
