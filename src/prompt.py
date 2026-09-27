"""
Build the prompt we send to the model for ONE issue.

Design choices:

1. Same rulebook as the answer key.
   The definitions + tie-break rules here are imported from labels.py - the exact
   text we used to build ground truth. We grade the model against the standard we
   actually gave it. No moving target.

2. Force a single word out, as JSON.
   We ask the model to reply with JSON like {"label": "bug"}. JSON is easy to
   parse reliably and lets us detect when a model rambles instead of answering.

3. Low temperature (set in inference.py, not here).
   Classification wants the most likely answer, not creativity. Temperature 0
   also makes runs reproducible.

4. Truncate the issue body.
   Some issues paste huge logs. We cap the body so a few giant issues don't blow
   up cost and latency. The cap keeps token cost predictable.
"""

from labels import LABELS, LABEL_DEFINITIONS, TIE_BREAK_RULES

# Cap the issue body length (characters). Long enough to capture the real content
# of almost every doctl issue, short enough to keep token cost predictable.
MAX_BODY_CHARS = 4000


def _build_system_prompt():
    """The fixed instructions: who the model is and the exact rules to follow."""
    lines = [
        "You are a triage assistant that classifies GitHub issues for the doctl",
        "repository into exactly ONE label. Use only these six labels:",
        "",
    ]
    for label in LABELS:
        lines.append(f"- {label}: {LABEL_DEFINITIONS[label]}")

    lines += ["", "Tie-break rules when an issue could fit more than one label:"]
    for rule in TIE_BREAK_RULES:
        lines.append(f"- {rule}")

    lines += [
        "",
        "Reply with ONLY a JSON object in this exact form, and nothing else:",
        '{"label": "<one of: ' + ", ".join(LABELS) + '>"}',
    ]
    return "\n".join(lines)


SYSTEM_PROMPT = _build_system_prompt()


def build_messages(issue):
    """Return the OpenAI-style [system, user] messages for one issue."""
    title = issue.get("title") or ""
    body = (issue.get("body") or "")[:MAX_BODY_CHARS]

    user_content = (
        f"Issue title: {title}\n\n"
        f"Issue body:\n{body}\n\n"
        "Classify this issue into exactly one label."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
