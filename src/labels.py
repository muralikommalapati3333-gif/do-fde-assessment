"""
The label rulebook (our classification schema).

Why this file exists:
    "bug vs enhancement vs question" sounds obvious but isn't. Is "the docs are
    wrong" a bug or a documentation issue? Is "can you add feature X?" an
    enhancement or a question? If we don't pin this down, our answer key AND the
    model's instructions become fuzzy, and we'd be grading against a moving
    target.

    The SAME definitions here are reused in two places:
      1. building the answer key we grade against
      2. the instructions we give the model
    so we always grade the model against the exact standard we asked it to follow.
"""

# The six labels, exactly as the customer's schema defines them.
LABELS = ["bug", "enhancement", "question", "documentation", "security", "other"]

# Plain-language definition of each label + tie-break rules.
# (This text is also fed to the model later, so keep it clear and short.)
LABEL_DEFINITIONS = {
    "bug": (
        "Something is broken or behaves wrong: a command errors, crashes, panics, "
        "returns wrong output, or does not do what it says it does."
    ),
    "enhancement": (
        "A request for a NEW feature or an improvement to an existing one. "
        "Includes feature requests and suggestions."
    ),
    "question": (
        "Someone asking how to do something or asking for clarification, "
        "without reporting a defect or requesting a new feature."
    ),
    "documentation": (
        "The issue is about the docs themselves being wrong, missing, or unclear "
        "- not about the code being broken."
    ),
    "security": (
        "A security vulnerability or concern: credential exposure, CVE, unsafe "
        "behavior, or anything with security impact."
    ),
    "other": (
        "Does not fit the five above: spam, a duplicate, off-topic, or too "
        "vague/ambiguous to classify."
    ),
}

# Tie-break rules for the tricky overlaps (also given to the model).
TIE_BREAK_RULES = [
    "If it reports something broken AND asks a question, choose 'bug' (the defect wins).",
    "A feature request phrased as a question ('can you add X?') is 'enhancement'.",
    "If the ask is to fix wrong/missing docs, choose 'documentation'; if the code "
    "itself is wrong, choose 'bug'.",
    "If there is not enough information to decide, choose 'other'.",
]

# How maintainer labels (messy, from GitHub) translate to OUR schema.
# This is only a HINT to speed up human review - not treated as ground truth.
# Anything not in this map (workflow tags like 'hacktoberfest', 'good first issue',
# 'snap', 'windows', etc.) gives no hint and must be labeled by hand.
MAINTAINER_LABEL_TO_SCHEMA = {
    "bug": "bug",
    "suggestion": "enhancement",
    "enhancement": "enhancement",
    "security vulnerability": "security",
    "question": "question",
    "troubleshooting": "question",
    "docs": "documentation",
    "documentation": "documentation",
}

# When an issue has several maintainer labels, which hint wins? Higher = stronger.
# e.g. a 'security vulnerability' + 'bug' issue should hint 'security'.
HINT_PRIORITY = ["security", "bug", "documentation", "question", "enhancement"]


def hint_from_maintainer_labels(maintainer_labels):
    """Return a best-guess schema label from messy maintainer labels, or '' if none."""
    mapped = set()
    for name in maintainer_labels:
        schema = MAINTAINER_LABEL_TO_SCHEMA.get(name.lower())
        if schema:
            mapped.add(schema)
    if not mapped:
        return ""
    # Pick the highest-priority hint among the mapped ones.
    for candidate in HINT_PRIORITY:
        if candidate in mapped:
            return candidate
    return ""
