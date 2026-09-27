"""
Per-token prices for each model, in USD per 1,000,000 tokens.

WHY THIS FILE IS SEPARATE AND EXPLICIT:
    The dollar figure must be traceable: token counts and per-token rates should
    visibly produce the cost. So prices live here as plain numbers with a source
    and a date - not buried in code.

KEEPING PRICES ACCURATE:
    These come from the DigitalOcean Serverless Inference pricing page. Confirm
    them in your DO console and update PRICING_SOURCE_DATE when they change.
    If a price is wrong, every dollar number downstream is wrong - so this is the
    one table to double-check.

    (input, output) = USD per 1,000,000 tokens.
"""

PRICING_SOURCE = "DigitalOcean Serverless Inference pricing page"
PRICING_SOURCE_DATE = "2026-09-27"  # <-- re-confirm when prices change

PRICING = {
    # model id                       input $/M   output $/M
    "mistral-3-14B":                 {"input": 0.20, "output": 0.20},   # "Ministral 3 14B"
    "gemma-4-31B-it":                {"input": 0.18, "output": 0.50},   # "Gemma 4"
    "qwen3.5-397b-a17b":             {"input": 0.55, "output": 3.50},   # "Qwen 3.5 397B A17B"
    "arcee-trinity-large-thinking":  {"input": 0.25, "output": 0.90},   # reasoning, public preview
    "openai-gpt-oss-20b":            {"input": 0.05, "output": 0.45},
    "openai-gpt-oss-120b":           {"input": 0.10, "output": 0.70},
    # --- added 2026-09-27: wider comparison pool (all confirmed from DO console) ---
    "deepseek-3.2":                  {"input": 0.50, "output": 1.60},   # "Deepseek 3.2"
    "deepseek-4-flash":              {"input": 0.14, "output": 0.28},   # "Deepseek V4 Flash"
    "deepseek-v4.1-flash":           {"input": 0.30, "output": 1.20},   # "DeepSeek V4.1 Flash"
    "glm-5.2":                       {"input": 1.40, "output": 4.40},   # "GLM-5.2"
    "glm-5.3":                       {"input": 1.40, "output": 4.40},   # "GLM5.3"
    "glm-5.3-flash":                 {"input": 0.15, "output": 0.50},   # "GLM5.3 Flash"
    "llama-4-maverick":              {"input": 0.25, "output": 0.87},   # "Llama 4 Maverick"
    "nemotron-nano-12b-v2-vl":       {"input": 0.20, "output": 0.60},   # "Nemotron-nano 12b v2-vl"
}


def price_for(model):
    """Return (input_per_m, output_per_m) or raise a clear error if unknown/unset."""
    entry = PRICING.get(model)
    if entry is None:
        raise KeyError(
            f"No price on file for '{model}'. Add it to PRICING in pricing.py "
            f"(source: {PRICING_SOURCE})."
        )
    if entry["input"] is None or entry["output"] is None:
        raise ValueError(
            f"Price for '{model}' is a placeholder. Fill in the real numbers from "
            f"the DO console before running - the cost math depends on it."
        )
    return entry["input"], entry["output"]
