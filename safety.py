"""Report safety guard.

A hospital-facing demo must never read as a treatment directive. This module
flags over-strong phrasing and softens it to research-use language. It runs on
the LLM output before rendering, and (importantly) also on the deterministic
fallback so both paths are safe.
"""
import re

# (pattern, replacement). Patterns are case-insensitive.
REWRITES = [
    (r"\bthe patient should (?:receive|be given|take)\b",
     "SIDISH prioritizes for validation"),
    (r"\b(?:the )?best treatment is\b",
     "the strongest candidate hypothesis is"),
    (r"\bwe recommend treating\b", "SIDISH prioritizes for validation"),
    (r"\b(?:will|can) (?:cure|treat|heal)\b", "may, pending validation, address"),
    (r"\bguarantee(?:s|d)?\b", "suggests (requires validation)"),
    (r"\bclinically proven\b", "computationally prioritized"),
    (r"\bproven (?:to be )?effective\b", "predicted, pending validation, to reduce risk"),
    (r"\bdefinitive(?:ly)? treatment\b", "candidate therapeutic hypothesis"),
]

# Phrases that should never appear; if found after rewrite, we raise so a human
# reviews before the report ships.
HARD_BLOCK = [
    r"\bdiagnos(?:e|is) confirmed\b",
    r"\bstop (?:all )?other (?:drugs|treatment)\b",
    r"\bno further treatment needed\b",
]

DISCLAIMER = (
    "Prepared to support molecular tumor board discussion and clinical decision-making. "
    "SIDISH findings are computational predictions and should be confirmed with appropriate "
    "diagnostics and reviewed by the treating physician before clinical action."
)


def sanitize(text: str) -> str:
    """Soften over-strong claims. Returns the cleaned text."""
    out = text
    for pat, repl in REWRITES:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def audit(text: str) -> list[str]:
    """Return a list of hard-block hits (empty means clean)."""
    hits = []
    for pat in HARD_BLOCK:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(pat)
    return hits


def guard(sections: dict) -> dict:
    """Sanitize every section; raise if a hard-blocked phrase survives."""
    cleaned = {k: sanitize(v) for k, v in sections.items()}
    all_text = " ".join(cleaned.values())
    hits = audit(all_text)
    if hits:
        raise ValueError(
            f"Report contains phrasing that needs human review: {hits}. "
            "Not shipping automatically."
        )
    return cleaned
