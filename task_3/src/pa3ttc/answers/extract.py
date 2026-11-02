"""Answer extraction, exactly the rule in ATML-PA3-Data/README.MD (used for the live generations; the stored
rollouts already carry `extracted_answer` made with the same rule for MATH):

    find the last literal "\\boxed{", scan balanced braces (nested LaTeX preserved), strip surrounding whitespace;
    an absent, empty or unclosed last box gives None; there is no numeric fallback.
"""
from __future__ import annotations

OPEN = "\\boxed{"


def last_boxed(text: str | None) -> str | None:
    if not text:
        return None
    i = text.rfind(OPEN)
    if i < 0:
        return None
    start = i + len(OPEN)
    depth = 1
    for j in range(start, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                ans = text[start:j].strip()
                return ans or None
    return None
