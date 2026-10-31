"""Last \\boxed{...} answer with balanced braces (the Hendrycks-MATH `last_boxed_only_string` rule, plus the
`\\boxed 5` space form).  An unclosed last box (e.g. cut off by the token cap) gives None."""
from __future__ import annotations


def last_boxed(text: str | None) -> str | None:
    if not text:
        return None
    idx = text.rfind("\\boxed")
    cmd = "\\boxed"
    if idx < 0:
        idx = text.rfind("\\fbox")
        cmd = "\\fbox"
        if idx < 0:
            return None
    i = idx + len(cmd)
    while i < len(text) and text[i] in " \t":
        i += 1
    if i < len(text) and text[i] == "{":
        depth = 0
        for j in range(i, len(text)):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    ans = text[i + 1:j].strip()
                    return ans if ans else None
        return None
    if i > idx + len(cmd):   # "\boxed 5" form: up to whitespace or '$'
        j = i
        while j < len(text) and text[j] not in " \t\n$":
            j += 1
        ans = text[i:j].strip()
        return ans if ans else None
    return None
