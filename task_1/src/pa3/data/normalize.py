"""Text normalisation for GSM8K-test decontamination (approved: NFKC, lowercase, collapse ws)."""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize_text(s: str | None) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    return _WS.sub(" ", s).strip()
