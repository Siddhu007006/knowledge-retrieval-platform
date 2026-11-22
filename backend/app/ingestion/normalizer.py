"""Text normalization applied to all parsed documents before chunking.

Goals:
  - collapse excessive whitespace
  - normalize unicode dashes / quotes
  - remove control characters
  - preserve paragraph boundaries (double newline)
"""

import re
import unicodedata

_WS_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    lines = text.split("\n")
    normalized_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        normalized_lines.append(stripped)
    text = "\n".join(normalized_lines)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()
