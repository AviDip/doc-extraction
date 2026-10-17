"""Parse monetary amounts from OCR words.

Handles both separator conventions ("1,234.56" and "1.234,56"), thousands
groups without decimals ("45,000" in CORD's Rupiah receipts), currency
symbols, and trailing or bracketed negatives. Dates, percentages and codes
return None.
"""

from __future__ import annotations

import re

import numpy as np

from .schema import Document

_EDGE_NOISE = re.compile(r"^[^\d(+-]+|[^\d)-]+$")  # currency symbols, letters, colons at the edges
_NUMBER = re.compile(r"^[-+(]?\d[\d.,' ]*[)-]?$")
_LEADING_GROUP = re.compile(r"^[-+(]?\d{1,3}$")
_THOUSANDS_TAIL = re.compile(r"^\d{3}(?:[.,]\d{1,2})?\D*$")


def parse_amount(text: str) -> float | None:
    if "%" in text or "/" in text or ":" in text.strip(": "):
        return None
    s = _EDGE_NOISE.sub("", text.strip())
    if not s or not _NUMBER.match(s):
        return None

    negative = s.startswith("-") or s.endswith("-") or (s.startswith("(") and s.endswith(")"))
    s = s.strip("+-()").replace("'", "").replace(" ", "")
    if not s or not s[0].isdigit() or not s[-1].isdigit():
        return None

    separators = [c for c in s if c in ".,"]
    if not separators:
        number = s
    elif len(set(separators)) == 2:
        decimal = "." if s.rfind(".") > s.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        integer, _, fraction = s.rpartition(decimal)
        if not _valid_groups(integer, thousands):
            return None
        number = integer.replace(thousands, "") + "." + fraction
    else:
        sep = separators[0]
        parts = s.split(sep)
        if len(parts) > 2 or len(parts[-1]) == 3:
            # one or more thousands separators, e.g. "45,000" or "1.234.567"
            if not _valid_groups(s, sep):
                return None
            number = s.replace(sep, "")
        else:
            number = parts[0] + "." + parts[1]
    try:
        value = float(number)
    except ValueError:
        return None
    return -value if negative else value


def _valid_groups(integer: str, sep: str) -> bool:
    groups = integer.split(sep)
    return 1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])


def word_values(doc: Document) -> np.ndarray:
    """Numeric value of each word, 0 where the word is not an amount.

    OCR often splits "1 234,56" into two words. When a 1-3 digit word is
    followed on the same line by a word starting with a 3-digit group, the
    pair is parsed together and the value is placed on the second word,
    so summing values over a labelled span still gives the right amount.
    """
    values = np.zeros(len(doc), dtype=np.float32)
    for i, word in enumerate(doc.words):
        value = parse_amount(word)
        values[i] = 0.0 if value is None else value

    for i in range(len(doc) - 1):
        if (
            doc.lines[i] == doc.lines[i + 1]
            and _LEADING_GROUP.match(doc.words[i])
            and _THOUSANDS_TAIL.match(doc.words[i + 1])
            and _horizontal_gap(doc.boxes[i], doc.boxes[i + 1]) < doc.boxes[i, 3] - doc.boxes[i, 1]
        ):
            merged = parse_amount(doc.words[i] + doc.words[i + 1])
            if merged is not None:
                values[i], values[i + 1] = 0.0, merged
    return values


def span_value(doc: Document, values: np.ndarray, start: int, end: int) -> float:
    """Value of an entity: the sum of its word values."""
    return float(values[start:end].sum())


def _horizontal_gap(left: np.ndarray, right: np.ndarray) -> float:
    return float(right[0] - left[2])
