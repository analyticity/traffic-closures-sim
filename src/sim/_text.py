"""Shared text normalization helpers used across the simulation pipeline."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def strip_diacritics(text: Any) -> str:
    """Remove combining diacritical marks (accents) from *text*."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(ch)
    )


def norm_name(value: Any, *, keep_slash: bool = False) -> str:
    """Lowercase, strip diacritics/punctuation, collapse whitespace.

    Parameters
    ----------
    keep_slash : bool
        If *True*, ``/`` is preserved (useful for road refs like ``I/43``).
    """
    if value is None:
        return ""
    text = str(value).strip()
    text = strip_diacritics(text).lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    pattern = r"[^\w\s\-/]" if keep_slash else r"[^\w\s\-]"
    text = re.sub(pattern, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_col(name: Any) -> str:
    """Normalize a column name to ``snake_case`` ASCII."""
    text = strip_diacritics(name).strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")
