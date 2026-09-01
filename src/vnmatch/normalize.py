"""Vietnamese hardware product-name normalisation.

Shop staff type product names the way they speak them: no diacritics, heavy
abbreviation, `*` instead of `x`, millimetre marks dropped.  Everything in this
module maps those surface forms onto one canonical string so that lexical
baselines and the embedding model see the same token stream.
"""

from __future__ import annotations

import re
import unicodedata

# Abbreviations shop staff actually use, longest first so that "bu long" is
# rewritten before the bare "long" (a washer, "long den") can match inside it.
ABBREVIATIONS: dict[str, str] = {
    "bulong": "bu long",
    "bl": "bu long",
    "lgn": "luc giac ngoai",
    "lgc": "luc giac chim",
    "lg": "luc giac",
    "ldp": "long den phang",
    "ldv": "long den veng",
    "longden": "long den",
    "ld": "long den",
    "tacke": "tac ke",
    "tk": "tac ke",
    "i304": "inox 304",
    "i201": "inox 201",
    "i316": "inox 316",
    "sus304": "inox 304",
    "sus201": "inox 201",
    "inoc": "inox",
    "vit": "vit",
    "dn": "duong kinh",
    "ma kem": "ma kem",
    "mk": "ma kem",
}

# Units of measure as they appear in the catalogue, plus common shorthands.
UNIT_ALIASES: dict[str, str] = {
    "cai": "cai",
    "c": "cai",
    "con": "con",
    "kg": "kg",
    "kilo": "kg",
    "bo": "bo",
    "cuon": "cuon",
    "cay": "cay",
    "to": "to",
    "hop": "hop",
    "vien": "vien",
    "met": "met",
    "m": "met",
    "lon": "lon",
    "lo": "lo",
    "xau": "xau",
    "chai": "chai",
    "thung": "thung",
    "bich": "bich",
}

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9x]+")
# Only collapse a separator sitting between two digits: the bare `x` inside
# "inox" must not be treated as a multiplication sign.
_DIM_SEPARATOR = re.compile(r"(?<=\d)\s*[x*×]\s*(?=\d)", re.IGNORECASE)
# A dimension run such as 10x50x17, or a single measurement such as "m8".
_DIMENSION = re.compile(r"\b\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?)+\b")
_METRIC_THREAD = re.compile(r"\bm\s*(\d+(?:\.\d+)?)\b")


def strip_diacritics(text: str) -> str:
    """Fold Vietnamese diacritics away, keeping `d` and `đ` distinguishable."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def expand_abbreviations(text: str) -> str:
    """Rewrite shorthand tokens to their full form, longest pattern first."""
    for short in sorted(ABBREVIATIONS, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(short)}\b", ABBREVIATIONS[short], text)
    return text


def normalize(text: str, *, expand: bool = True) -> str:
    """Canonical form used by every scorer in this project.

    Lowercases, drops diacritics, unifies dimension separators onto `x`, and
    optionally expands abbreviations.
    """
    text = strip_diacritics(str(text).lower())
    text = _DIM_SEPARATOR.sub("x", text)
    # `m8` and `m 8` are the same thread size; write it as one token.
    text = _METRIC_THREAD.sub(r"m\1", text)
    text = _NON_ALNUM.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if expand:
        text = expand_abbreviations(text)
        text = _WHITESPACE.sub(" ", text).strip()
    return text


def normalize_unit(unit: str) -> str:
    """Map a unit of measure onto its canonical spelling."""
    key = normalize(unit, expand=False)
    return UNIT_ALIASES.get(key, key)


def extract_dimensions(text: str) -> list[str]:
    """Pull dimension runs (`10x50x17`) and metric threads (`m8`) out of a name.

    These carry most of the discriminative power in a hardware catalogue and are
    exactly what embedding models handle worst, so the hybrid scorer treats them
    as a separate signal rather than trusting the encoder with them.
    """
    canonical = normalize(text, expand=False)
    found = _DIMENSION.findall(canonical)
    found.extend(f"m{m}" for m in _METRIC_THREAD.findall(canonical))
    return found


def extract_material(text: str) -> str | None:
    """Return the material grade (`inox 304`, `ma kem`, ...) when stated."""
    canonical = normalize(text)
    for grade in ("inox 304", "inox 316", "inox 201", "inox", "ma kem", "sat xi", "sat"):
        if grade in canonical:
            return grade
    return None
