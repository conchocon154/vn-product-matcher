"""Generate realistic noisy queries from clean catalogue names.

The shop has no log of what staff typed, so the training and evaluation queries
are synthesised.  Every corruption here was chosen from how the names actually
degrade on handwritten order slips and in the invoice line items: shorthand,
missing diacritics, `*` for `x`, partial dimensions, and ordinary typos.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .normalize import normalize, strip_diacritics

# Full form -> the shorthand staff write.  Applied to clean names to make them
# messy, which is the opposite direction to normalize.ABBREVIATIONS.
SHORTHAND: dict[str, tuple[str, ...]] = {
    "bu lông": ("bl", "bulong", "bu long"),
    "lục giác ngoài": ("lgn", "lg ngoai", "luc giac ngoai"),
    "lục giác chìm": ("lgc", "lg chim", "luc giac chim"),
    "lục giác": ("lg", "luc giac"),
    "long đen phẳng": ("ldp", "long den phang"),
    "long đen": ("ld", "long den"),
    "inox 304": ("i304", "inox304", "sus304", "inox 304"),
    "inox 201": ("i201", "inox201", "inox 201"),
    "inox 316": ("i316", "inox316"),
    "mạ kẽm": ("mk", "ma kem"),
    "tắc kê": ("tk", "tac ke", "tacke"),
    "đường kính": ("dk", "duong kinh"),
}

_DIMENSION_RUN = re.compile(r"\b\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?)+\b")
_KEYBOARD_NEIGHBOURS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "g": "fh", "h": "gj",
    "i": "uo", "k": "jl", "l": "k", "m": "n", "n": "bm", "o": "ip", "p": "o",
    "r": "et", "s": "ad", "t": "ry", "u": "yi", "v": "cb", "x": "zc", "y": "tu",
}


@dataclass(frozen=True)
class Query:
    text: str
    sku_id: str
    corruptions: tuple[str, ...]


def _apply_shorthand(text: str, rng: random.Random) -> tuple[str, bool]:
    changed = False
    for full in sorted(SHORTHAND, key=len, reverse=True):
        if full in text.lower():
            replacement = rng.choice(SHORTHAND[full])
            text = re.sub(re.escape(full), replacement, text, flags=re.IGNORECASE)
            changed = True
    return text, changed


def _typo(text: str, rng: random.Random) -> tuple[str, bool]:
    """Introduce one keystroke-level error in an alphabetic word."""
    positions = [i for i, ch in enumerate(text) if ch.isalpha()]
    if not positions:
        return text, False
    index = rng.choice(positions)
    letter = text[index].lower()
    kind = rng.choice(("swap", "drop", "double", "neighbour"))
    if kind == "swap" and index + 1 < len(text) and text[index + 1].isalpha():
        text = text[:index] + text[index + 1] + text[index] + text[index + 2:]
    elif kind == "drop":
        text = text[:index] + text[index + 1:]
    elif kind == "double":
        text = text[:index + 1] + text[index] + text[index + 1:]
    elif kind == "neighbour" and letter in _KEYBOARD_NEIGHBOURS:
        text = text[:index] + rng.choice(_KEYBOARD_NEIGHBOURS[letter]) + text[index + 1:]
    else:
        return text, False
    return text, True


def _partial_dimensions(text: str, rng: random.Random) -> tuple[str, bool]:
    """Staff usually type diameter and length only, dropping the wrench size."""
    match = _DIMENSION_RUN.search(text)
    if not match:
        return text, False
    parts = match.group(0).split("x")
    if len(parts) < 3:
        return text, False
    keep = "x".join(parts[:2])
    return text[: match.start()] + keep + text[match.end():], True


def _separator_variant(text: str, rng: random.Random) -> tuple[str, bool]:
    if not _DIMENSION_RUN.search(text):
        return text, False
    separator = rng.choice(("*", " x ", "×"))
    return _DIMENSION_RUN.sub(lambda m: m.group(0).replace("x", separator), text), True


def _drop_words(text: str, rng: random.Random) -> tuple[str, bool]:
    """Drop a descriptive word, the way a rushed order slip does."""
    words = text.split()
    if len(words) < 4:
        return text, False
    index = rng.randrange(len(words))
    if _DIMENSION_RUN.search(words[index]):  # never drop the dimensions
        return text, False
    return " ".join(words[:index] + words[index + 1:]), True


def _swap_words(text: str, rng: random.Random) -> tuple[str, bool]:
    words = text.split()
    if len(words) < 3:
        return text, False
    i = rng.randrange(len(words) - 1)
    words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words), True



# Regional and trade synonyms.  These are the corruption that lexical matching
# genuinely cannot undo: no amount of character n-gram overlap connects "ốc" to
# "bu lông".  They are the reason this project trains an encoder at all.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "bu lông": ("ốc", "bù loong", "oc vit"),
    "long đen": ("lông đền", "vòng đệm"),
    "tán": ("đai ốc", "ê cu", "ecu"),
    "vít": ("đinh vít", "ốc vít"),
    "tắc kê": ("nở", "nở sắt"),
    "nhám": ("giấy nhám", "giấy ráp"),
    "đinh rút": ("rivet", "ri vê"),
    "pát": ("bát", "ke góc"),
    "pat": ("bat", "ke goc"),
    "cùm": ("kẹp ống", "đai ôm"),
    "keo": ("keo dán",),
    "bạt": ("tấm bạt",),
    "mạ kẽm": ("xi kẽm", "kẽm"),
    "sắt xi": ("xi trắng", "sắt mạ"),
}


def _apply_synonym(text: str, rng: random.Random) -> tuple[str, bool]:
    """Swap one term for the word a different shop assistant would use."""
    candidates = [full for full in SYNONYMS if full in text.lower()]
    if not candidates:
        return text, False
    full = max(candidates, key=len)
    replacement = rng.choice(SYNONYMS[full])
    return re.sub(re.escape(full), replacement, text, count=1, flags=re.IGNORECASE), True


def _terse(text: str, rng: random.Random) -> tuple[str, bool]:
    """Collapse to the way an order is actually shouted across the counter:
    head noun, dimensions, material - every adjective dropped."""
    words = text.split()
    if len(words) < 5:
        return text, False
    keep = words[: rng.choice((1, 2))]
    keep += [w for w in words if _DIMENSION_RUN.search(w) or w.lower() in {"inox", "304", "201", "316"}]
    terse = " ".join(dict.fromkeys(keep))
    # Fewer than three tokens rarely identifies a single SKU; keep the full name.
    if len(terse.split()) < 3:
        return text, False
    return terse, True


# Corruptions that destroy information: after these the query may no longer
# identify a single SKU, so they run first and their result is checked.
LOSSY_CORRUPTIONS = (
    ("terse", _terse, 0.20),
    ("drop_word", _drop_words, 0.25),
    ("partial_dims", _partial_dimensions, 0.35),
)

# Corruptions that only change spelling.  They keep the query answerable, so no
# uniqueness check is needed after them.
SURFACE_CORRUPTIONS = (
    ("synonym", _apply_synonym, 0.45),
    ("shorthand", _apply_shorthand, 0.65),
    ("separator", _separator_variant, 0.30),
    ("swap_words", _swap_words, 0.15),
    ("typo", _typo, 0.30),
)


class AmbiguityChecker:
    """Answers: does this shortened query still name exactly one SKU?

    Built as an inverted index over canonical tokens so the check costs a few
    set intersections instead of a scan over the catalogue.
    """

    def __init__(self, products) -> None:
        self._postings: dict[str, set[int]] = {}
        self._token_sets: list[set[str]] = []
        for index, product in enumerate(products):
            tokens = set(product.canonical.split())
            self._token_sets.append(tokens)
            for token in tokens:
                self._postings.setdefault(token, set()).add(index)

    def matching_skus(self, text: str, limit: int = 5) -> int:
        """Count catalogue entries whose tokens cover every token of `text`."""
        tokens = set(normalize(text).split())
        if not tokens:
            return 0
        candidates: set[int] | None = None
        for token in tokens:
            posting = self._postings.get(token)
            if posting is None:
                return 0  # a token no SKU has - typo territory, not ambiguity
            candidates = posting if candidates is None else candidates & posting
            if not candidates:
                return 0
        return len(candidates)

def corrupt(
    full_name: str,
    rng: random.Random,
    checker: "AmbiguityChecker | None" = None,
) -> tuple[str, tuple[str, ...]] | None:
    """Corrupt one catalogue name into something a person would type.

    Returns ``None`` when the lossy stage left a query that fits several SKUs -
    such a query has no single right answer and would only add label noise.
    """
    text = full_name
    applied: list[str] = []

    for label, transform, probability in LOSSY_CORRUPTIONS:
        if rng.random() < probability:
            text, changed = transform(text, rng)
            if changed:
                applied.append(label)

    if applied and checker is not None and checker.matching_skus(text) > 1:
        return None

    for label, transform, probability in SURFACE_CORRUPTIONS:
        if rng.random() < probability:
            text, changed = transform(text, rng)
            if changed:
                applied.append(label)

    # Shop staff type without diacritics most of the time.
    if rng.random() < 0.80:
        text = strip_diacritics(text)
        applied.append("no_diacritics")

    text = re.sub(r"\s+", " ", text).strip().lower()
    return text, tuple(applied)


def generate(products, per_product: int, seed: int, catalogue=None) -> list[Query]:
    """Build up to `per_product` distinct answerable queries for every SKU.

    `catalogue` is the full product list used for the ambiguity check; it must
    be the whole catalogue, not just the subset being augmented, or a query
    could look unique only because its true competitors were not considered.
    """
    rng = random.Random(seed)
    checker = AmbiguityChecker(catalogue if catalogue is not None else products)
    queries: list[Query] = []
    for product in products:
        seen: set[str] = set()
        for _ in range(per_product * 6):  # oversample, ambiguous draws are rejected
            if len(seen) >= per_product:
                break
            result = corrupt(product.full_name, rng, checker)
            if result is None:
                continue
            text, applied = result
            # A query identical to the clean name teaches the model nothing.
            if not text or text in seen or not applied:
                continue
            if normalize(text) == product.canonical and len(applied) == 1:
                continue
            seen.add(text)
            queries.append(Query(text=text, sku_id=product.sku_id, corruptions=applied))
    return queries
