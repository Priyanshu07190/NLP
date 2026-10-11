"""Rule and Dictionary Based Morphological Lemmatizer (Pipeline A).

Implements POS-guided inflectional reduction preserving domain semantics,
preventing the over-truncation errors characteristic of Porter/Snowball stemmers.
Zero-network dependency with full offline predictability.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# Irregular forms table for English & Financial vocabulary
IRREGULAR_LEMMAS: Dict[str, str] = {
    # Irregular Nouns
    "children": "child",
    "people": "person",
    "men": "man",
    "women": "woman",
    "criteria": "criterion",
    "data": "datum",
    "indices": "index",
    "indexes": "index",
    "matrices": "matrix",
    "analyses": "analysis",
    "crises": "crisis",
    "bases": "base",
    "currencies": "currency",
    "companies": "company",
    "entities": "entity",
    "subsidies": "subsidy",
    "policies": "policy",
    "commodities": "commodity",
    "monies": "money",
    "taxes": "tax",
    "losses": "loss",
    "businesses": "business",
    "processes": "process",
    # Irregular Verbs
    "paid": "pay",
    "paying": "pay",
    "spent": "spend",
    "spending": "spend",
    "bought": "buy",
    "buying": "buy",
    "sold": "sell",
    "selling": "sell",
    "held": "hold",
    "holding": "hold",
    "rose": "rise",
    "risen": "rise",
    "rising": "rise",
    "fell": "fall",
    "fallen": "fall",
    "falling": "fall",
    "grew": "grow",
    "grown": "grow",
    "growing": "grow",
    "led": "lead",
    "leading": "lead",
    "lent": "lend",
    "lending": "lend",
    "built": "build",
    "building": "build",
    "lost": "lose",
    "losing": "lose",
    "made": "make",
    "making": "make",
    "taken": "take",
    "took": "take",
    "taking": "take",
    "given": "give",
    "gave": "give",
    "giving": "give",
    "chosen": "choose",
    "chose": "choose",
    "choosing": "choose",
    "written": "write",
    "wrote": "write",
    "writing": "write",
    "spoken": "speak",
    "spoke": "speak",
    "speaking": "speak",
    "known": "know",
    "knew": "know",
    "knowing": "know",
    "seen": "see",
    "saw": "see",
    "seeing": "see",
    "became": "become",
    "becoming": "become",
    "began": "begin",
    "begun": "begin",
    "beginning": "begin",
    "found": "find",
    "finding": "find",
    "went": "go",
    "gone": "go",
    "going": "go",
    "goes": "go",
    "came": "come",
    "coming": "come",
    "comes": "come",
    "said": "say",
    "saying": "say",
    "says": "say",
    "done": "do",
    "did": "do",
    "doing": "do",
    "does": "do",
    "been": "be",
    "was": "be",
    "were": "be",
    "is": "be",
    "am": "be",
    "are": "be",
    "being": "be",
}

# Suffix transformation rules for regular English inflection
NOUN_SUFFIX_RULES = [
    (re.compile(r"ies$"), "y"),         # companies -> company
    (re.compile(r"(ches|shes|sses|xes|zes)$"), lambda m: m.group(1)[:-2]), # boxes -> box
    (re.compile(r"(ves)$"), "f"),       # halves -> half
    (re.compile(r"s$"), ""),            # investments -> investment, shares -> share
]

VERB_SUFFIX_RULES = [
    (re.compile(r"ing$"), ""),          # defaulting fallback
    (re.compile(r"ed$"), ""),           # liquidated fallback
]


class FinancialLemmatizer:
    """Robust lemmatizer representing Pipeline A morphology."""

    def __init__(self, exception_map: Dict[str, str] | None = None):
        self.exceptions = dict(IRREGULAR_LEMMAS)
        if exception_map:
            self.exceptions.update(exception_map)

    def lemmatize(self, word: str, pos: Optional[str] = None) -> str:
        if not word or len(word) <= 2:
            return word

        # Preserve currency amounts or fiscal tags
        if word.startswith(("$", "€", "£", "₹", "¥")) or word.endswith("%") or "401" in word:
            return word

        lower = word.casefold()

        # 1. Exact irregular lookup
        if lower in self.exceptions:
            lemma = self.exceptions[lower]
            return lemma if word.islower() else (lemma.capitalize() if word.istitle() else lemma)

        # 2. General financial rules
        # Preserve words that naturally end in 's'
        if lower in {"gas", "lens", "status", "crisis", "series", "species", "stimulus", "surplus", "deficit"}:
            return word

        # Handling verb -ing / -ed
        if lower.endswith("ing") and len(lower) > 5:
            base = lower[:-3]
            # Doubled consonant: running -> run, dropping -> drop
            if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "ls":
                return base[:-1]
            # Silent e: proposing -> propose, liquidating -> liquidate
            if base.endswith(("at", "iz", "is", "uc", "us", "id", "iv", "ut")):
                return base + "e"
            return base

        if lower.endswith("ed") and len(lower) > 4:
            base = lower[:-2]
            if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "ls":
                return base[:-1]
            if base.endswith(("at", "iz", "is", "uc", "us", "id", "iv", "ut")):
                return base + "e"
            if lower.endswith("ied"):
                return lower[:-3] + "y"
            return base

        # Handling noun plurals
        if lower.endswith("ies") and len(lower) > 4:
            return lower[:-3] + "y"

        if lower.endswith("es") and len(lower) > 4:
            if lower.endswith(("sses", "ches", "shes", "xes", "zes")):
                return lower[:-2]
            if lower in {"halves", "calves", "shelves", "wolves", "thieves", "scarves"}:
                return lower[:-3] + "f"
            if lower in {"wives", "knives", "lives"}:
                return lower[:-3] + "fe"
            if lower.endswith(("ves", "res", "des", "tes", "ses", "mes", "nes", "les", "pes", "ges")):
                return lower[:-1]

        if lower.endswith("s") and not lower.endswith(("ss", "us", "is")):
            return lower[:-1]

        return lower


lemmatizer = FinancialLemmatizer()
