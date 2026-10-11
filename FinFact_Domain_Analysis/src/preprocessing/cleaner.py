"""Conservative Financial Text Cleaner for Web & Fact-Checking Corpus.

Preserves all monetary expressions, rates, fiscal codes, and numbers while
cleaning web extraction artefacts, unicode quotes, and promotional boilerplate.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Set

UNICODE_REPLACEMENTS = {
    "\u2014": " — ",
    "\u2013": " – ",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u00a0": " ",
    "\ufeff": "",
    "\u200b": "",
}

# Regex to safely separate glued words after punctuation without breaking URLs or decimals
GLUED_COMMA = re.compile(r"([a-zA-Z]),([a-zA-Z])")
GLUED_PERIOD = re.compile(r"([a-z])\.([A-Z])")
COLLAPSE_WS = re.compile(r"[ \t\f\v]+")
COLLAPSE_NL = re.compile(r"\n{2,}")

# Boilerplate patterns from fact-checking scraper artifacts
DEFAULT_BOILERPLATE = [
    r"Become a Founding Member to help us hire more fact-checkers\.",
    r"Snopes is still fighting an infodemic of rumors and misinformation.*?(?=\.|$)",
    r"Find out what we've learned and how to inoculate yourself against COVID-19 misinformation\.",
    r"Read the latest fact checks about the vaccines\.",
    r"Submit any questionable rumors and advice you encounter\.",
    r"And, please, follow the CDC or WHO for guidance on protecting your community from the disease\.",
    r"Support the Snopes newsroom by becoming a member\.",
]


class ConservativeFinancialCleaner:
    def __init__(
        self,
        boilerplate_patterns: Sequence[str] | None = None,
        filter_boilerplate: bool = True,
        normalize_unicode: bool = True,
        fix_glued_punctuation: bool = True,
        fix_common_contractions: bool = True,
    ):
        self.normalize_unicode_flag = normalize_unicode
        self.fix_glued_punctuation_flag = fix_glued_punctuation
        self.fix_contractions_flag = fix_common_contractions
        self.filter_boilerplate_flag = filter_boilerplate

        patterns = list(boilerplate_patterns) if boilerplate_patterns else DEFAULT_BOILERPLATE
        self.boilerplate_regexes = [re.compile(p, re.IGNORECASE) for p in patterns]

    def clean(self, text: str) -> str:
        if not text:
            return ""

        out = text

        # 1. Unicode character normalization
        if self.normalize_unicode_flag:
            for k, v in UNICODE_REPLACEMENTS.items():
                out = out.replace(k, v)

        # 2. Filter out scraper promotional boilerplate sentences
        if self.filter_boilerplate_flag:
            for bp in self.boilerplate_regexes:
                out = bp.sub("", out)

        # 3. Fix glued punctuation from scraping (e.g., 'interview,which' -> 'interview, which')
        if self.fix_glued_punctuation_flag:
            out = GLUED_COMMA.sub(r"\1, \2", out)
            out = GLUED_PERIOD.sub(r"\1. \2", out)

        # 4. Fix broken contractions in web text
        if self.fix_contractions_flag:
            out = re.sub(r"\bWell have\b", "We'll have", out)
            out = re.sub(r"\bThats\b", "That's", out)
            out = re.sub(r"\bTheres\b", "There's", out)
            out = re.sub(r"\bDont\b", "Don't", out)
            out = re.sub(r"\bWont\b", "Won't", out)
            out = re.sub(r"\bCant\b", "Can't", out)

        # 5. Whitespace and newline collapsing
        out = COLLAPSE_WS.sub(" ", out)
        out = COLLAPSE_NL.sub("\n", out)

        return out.strip()


cleaner = ConservativeFinancialCleaner()
