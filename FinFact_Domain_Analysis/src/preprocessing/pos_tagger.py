"""Part-of-Speech (POS) Tagger & Syntactic Analyzer for Fin-Fact.

Provides fine-grained Penn Treebank POS tagging, WordNet POS mapping for
accurate POS-guided lemmatization, financial entity/token corrections,
and noun phrase (key concept) extraction.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("finfact.pos")

try:
    import nltk
    from nltk.corpus import wordnet
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    wordnet = None


# Domain token corrections for terms commonly misclassified by general taggers
FINANCIAL_POS_OVERRIDES: Dict[str, str] = {
    "401(k)": "NN",
    "401k": "NN",
    "covid-19": "NNP",
    "shareholder-capitalism": "NN",
    "rainy-day": "JJ",
    "long-term": "JJ",
    "short-term": "JJ",
    "post-pandemic": "JJ",
    "pre-tax": "JJ",
    "after-tax": "JJ",
    "biden": "NNP",
    "trump": "NNP",
    "obama": "NNP",
    "fed": "NNP",
    "sec": "NNP",
    "rbi": "NNP",
    "sebi": "NNP",
    "irs": "NNP",
}


def penn_to_wordnet(tag: str) -> str:
    """Map Penn Treebank tag to WordNet POS character ('n', 'v', 'a', 'r')."""
    if not tag:
        return "n"
    tag_upper = tag.upper()
    if tag_upper.startswith("J"):
        return "a"
    if tag_upper.startswith("V"):
        return "v"
    if tag_upper.startswith("N"):
        return "n"
    if tag_upper.startswith("R"):
        return "r"
    return "n"


class FinancialPOSTagger:
    """Production-grade financial POS tagger using Averaged Perceptron."""

    def __init__(self):
        self._enabled = _NLTK_AVAILABLE

    def tag_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Given a list of token strings, return list of (token, pos_tag)."""
        if not tokens:
            return []

        if not self._enabled:
            # Fallback heuristic tagger if NLTK is unavailable
            return [(t, self._heuristic_tag(t)) for t in tokens]

        try:
            raw_tags = nltk.pos_tag(tokens)
        except Exception as e:
            logger.warning(f"NLTK pos_tag error: {e}. Falling back to heuristic tagging.")
            return [(t, self._heuristic_tag(t)) for t in tokens]

        corrected_tags: List[Tuple[str, str]] = []
        for word, tag in raw_tags:
            lower = word.casefold()

            # Currency / percentage / numerical pattern overrides
            if word.startswith(("$", "€", "£", "₹", "¥")) or word.endswith("%") or any(c.isdigit() for c in word):
                corrected_tags.append((word, "CD"))
            elif lower in FINANCIAL_POS_OVERRIDES:
                corrected_tags.append((word, FINANCIAL_POS_OVERRIDES[lower]))
            else:
                corrected_tags.append((word, tag))

        return corrected_tags

    def _heuristic_tag(self, word: str) -> str:
        """Lightweight heuristic fallback tagger."""
        if not word:
            return "NN"
        if word.startswith(("$", "€", "£", "₹", "¥")) or word.endswith("%") or any(c.isdigit() for c in word):
            return "CD"
        lower = word.casefold()
        if lower in FINANCIAL_POS_OVERRIDES:
            return FINANCIAL_POS_OVERRIDES[lower]
        if lower in {"not", "never", "no", "hardly", "scarcely", "barely"}:
            return "RB"
        if lower in {"will", "would", "can", "could", "may", "might", "shall", "should", "must"}:
            return "MD"
        if lower.endswith(("ing", "ed", "ize", "ise", "ate")):
            return "VB"
        if lower.endswith(("ly")):
            return "RB"
        if lower.endswith(("able", "ible", "al", "ic", "ive", "ous", "ful", "less")):
            return "JJ"
        if word[0].isupper():
            return "NNP"
        return "NN"

    def extract_noun_phrases(self, tagged_tokens: List[Tuple[str, str]]) -> List[str]:
        """Extract multi-word noun phrase concepts (e.g. [JJ]* [NN]+)."""
        phrases: List[str] = []
        current: List[str] = []

        for word, tag in tagged_tokens:
            tag_upper = tag.upper()
            if tag_upper.startswith("J") or tag_upper.startswith("N"):
                current.append(word)
            else:
                if len(current) >= 2:
                    phrases.append(" ".join(current))
                current = []

        if len(current) >= 2:
            phrases.append(" ".join(current))

        return phrases


pos_tagger = FinancialPOSTagger()
