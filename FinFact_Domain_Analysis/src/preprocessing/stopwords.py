"""Domain-Aware Stopword Removal with Polarity Protection.

Filters non-informative grammatical noise while preserving:
1. Critical domain financial vocabulary (e.g. rate, tax, bank, fund, debt, profit).
2. Polarity / negation operators (e.g. not, no, never, without), which determine
   the factual veracity of claims.
"""

from __future__ import annotations

from typing import Set

# Base English grammatical stopword list
BASE_ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "hed", "hell", "hes", "her", "here", "heres",
    "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id",
    "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
    "lets", "me", "more", "most", "my", "myself", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "shed", "shell", "shes", "should", "so", "some",
    "such", "than", "that", "thats", "the", "their", "theirs", "them",
    "themselves", "then", "there", "theres", "these", "they", "theyd",
    "theyll", "theyre", "theyve", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "wed", "well", "were",
    "weve", "what", "whats", "when", "whens", "where", "wheres", "which",
    "while", "who", "whos", "whom", "why", "whys", "with", "would", "you",
    "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves"
}

# Negation and polarity words that must NEVER be stripped in fact-checking
PROTECTED_POLARITY_TERMS: Set[str] = {
    "not", "no", "nor", "neither", "never", "none", "without", "against",
    "cannot", "cant", "wont", "dont", "isnt", "arent", "wasnt", "werent",
    "hardly", "scarcely", "barely"
}

# Financial domain terms that carry high retrieval value
PROTECTED_FINANCIAL_TERMS: Set[str] = {
    "tax", "taxes", "rate", "rates", "debt", "bank", "banks", "banking",
    "loss", "losses", "profit", "profits", "gain", "gains", "capital",
    "capitalism", "inflation", "recession", "stock", "stocks", "share",
    "shares", "shareholder", "shareholders", "stakeholder", "stakeholders",
    "dividend", "dividends", "pension", "pensions", "bond", "bonds",
    "yield", "fund", "funds", "asset", "assets", "liability", "liabilities",
    "credit", "loan", "loans", "interest", "audit", "trade", "market",
    "markets", "revenue", "expenditure", "deficit", "surplus", "gdp", "gva",
    "equity", "mortgage", "bailout", "reserve", "reserves", "stimulus",
    "cost", "costs", "spending", "budget", "liquidity", "solvency", "bill",
    "billion", "million", "trillion", "dollar", "rupee", "euro", "yen"
}


class DomainAwareStopwordFilter:
    def __init__(
        self,
        preserve_polarity: bool = True,
        preserve_finance: bool = True,
    ):
        self.stop_words: Set[str] = set(BASE_ENGLISH_STOPWORDS)

        if preserve_polarity:
            self.stop_words.difference_update(PROTECTED_POLARITY_TERMS)

        if preserve_finance:
            self.stop_words.difference_update(PROTECTED_FINANCIAL_TERMS)

    def is_stopword(self, word: str) -> bool:
        return word.casefold() in self.stop_words


stopword_filter = DomainAwareStopwordFilter()
