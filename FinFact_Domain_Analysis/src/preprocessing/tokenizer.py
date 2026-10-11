"""Custom Finance Tokenizer.

Preserves domain compound expressions, currency-bound numbers, percentage rates,
fiscal year tags, and hyphenated economic terms as single atomic tokens.
"""

from __future__ import annotations

import re
from typing import List, NamedTuple, Pattern


class Token(NamedTuple):
    text: str
    lemma: str
    pos_idx: int
    start_char: int
    end_char: int


# Tokenization Regex Patterns designed specifically for Financial Corpus
FINANCE_TOKEN_PATTERN = re.compile(
    r"""
    # 1. Currency with amount (e.g., $11B, $8.5, €100M, ₹5,000, £2.4T)
    (?:[\$€£₹¥]\s*[\d,]+(?:\.\d+)?(?:[kKmMbBtT]|(?:\s*(?:billion|million|trillion|lakh|crore)))?)
    |
    # 2. Percentage rates and basis points (e.g., 5.5%, 25bps, -0.25%)
    (?:[+-]?\d+(?:\.\d+)?%|[+-]?\d+(?:\.\d+)?\s*(?:bps|basis\s+points))
    |
    # 3. Fiscal years and quarters (e.g., FY2025, FY26, Q3FY24, 401(k), 401k)
    (?:401\(k\)|401k|FY\s*\d{2,4}|Q[1-4]\s*(?:FY\s*\d{2,4})?|H[1-2]\s*(?:FY\s*\d{2,4})?)
    |
    # 4. Standard numerical expressions (integers, decimals, comma separated)
    (?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)
    |
    # 5. Hyphenated financial terms & compound expressions (e.g. shareholder-capitalism, COVID-19)
    (?:[a-zA-Z]+(?:-[a-zA-Z0-9]+)+)
    |
    # 6. Standard words with contraction support (e.g., doesn't, we'll, economic)
    (?:[a-zA-Z]+(?:'[a-zA-Z]+)?)
    |
    # 7. Remaining single domain symbols
    (?:[\$€£₹¥%])
    """,
    re.VERBOSE | re.IGNORECASE,
)


class CustomFinanceTokenizer:
    def __init__(self, pattern: Pattern[str] = FINANCE_TOKEN_PATTERN):
        self.pattern = pattern

    def tokenize(self, text: str) -> List[Token]:
        if not text:
            return []

        tokens: List[Token] = []
        pos_idx = 0

        for match in self.pattern.finditer(text):
            tok_text = match.group(0).strip()
            if not tok_text:
                continue

            tokens.append(
                Token(
                    text=tok_text,
                    lemma=tok_text,  # Populated later in morphology stage
                    pos_idx=pos_idx,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )
            pos_idx += 1

        return tokens


tokenizer = CustomFinanceTokenizer()
