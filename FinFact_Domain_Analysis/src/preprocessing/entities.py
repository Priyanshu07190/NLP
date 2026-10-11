"""Lightweight Rule & Pattern-Based Financial Entity Recognizer.

Extracts domain entities (MONEY, PERCENT, DATE, ORG, PERSON) critical
for financial claim verification and provenance tracking.
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple


class Entity(NamedTuple):
    text: str
    label: str
    start_char: int
    end_char: int


MONEY_PATTERN = re.compile(
    r"""(?:[\$€£₹¥]\s*[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion|lakh|crore|[kKmMbBtT]))?|\b[\d,]+(?:\.\d+)?\s*(?:dollars|rupees|euros|pounds|yen)\b)""",
    re.IGNORECASE,
)

PERCENT_PATTERN = re.compile(
    r"""(?:\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*(?:percent|per\s+cent|bps|basis\s+points)\b)""",
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"""(?:\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\bfiscal\s+(?:year\s+)?\d{4}\b|\bFY\s*\d{2,4}\b)""",
    re.IGNORECASE,
)

# Common High-Frequency Known Financial / Political Entities
KNOWN_ORGS = {
    "snopes", "politifact", "amazon", "business roundtable", "federal reserve",
    "fed", "rbi", "sebi", "sec", "irs", "cdc", "who", "imf", "world bank",
    "treasury", "congress", "senate", "white house", "klbj", "texas state government"
}

KNOWN_PERSONS = {
    "joe biden", "biden", "donald trump", "trump", "dan patrick", "glenn hegar",
    "jared kushner", "kushner", "milton friedman", "friedman", "edward freeman",
    "freeman", "barack obama", "obama", "kamala harris", "harris"
}


class FinancialEntityRecognizer:
    def extract_entities(self, text: str) -> List[Entity]:
        if not text:
            return []

        entities: List[Entity] = []

        # 1. Money
        for m in MONEY_PATTERN.finditer(text):
            entities.append(Entity(text=m.group(0), label="MONEY", start_char=m.start(), end_char=m.end()))

        # 2. Percent
        for m in PERCENT_PATTERN.finditer(text):
            entities.append(Entity(text=m.group(0), label="PERCENT", start_char=m.start(), end_char=m.end()))

        # 3. Date
        for m in DATE_PATTERN.finditer(text):
            entities.append(Entity(text=m.group(0), label="DATE", start_char=m.start(), end_char=m.end()))

        # 4. Known Org & Person matching
        lower = text.casefold()
        for org in KNOWN_ORGS:
            idx = 0
            while True:
                pos = lower.find(org, idx)
                if pos == -1:
                    break
                # Check word boundaries
                if (pos == 0 or not text[pos - 1].isalnum()) and (pos + len(org) == len(text) or not text[pos + len(org)].isalnum()):
                    entities.append(Entity(text=text[pos:pos + len(org)], label="ORG", start_char=pos, end_char=pos + len(org)))
                idx = pos + len(org)

        for person in KNOWN_PERSONS:
            idx = 0
            while True:
                pos = lower.find(person, idx)
                if pos == -1:
                    break
                if (pos == 0 or not text[pos - 1].isalnum()) and (pos + len(person) == len(text) or not text[pos + len(person)].isalnum()):
                    entities.append(Entity(text=text[pos:pos + len(person)], label="PERSON", start_char=pos, end_char=pos + len(person)))
                idx = pos + len(person)

        return sorted(entities, key=lambda e: e.start_char)


entity_recognizer = FinancialEntityRecognizer()
