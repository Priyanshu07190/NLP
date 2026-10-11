"""Positional Inverted Index Architecture.

Stores exact term positions per content unit, enabling exact phrase distance
verification (pos_{i+1} = pos_i + 1) and deterministic relevance ranking.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set


@dataclass
class Posting:
    unit_id: str
    tf: int
    positions: List[int]


class PositionalInvertedIndex:
    def __init__(self):
        # term -> {unit_id: Posting}
        self.index: Dict[str, Dict[str, Posting]] = defaultdict(dict)
        self.total_units: int = 0
        self.indexed_unit_ids: Set[str] = set()

    def add_term_occurrence(self, term: str, unit_id: str, position: int) -> None:
        term_clean = term.casefold()
        postings_map = self.index[term_clean]
        if unit_id not in postings_map:
            postings_map[unit_id] = Posting(unit_id=unit_id, tf=1, positions=[position])
            self.indexed_unit_ids.add(unit_id)
        else:
            posting = postings_map[unit_id]
            posting.tf += 1
            posting.positions.append(position)

    def add_unit_terms(self, unit_id: str, indexed_terms: List[tuple[str, int]]) -> None:
        self.total_units += 1
        for term, pos in indexed_terms:
            self.add_term_occurrence(term, unit_id, pos)

    def get_postings(self, term: str) -> List[Posting]:
        term_clean = term.casefold()
        postings_map = self.index.get(term_clean, {})
        return list(postings_map.values())

    def get_posting(self, term: str, unit_id: str) -> Optional[Posting]:
        term_clean = term.casefold()
        return self.index.get(term_clean, {}).get(unit_id)

    def df(self, term: str) -> int:
        term_clean = term.casefold()
        return len(self.index.get(term_clean, {}))

    @property
    def vocabulary_size(self) -> int:
        return len(self.index)

    @property
    def total_postings(self) -> int:
        return sum(len(postings) for postings in self.index.values())

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable: Dict[str, List[Dict[str, Any]]] = {}
        for term, postings_map in self.index.items():
            serializable[term] = [
                {"unit_id": p.unit_id, "tf": p.tf, "positions": p.positions}
                for p in postings_map.values()
            ]

        data = {
            "total_units": self.total_units,
            "vocabulary_size": len(self.index),
            "index": serializable,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str | Path) -> PositionalInvertedIndex:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        idx = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx.total_units = data.get("total_units", 0)
        raw_index = data.get("index", {})

        for term, postings_list in raw_index.items():
            postings_map: Dict[str, Posting] = {}
            for p in postings_list:
                posting = Posting(
                    unit_id=p["unit_id"],
                    tf=p["tf"],
                    positions=p["positions"],
                )
                postings_map[p["unit_id"]] = posting
                idx.indexed_unit_ids.add(p["unit_id"])
            idx.index[term] = postings_map

        return idx
