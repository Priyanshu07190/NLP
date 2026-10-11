"""Information Retrieval & Fact-Checking Search Engine.

Executes AST Boolean queries, exact positional phrase verification, and
deterministic ranking over the Pipeline A Positional Inverted Index.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from ..indexing.inverted_index import PositionalInvertedIndex, Posting
from .query_parser import (
    AndNode,
    NotNode,
    OrNode,
    PhraseNode,
    QueryNode,
    TermNode,
    parse_query,
)
from .snippet_generator import SnippetGenerator, snippet_generator


class SearchResult:
    def __init__(
        self,
        unit_id: str,
        score: float,
        matched_terms: Set[str],
        matched_phrases: List[str],
        unit_data: Dict[str, Any],
        snippet: str,
    ):
        self.unit_id = unit_id
        self.score = round(score, 4)
        self.matched_terms = list(matched_terms)
        self.matched_phrases = matched_phrases
        self.claim_id = unit_data.get("claim_id", "")
        self.unit_type = unit_data.get("unit_type", "")
        self.label = unit_data.get("label", "nei")
        self.author = unit_data.get("author", "")
        self.posted = unit_data.get("posted", "")
        self.url = unit_data.get("url", "")
        self.parent_claim = unit_data.get("parent_claim", "")
        self.text = unit_data.get("text", "")
        self.cleaned_text = unit_data.get("cleaned_text", "")
        self.entities = unit_data.get("entities", [])
        self.hrefs = unit_data.get("hrefs", [])
        self.snippet = snippet

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "score": self.score,
            "claim_id": self.claim_id,
            "unit_type": self.unit_type,
            "label": self.label,
            "author": self.author,
            "posted": self.posted,
            "url": self.url,
            "parent_claim": self.parent_claim,
            "snippet": self.snippet,
            "matched_terms": self.matched_terms,
            "matched_phrases": self.matched_phrases,
            "entities": self.entities,
            "hrefs": self.hrefs,
        }


class SearchEngine:
    def __init__(
        self,
        index: PositionalInvertedIndex,
        unit_store: Dict[str, Dict[str, Any]],
        snippet_gen: SnippetGenerator = snippet_generator,
        term_weight: float = 1.0,
        phrase_bonus: float = 2.0,
        log_tf_weight: float = 0.25,
    ):
        self.index = index
        self.unit_store = unit_store
        self.snippet_generator = snippet_gen
        self.term_weight = term_weight
        self.phrase_bonus = phrase_bonus
        self.log_tf_weight = log_tf_weight

    # -----------------------------------------------------------------------
    # Positional Exact Phrase Verification: pos_{i+1} = pos_i + 1
    # -----------------------------------------------------------------------
    def match_phrase(self, terms: List[str]) -> Set[str]:
        if not terms:
            return set()
        if len(terms) == 1:
            return {p.unit_id for p in self.index.get_postings(terms[0])}

        # Retrieve postings for each term
        postings_per_term: List[Dict[str, Posting]] = []
        for term in terms:
            postings_list = self.index.get_postings(term)
            if not postings_list:
                return set()  # If any term has 0 postings, phrase cannot match
            postings_per_term.append({p.unit_id: p for p in postings_list})

        # Candidate units are the intersection of all term document sets
        candidate_units = set(postings_per_term[0].keys())
        for term_map in postings_per_term[1:]:
            candidate_units.intersection_update(term_map.keys())

        matched_units: Set[str] = set()

        for unit_id in candidate_units:
            # Check adjacent positions: pos_{i+1} = pos_i + 1
            pos_lists = [postings_per_term[i][unit_id].positions for i in range(len(terms))]
            if self._verify_positions_chain(pos_lists):
                matched_units.add(unit_id)

        return matched_units

    def _verify_positions_chain(self, pos_lists: List[List[int]]) -> bool:
        # Check if there exists pos in pos_lists[0] such that pos+1 in pos_lists[1], etc.
        first_positions = set(pos_lists[0])
        for p in first_positions:
            chain_valid = True
            for offset in range(1, len(pos_lists)):
                if (p + offset) not in set(pos_lists[offset]):
                    chain_valid = False
                    break
            if chain_valid:
                return True
        return False

    # -----------------------------------------------------------------------
    # AST Boolean Node Evaluation
    # -----------------------------------------------------------------------
    def evaluate_node(self, node: QueryNode) -> Set[str]:
        if isinstance(node, TermNode):
            if not node.term:
                return set()
            return {p.unit_id for p in self.index.get_postings(node.term)}

        if isinstance(node, PhraseNode):
            return self.match_phrase(node.terms)

        if isinstance(node, AndNode):
            left_units = self.evaluate_node(node.left)
            right_units = self.evaluate_node(node.right)
            return left_units.intersection(right_units)

        if isinstance(node, OrNode):
            left_units = self.evaluate_node(node.left)
            right_units = self.evaluate_node(node.right)
            return left_units.union(right_units)

        if isinstance(node, NotNode):
            excluded = self.evaluate_node(node.child)
            return self.index.indexed_unit_ids.difference(excluded)

        return set()

    # -----------------------------------------------------------------------
    # Query Term & Phrase Extraction
    # -----------------------------------------------------------------------
    def extract_query_features(self, node: QueryNode) -> Tuple[Set[str], List[Tuple[List[str], str]]]:
        terms: Set[str] = set()
        phrases: List[Tuple[List[str], str]] = []

        def _traverse(n: QueryNode):
            if isinstance(n, TermNode):
                if n.term:
                    terms.add(n.term.casefold())
            elif isinstance(n, PhraseNode):
                if n.terms:
                    phrases.append((n.terms, n.raw_phrase))
                    terms.update([t.casefold() for t in n.terms])
            elif isinstance(n, AndNode) or isinstance(n, OrNode):
                _traverse(n.left)
                _traverse(n.right)
            elif isinstance(n, NotNode):
                # We do not score positively for terms inside a NOT branch
                pass

        _traverse(node)
        return terms, phrases

    # -----------------------------------------------------------------------
    # Ranking Score Function
    # Score = 1.0 * N_matched + 2.0 * I_phrase + 0.25 * log10(1 + TF)
    # -----------------------------------------------------------------------
    def score_unit(
        self,
        unit_id: str,
        query_terms: Set[str],
        query_phrases: List[Tuple[List[str], str]],
    ) -> Tuple[float, Set[str], List[str]]:
        matched_terms: Set[str] = set()
        total_tf = 0
        phrase_matches: List[str] = []

        # 1. Term matches & TF
        for term in query_terms:
            posting = self.index.get_posting(term, unit_id)
            if posting:
                matched_terms.add(term)
                total_tf += posting.tf

        # 2. Phrase bonus
        for p_terms, raw_phrase in query_phrases:
            p_matched_units = self.match_phrase(p_terms)
            if unit_id in p_matched_units:
                phrase_matches.append(raw_phrase)

        if not matched_terms and not phrase_matches:
            return 0.0, set(), []

        n_matched = len(matched_terms)
        i_phrase = len(phrase_matches)
        tf_component = math.log10(1 + total_tf) if total_tf > 0 else 0.0

        score = (
            (self.term_weight * n_matched)
            + (self.phrase_bonus * i_phrase)
            + (self.log_tf_weight * tf_component)
        )

        return score, matched_terms, phrase_matches

    # -----------------------------------------------------------------------
    # Primary Search Execution
    # -----------------------------------------------------------------------
    def search(
        self,
        query_str: str,
        top_k: int = 10,
        unit_type_filter: Optional[str] = None,
        label_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not query_str.strip():
            return {
                "query": query_str,
                "total_matched": 0,
                "results": [],
                "verdict_summary": {"true": 0, "false": 0, "nei": 0},
            }

        ast_root = parse_query(query_str)
        candidate_unit_ids = self.evaluate_node(ast_root)

        query_terms, query_phrases = self.extract_query_features(ast_root)

        scored_candidates: List[SearchResult] = []

        for unit_id in candidate_unit_ids:
            unit_data = self.unit_store.get(unit_id)
            if not unit_data:
                continue

            # Apply filters if requested
            if unit_type_filter and unit_type_filter.lower() != "all":
                if unit_data.get("unit_type") != unit_type_filter.lower():
                    continue

            if label_filter and label_filter.lower() != "all":
                if unit_data.get("label") != label_filter.lower():
                    continue

            score, matched_terms, matched_p = self.score_unit(unit_id, query_terms, query_phrases)

            raw_phrase_strings = [p[1] for p in query_phrases]
            snippet = self.snippet_generator.generate_snippet(
                unit_data.get("cleaned_text") or unit_data.get("text", ""),
                matched_terms,
                raw_phrase_strings,
            )

            res = SearchResult(
                unit_id=unit_id,
                score=score,
                matched_terms=matched_terms,
                matched_phrases=matched_p,
                unit_data=unit_data,
                snippet=snippet,
            )
            scored_candidates.append(res)

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        top_results = scored_candidates[:top_k]

        # Calculate verdict summary across top retrieved units
        verdict_summary = {"true": 0, "false": 0, "nei": 0}
        for r in top_results:
            lbl = r.label.lower()
            if lbl in verdict_summary:
                verdict_summary[lbl] += 1
            else:
                verdict_summary["nei"] += 1

        return {
            "query": query_str,
            "total_matched": len(scored_candidates),
            "top_k": len(top_results),
            "results": [r.to_dict() for r in top_results],
            "verdict_summary": verdict_summary,
        }
