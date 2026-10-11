"""Multi-Term Evidence Snippet Generator with Query Highlighting.

Constructs compact, context-preserving evidence windows around matched terms
and exact phrases, maintaining strict positional provenance.
"""

from __future__ import annotations

import re
from typing import List, Set


class SnippetGenerator:
    def __init__(self, window_words: int = 35):
        self.window_words = window_words

    def generate_snippet(
        self,
        text: str,
        matched_terms: Set[str],
        matched_phrases: List[str] | None = None,
    ) -> str:
        if not text:
            return ""

        words = text.split()
        if len(words) <= self.window_words:
            # Highlight all matched terms in the entire text
            return self._highlight(text, matched_terms, matched_phrases)

        # Find word indices that contain any of the matched terms
        term_clean = {t.casefold() for t in matched_terms if t}
        phrase_clean = [p.casefold() for p in (matched_phrases or []) if p]

        hit_indices: List[int] = []
        for idx, w in enumerate(words):
            w_strip = re.sub(r"[^\w\$€£₹%]", "", w).casefold()
            if w_strip in term_clean or any(t in w_strip for t in term_clean):
                hit_indices.append(idx)

        if not hit_indices:
            # Default to first window_words
            snippet_text = " ".join(words[:self.window_words]) + "..."
            return self._highlight(snippet_text, matched_terms, matched_phrases)

        # Pick best center index (highest density of hits in a window)
        best_center = hit_indices[0]
        max_hits_in_window = 0

        for candidate in hit_indices:
            start_w = max(0, candidate - self.window_words // 2)
            end_w = min(len(words), start_w + self.window_words)
            hits_count = sum(1 for h in hit_indices if start_w <= h < end_w)
            if hits_count > max_hits_in_window:
                max_hits_in_window = hits_count
                best_center = candidate

        start_w = max(0, best_center - self.window_words // 2)
        end_w = min(len(words), start_w + self.window_words)

        prefix = "... " if start_w > 0 else ""
        suffix = " ..." if end_w < len(words) else ""

        snippet_raw = prefix + " ".join(words[start_w:end_w]) + suffix

        # Check if some hits were outside this window
        outside_hits = any(h < start_w or h >= end_w for h in hit_indices)
        if outside_hits:
            snippet_raw += " [Additional matched term occurs elsewhere in this content unit.]"

        return self._highlight(snippet_raw, matched_terms, matched_phrases)

    def _highlight(
        self,
        text: str,
        matched_terms: Set[str],
        matched_phrases: List[str] | None = None,
    ) -> str:
        highlighted = text

        # 1. Phrases first (longer matches)
        if matched_phrases:
            for phrase in matched_phrases:
                if not phrase.strip():
                    continue
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                highlighted = pattern.sub(r'<mark class="query-hit query-phrase">\g<0></mark>', highlighted)

        # 2. Individual terms
        for term in matched_terms:
            if not term or len(term) < 2:
                continue
            # Match whole word boundaries if possible
            pattern = re.compile(r"\b(" + re.escape(term) + r"[a-z]*)\b", re.IGNORECASE)
            highlighted = pattern.sub(r'<mark class="query-hit">\1</mark>', highlighted)

        return highlighted


snippet_generator = SnippetGenerator()
