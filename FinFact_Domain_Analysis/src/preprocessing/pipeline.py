"""End-to-End Preprocessing Orchestrator (Pipeline A: Lemmatization).

Processes content units through conservative cleaning, finance tokenization,
domain-aware stopword filtering, lemmatization, and entity extraction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .cleaner import ConservativeFinancialCleaner, cleaner
from .entities import FinancialEntityRecognizer, entity_recognizer
from .lemmatizer import FinancialLemmatizer, lemmatizer
from .stopwords import DomainAwareStopwordFilter, stopword_filter
from .tokenizer import CustomFinanceTokenizer, Token, tokenizer


class PipelineAProcessor:
    """The empirically validated winning pipeline from Phase 2/3."""

    def __init__(
        self,
        cleaner_inst: ConservativeFinancialCleaner = cleaner,
        tokenizer_inst: CustomFinanceTokenizer = tokenizer,
        stopword_inst: DomainAwareStopwordFilter = stopword_filter,
        lemmatizer_inst: FinancialLemmatizer = lemmatizer,
        entity_inst: FinancialEntityRecognizer = entity_recognizer,
    ):
        self.cleaner = cleaner_inst
        self.tokenizer = tokenizer_inst
        self.stopword_filter = stopword_inst
        self.lemmatizer = lemmatizer_inst
        self.entity_recognizer = entity_inst

    def process_text(self, text: str) -> Dict[str, Any]:
        cleaned = self.cleaner.clean(text)
        raw_tokens = self.tokenizer.tokenize(cleaned)
        entities = self.entity_recognizer.extract_entities(cleaned)

        processed_tokens: List[Dict[str, Any]] = []
        indexed_terms: List[Tuple[str, int]] = []  # (term, pos_idx)

        for tok in raw_tokens:
            is_stop = self.stopword_filter.is_stopword(tok.text)
            lemma = self.lemmatizer.lemmatize(tok.text)

            tok_dict = {
                "text": tok.text,
                "lemma": lemma,
                "pos_idx": tok.pos_idx,
                "start_char": tok.start_char,
                "end_char": tok.end_char,
                "is_stopword": is_stop,
            }
            processed_tokens.append(tok_dict)

            # In Pipeline A, non-stopword lemmas enter the positional inverted index
            if not is_stop:
                indexed_terms.append((lemma.casefold(), tok.pos_idx))

        return {
            "cleaned_text": cleaned,
            "tokens": processed_tokens,
            "indexed_terms": indexed_terms,
            "entities": [{"text": e.text, "label": e.label, "start": e.start_char, "end": e.end_char} for e in entities],
            "token_count": len(raw_tokens),
            "indexed_count": len(indexed_terms),
        }

    def process_unit(self, unit: Dict[str, Any]) -> Dict[str, Any]:
        text = unit.get("text") or ""
        res = self.process_text(text)

        processed_unit = dict(unit)
        processed_unit["cleaned_text"] = res["cleaned_text"]
        processed_unit["tokens"] = res["tokens"]
        processed_unit["indexed_terms"] = res["indexed_terms"]
        processed_unit["entities"] = res["entities"]
        processed_unit["token_count"] = res["token_count"]
        processed_unit["indexed_count"] = res["indexed_count"]

        return processed_unit


pipeline_a = PipelineAProcessor()
