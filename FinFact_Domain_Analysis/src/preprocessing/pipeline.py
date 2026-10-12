"""End-to-End Preprocessing Orchestrators for Fin-Fact:
1. Pipeline A (Baseline): Conservative Cleaning, Custom Tokenization, Polarity Stopwords, Lemmatization (Without POS).
2. Pipeline A+ (With POS Tagging): POS Tagging, WordNet POS-guided Lemmatization, Noun Phrase Extraction & Syntactic Enrichment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .cleaner import ConservativeFinancialCleaner, cleaner
from .entities import FinancialEntityRecognizer, entity_recognizer
from .lemmatizer import FinancialLemmatizer, lemmatizer
from .pos_tagger import FinancialPOSTagger, penn_to_wordnet, pos_tagger
from .stopwords import DomainAwareStopwordFilter, stopword_filter
from .tokenizer import CustomFinanceTokenizer, Token, tokenizer


class PipelineAProcessor:
    """Pipeline A Baseline: Heuristic Lemmatization WITHOUT explicit POS tagging."""

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
        self.name = "Pipeline A (Without POS)"

    def process_text(self, text: str) -> Dict[str, Any]:
        cleaned = self.cleaner.clean(text)
        raw_tokens = self.tokenizer.tokenize(cleaned)
        entities = self.entity_recognizer.extract_entities(cleaned)

        processed_tokens: List[Dict[str, Any]] = []
        indexed_terms: List[Tuple[str, int]] = []  # (term, pos_idx)

        for tok in raw_tokens:
            is_stop = self.stopword_filter.is_stopword(tok.text)
            lemma = self.lemmatizer.lemmatize(tok.text, pos=None)  # No POS passed

            tok_dict = {
                "text": tok.text,
                "lemma": lemma,
                "pos_tag": "",  # Empty for baseline
                "pos_idx": tok.pos_idx,
                "start_char": tok.start_char,
                "end_char": tok.end_char,
                "is_stopword": is_stop,
            }
            processed_tokens.append(tok_dict)

            if not is_stop:
                indexed_terms.append((lemma.casefold(), tok.pos_idx))

        return {
            "cleaned_text": cleaned,
            "tokens": processed_tokens,
            "indexed_terms": indexed_terms,
            "entities": [{"text": e.text, "label": e.label, "start": e.start_char, "end": e.end_char} for e in entities],
            "noun_phrases": [],
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
        processed_unit["noun_phrases"] = res["noun_phrases"]
        processed_unit["token_count"] = res["token_count"]
        processed_unit["indexed_count"] = res["indexed_count"]
        processed_unit["pipeline_variant"] = "baseline_without_pos"

        return processed_unit


class PipelineAPosProcessor:
    """Pipeline A+ (With POS Tagging): POS Tagging, POS-Guided Lemmatization & Syntactic Analysis."""

    def __init__(
        self,
        cleaner_inst: ConservativeFinancialCleaner = cleaner,
        tokenizer_inst: CustomFinanceTokenizer = tokenizer,
        stopword_inst: DomainAwareStopwordFilter = stopword_filter,
        lemmatizer_inst: FinancialLemmatizer = lemmatizer,
        pos_tagger_inst: FinancialPOSTagger = pos_tagger,
        entity_inst: FinancialEntityRecognizer = entity_recognizer,
    ):
        self.cleaner = cleaner_inst
        self.tokenizer = tokenizer_inst
        self.stopword_filter = stopword_inst
        self.lemmatizer = lemmatizer_inst
        self.pos_tagger = pos_tagger_inst
        self.entity_recognizer = entity_inst
        self.name = "Pipeline A+ (With POS Tagging)"

    def process_text(self, text: str) -> Dict[str, Any]:
        cleaned = self.cleaner.clean(text)
        raw_tokens = self.tokenizer.tokenize(cleaned)
        entities = self.entity_recognizer.extract_entities(cleaned)

        token_strings = [t.text for t in raw_tokens]
        tagged_pairs = self.pos_tagger.tag_tokens(token_strings)
        noun_phrases = self.pos_tagger.extract_noun_phrases(tagged_pairs)

        processed_tokens: List[Dict[str, Any]] = []
        indexed_terms: List[Tuple[str, int]] = []  # (term, pos_idx)

        for i, tok in enumerate(raw_tokens):
            penn_tag = tagged_pairs[i][1] if i < len(tagged_pairs) else "NN"
            wn_pos = penn_to_wordnet(penn_tag)
            is_stop = self.stopword_filter.is_stopword(tok.text)

            # POS-Guided Lemmatization: WordNet receives disambiguated POS
            lemma = self.lemmatizer.lemmatize(tok.text, pos=wn_pos)

            tok_dict = {
                "text": tok.text,
                "lemma": lemma,
                "pos_tag": penn_tag,
                "pos_idx": tok.pos_idx,
                "start_char": tok.start_char,
                "end_char": tok.end_char,
                "is_stopword": is_stop,
            }
            processed_tokens.append(tok_dict)

            if not is_stop:
                indexed_terms.append((lemma.casefold(), tok.pos_idx))

        return {
            "cleaned_text": cleaned,
            "tokens": processed_tokens,
            "indexed_terms": indexed_terms,
            "entities": [{"text": e.text, "label": e.label, "start": e.start_char, "end": e.end_char} for e in entities],
            "noun_phrases": noun_phrases,
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
        processed_unit["noun_phrases"] = res["noun_phrases"]
        processed_unit["token_count"] = res["token_count"]
        processed_unit["indexed_count"] = res["indexed_count"]
        processed_unit["pipeline_variant"] = "with_pos_tagging"

        return processed_unit


# Baseline (Without POS) and Active POS-guided instances
pipeline_a_baseline = PipelineAProcessor()
pipeline_a_pos = PipelineAPosProcessor()
# Default active pipeline instance
pipeline_a = pipeline_a_pos
