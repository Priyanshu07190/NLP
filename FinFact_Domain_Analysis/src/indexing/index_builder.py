"""Index Builder for Pipeline A & Pipeline A+ (POS-Guided).

Processes structured content units from corpus.jsonl and builds:
1. Positional Inverted Index (term -> unit_id -> positions)
2. Document Store (unit_id -> unit metadata, POS tags, keyphrases, and cleaned text)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from ..preprocessing.pipeline import pipeline_a_baseline, pipeline_a_pos
from .inverted_index import PositionalInvertedIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finfact.indexer")


def build_pipeline_a_index(
    corpus_jsonl_path: str | Path,
    output_index_path: str | Path,
    output_unit_store_path: str | Path,
    processor: Any = None,
    max_units: int = 0,
) -> Dict[str, Any]:
    corpus_file = Path(corpus_jsonl_path)
    index_file = Path(output_index_path)
    store_file = Path(output_unit_store_path)

    index_file.parent.mkdir(parents=True, exist_ok=True)
    store_file.parent.mkdir(parents=True, exist_ok=True)

    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    active_processor = processor if processor is not None else pipeline_a_pos
    proc_name = getattr(active_processor, "name", "Pipeline A")
    logger.info(f"Building Positional Inverted Index using [{proc_name}] from {corpus_file}...")

    index = PositionalInvertedIndex()
    unit_store: Dict[str, Dict[str, Any]] = {}

    total_tokens = 0
    total_indexed_terms = 0
    unit_count = 0

    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw_unit = json.loads(line)
            unit_id = raw_unit["unit_id"]

            processed = active_processor.process_unit(raw_unit)

            index.add_unit_terms(unit_id, processed["indexed_terms"])

            # Store compact token POS previews (first 20 tokens) for frontend inspection
            pos_preview = [
                {"text": t["text"], "pos": t.get("pos_tag", "")}
                for t in processed["tokens"][:20]
                if t.get("pos_tag")
            ]

            unit_store[unit_id] = {
                "unit_id": unit_id,
                "claim_id": processed["claim_id"],
                "unit_type": processed["unit_type"],
                "text": processed["text"],
                "cleaned_text": processed["cleaned_text"],
                "parent_claim": processed.get("parent_claim", ""),
                "label": processed.get("label", "nei"),
                "author": processed.get("author", ""),
                "posted": processed.get("posted", ""),
                "url": processed.get("url", ""),
                "issues": processed.get("issues", []),
                "hrefs": processed.get("hrefs", []),
                "entities": processed.get("entities", []),
                "noun_phrases": processed.get("noun_phrases", [])[:6],
                "pos_preview": pos_preview,
                "token_count": processed["token_count"],
                "pipeline_variant": processed.get("pipeline_variant", "with_pos_tagging"),
            }

            total_tokens += processed["token_count"]
            total_indexed_terms += processed["indexed_count"]
            unit_count += 1

            if unit_count % 5000 == 0:
                logger.info(f"[{proc_name}] Indexed {unit_count:,} units... (Current vocab: {index.vocabulary_size:,})")

            if max_units and unit_count >= max_units:
                logger.info(f"Reached max_units cap ({max_units}). Stopping early.")
                break

    logger.info(f"Saving Positional Inverted Index to {index_file}...")
    index.save(index_file)

    logger.info(f"Saving Unit Metadata Store to {store_file}...")
    with open(store_file, "w", encoding="utf-8") as f:
        json.dump(unit_store, f, ensure_ascii=False)

    stats = {
        "pipeline_name": proc_name,
        "indexed_units": unit_count,
        "vocabulary_size": index.vocabulary_size,
        "total_postings": index.total_postings,
        "total_tokens": total_tokens,
        "total_indexed_terms": total_indexed_terms,
    }
    logger.info(f"Indexing complete! Stats: {stats}")
    return stats
