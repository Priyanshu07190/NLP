"""Deterministic Ingestion and Content-Unit Extraction for Fin-Fact Dataset.

Maps each claim record into structured, traceable content units:
- Claims (unit_type='claim')
- Justifications (unit_type='justification')
- Digests (unit_type='digest')
- Evidence sentences (unit_type='evidence')
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finfact.ingest")


def ingest_finfact(
    raw_json_path: str | Path,
    output_corpus_jsonl: str | Path,
    output_claim_registry: str | Path,
    output_unit_registry: str | Path,
    output_summary_json: str | Path,
) -> Dict[str, Any]:
    raw_path = Path(raw_json_path).resolve()
    corpus_path = Path(output_corpus_jsonl).resolve()
    claim_reg_path = Path(output_claim_registry).resolve()
    unit_reg_path = Path(output_unit_registry).resolve()
    summary_path = Path(output_summary_json).resolve()

    for p in [corpus_path, claim_reg_path, unit_reg_path, summary_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw Fin-Fact dataset not found at {raw_path}")

    logger.info(f"Loading raw Fin-Fact dataset from {raw_path}...")
    with open(raw_path, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(records):,} records. Extracting content units...")

    units: List[Dict[str, Any]] = []
    claim_rows: List[Dict[str, Any]] = []
    unit_rows: List[Dict[str, Any]] = []

    label_counts: Dict[str, int] = {}
    unit_type_counts: Dict[str, int] = {}
    total_words = 0
    total_chars = 0

    for idx, item in enumerate(records, start=1):
        claim_id = f"CLM_{idx:04d}"
        url = item.get("url") or ""
        claim_text = (item.get("claim") or "").strip()
        author = item.get("author") or ""
        posted = item.get("posted") or ""
        justification = (item.get("justification") or "").strip()
        sci_digest_list = item.get("sci_digest") or []
        sci_digest = " ".join([d.strip() for d in sci_digest_list if isinstance(d, str)]).strip()
        issues = item.get("issues") or []
        evidence_list = item.get("evidence") or []
        image_data = item.get("image_data") or []
        label = (item.get("label") or "NEI").strip().casefold()
        if label not in {"true", "false", "nei"}:
            label = "nei"

        label_counts[label] = label_counts.get(label, 0) + 1

        claim_rows.append({
            "claim_id": claim_id,
            "url": url,
            "claim": claim_text,
            "author": author,
            "posted": posted,
            "label": label,
            "evidence_count": len(evidence_list),
            "image_count": len(image_data),
            "issues": ";".join(issues) if isinstance(issues, list) else str(issues),
        })

        # 1. Claim Unit
        if claim_text:
            c_unit_id = f"{claim_id}_C"
            c_words = len(claim_text.split())
            c_chars = len(claim_text)
            total_words += c_words
            total_chars += c_chars
            unit_type_counts["claim"] = unit_type_counts.get("claim", 0) + 1

            u_claim = {
                "unit_id": c_unit_id,
                "claim_id": claim_id,
                "unit_type": "claim",
                "text": claim_text,
                "parent_claim": claim_text,
                "label": label,
                "author": author,
                "posted": posted,
                "url": url,
                "issues": issues,
                "hrefs": [],
            }
            units.append(u_claim)
            unit_rows.append({
                "unit_id": c_unit_id,
                "claim_id": claim_id,
                "unit_type": "claim",
                "char_count": c_chars,
                "word_count": c_words,
                "label": label,
            })

        # 2. Justification Unit
        if justification:
            j_unit_id = f"{claim_id}_J"
            j_words = len(justification.split())
            j_chars = len(justification)
            total_words += j_words
            total_chars += j_chars
            unit_type_counts["justification"] = unit_type_counts.get("justification", 0) + 1

            u_just = {
                "unit_id": j_unit_id,
                "claim_id": claim_id,
                "unit_type": "justification",
                "text": justification,
                "parent_claim": claim_text,
                "label": label,
                "author": author,
                "posted": posted,
                "url": url,
                "issues": issues,
                "hrefs": [],
            }
            units.append(u_just)
            unit_rows.append({
                "unit_id": j_unit_id,
                "claim_id": claim_id,
                "unit_type": "justification",
                "char_count": j_chars,
                "word_count": j_words,
                "label": label,
            })

        # 3. Sci Digest Unit (if present)
        if sci_digest:
            d_unit_id = f"{claim_id}_D"
            d_words = len(sci_digest.split())
            d_chars = len(sci_digest)
            total_words += d_words
            total_chars += d_chars
            unit_type_counts["digest"] = unit_type_counts.get("digest", 0) + 1

            u_digest = {
                "unit_id": d_unit_id,
                "claim_id": claim_id,
                "unit_type": "digest",
                "text": sci_digest,
                "parent_claim": claim_text,
                "label": label,
                "author": author,
                "posted": posted,
                "url": url,
                "issues": issues,
                "hrefs": [],
            }
            units.append(u_digest)
            unit_rows.append({
                "unit_id": d_unit_id,
                "claim_id": claim_id,
                "unit_type": "digest",
                "char_count": d_chars,
                "word_count": d_words,
                "label": label,
            })

        # 4. Evidence Units
        for ev_idx, ev in enumerate(evidence_list, start=1):
            ev_sentence = (ev.get("sentence") or "").strip()
            if not ev_sentence:
                continue
            e_unit_id = f"{claim_id}_E{ev_idx:02d}"
            e_words = len(ev_sentence.split())
            e_chars = len(ev_sentence)
            total_words += e_words
            total_chars += e_chars
            unit_type_counts["evidence"] = unit_type_counts.get("evidence", 0) + 1

            u_ev = {
                "unit_id": e_unit_id,
                "claim_id": claim_id,
                "unit_type": "evidence",
                "text": ev_sentence,
                "parent_claim": claim_text,
                "label": label,
                "author": author,
                "posted": posted,
                "url": url,
                "issues": issues,
                "hrefs": ev.get("hrefs") or [],
            }
            units.append(u_ev)
            unit_rows.append({
                "unit_id": e_unit_id,
                "claim_id": claim_id,
                "unit_type": "evidence",
                "char_count": e_chars,
                "word_count": e_words,
                "label": label,
            })

    logger.info(f"Writing {len(units):,} content units to {corpus_path}...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    logger.info(f"Writing claim registry to {claim_reg_path}...")
    if claim_rows:
        with open(claim_reg_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(claim_rows[0].keys()))
            writer.writeheader()
            writer.writerows(claim_rows)

    logger.info(f"Writing unit registry to {unit_reg_path}...")
    if unit_rows:
        with open(unit_reg_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(unit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(unit_rows)

    summary = {
        "total_records": len(records),
        "total_content_units": len(units),
        "total_words": total_words,
        "total_characters": total_chars,
        "label_distribution": label_counts,
        "unit_type_distribution": unit_type_counts,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Ingestion complete! Total units: {len(units):,}, Total words: {total_words:,}")
    return summary


if __name__ == "__main__":
    from ..config import load_config
    cfg = load_config()
    ingest_finfact(
        cfg["paths"]["raw_dataset"],
        cfg["paths"]["structured_corpus"],
        cfg["paths"]["claim_registry"],
        cfg["paths"]["unit_registry"],
        cfg["paths"]["summary_stats"],
    )
