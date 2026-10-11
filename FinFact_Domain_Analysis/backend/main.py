"""FastAPI REST API & Interactive Search Engine Server.

Provides endpoints for Boolean retrieval, exact phrase verification,
claim-evidence inspection, and serves the Web GUI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import sys

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.indexing.inverted_index import PositionalInvertedIndex
from src.retrieval.query_parser import parse_query
from src.retrieval.search_engine import SearchEngine

CONFIG_PATH = BASE_DIR / "config" / "pipeline_config.yaml"
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"

# Load config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title="Fin-Fact NLP Domain-Specific Retrieval System",
    description="High-Precision Financial Fact-Checking and Evidence Retrieval Engine (Pipeline A: Lemmatization)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine state
index: Optional[PositionalInvertedIndex] = None
unit_store: Dict[str, Dict[str, Any]] = {}
engine: Optional[SearchEngine] = None
summary_stats: Dict[str, Any] = {}


@app.on_event("startup")
def startup_event():
    global index, unit_store, engine, summary_stats

    index_path = BASE_DIR / config["paths"]["index_file"]
    store_path = BASE_DIR / config["paths"]["unit_store"]
    stats_path = BASE_DIR / config["paths"]["summary_stats"]

    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            summary_stats = json.load(f)

    if index_path.exists() and store_path.exists():
        print(f"Loading positional index from {index_path}...")
        index = PositionalInvertedIndex.load(index_path)

        print(f"Loading unit store from {store_path}...")
        with open(store_path, "r", encoding="utf-8") as f:
            unit_store = json.load(f)

        engine = SearchEngine(
            index=index,
            unit_store=unit_store,
            term_weight=config["retrieval"]["term_match_weight"],
            phrase_bonus=config["retrieval"]["exact_phrase_bonus"],
            log_tf_weight=config["retrieval"]["log_tf_weight"],
        )
        print("Search Engine initialized successfully!")
    else:
        print("WARNING: Index files not found. Run pipeline indexing first.")


@app.get("/api/health")
def health():
    return {
        "status": "online" if engine is not None else "initializing",
        "index_loaded": engine is not None,
        "vocabulary_size": index.vocabulary_size if index else 0,
        "total_units": len(unit_store) if unit_store else 0,
        "total_postings": index.total_postings if index else 0,
    }


@app.get("/api/stats")
def stats():
    return {
        "corpus_summary": summary_stats,
        "index_summary": {
            "vocabulary_size": index.vocabulary_size if index else 0,
            "total_units": len(unit_store) if unit_store else 0,
            "total_postings": index.total_postings if index else 0,
            "pipeline": "Pipeline A (Lemmatization + Positional Postings)",
        },
    }


@app.get("/api/search")
def search(
    q: str = Query(..., description="Boolean query or exact phrase"),
    top_k: int = Query(10, ge=1, le=100),
    unit_type: str = Query("all", description="Filter by unit type (all, claim, evidence, justification, digest)"),
    label: str = Query("all", description="Filter by claim label (all, true, false, nei)"),
):
    if not engine:
        raise HTTPException(status_code=503, detail="Search engine index is not loaded yet.")

    try:
        res = engine.search(
            query_str=q,
            top_k=top_k,
            unit_type_filter=unit_type,
            label_filter=label,
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/claims/{claim_id}")
def get_claim(claim_id: str):
    if not unit_store:
        raise HTTPException(status_code=503, detail="Unit store not loaded.")

    matching_units = [u for u in unit_store.values() if u.get("claim_id") == claim_id]
    if not matching_units:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")

    claim_unit = next((u for u in matching_units if u.get("unit_type") == "claim"), None)
    justification_unit = next((u for u in matching_units if u.get("unit_type") == "justification"), None)
    digest_unit = next((u for u in matching_units if u.get("unit_type") == "digest"), None)
    evidence_units = [u for u in matching_units if u.get("unit_type") == "evidence"]

    return {
        "claim_id": claim_id,
        "claim": claim_unit.get("text", "") if claim_unit else "",
        "label": matching_units[0].get("label", "nei"),
        "author": matching_units[0].get("author", ""),
        "posted": matching_units[0].get("posted", ""),
        "url": matching_units[0].get("url", ""),
        "issues": matching_units[0].get("issues", []),
        "justification": justification_unit.get("text", "") if justification_unit else "",
        "digest": digest_unit.get("text", "") if digest_unit else "",
        "evidence_sentences": [
            {"unit_id": e["unit_id"], "sentence": e["text"], "hrefs": e.get("hrefs", [])}
            for e in evidence_units
        ],
    }


@app.get("/api/benchmark")
def get_benchmark():
    bench_file = BASE_DIR / config["paths"]["benchmark_results"]
    if not bench_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark results not generated yet.")
    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/", response_class=HTMLResponse)
def index_page():
    if FRONTEND_FILE.exists():
        return FileResponse(FRONTEND_FILE)
    return HTMLResponse("<h1>FinFact Search API is running. UI file missing.</h1>")
