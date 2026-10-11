"""Master Pipeline Orchestrator for FinFact Domain-Specific IR System.

Usage:
  python run_pipeline.py --all            # Run Ingestion, Indexing, Benchmark & Tests
  python run_pipeline.py --ingest         # Run Ingestion only
  python run_pipeline.py --index          # Build Positional Inverted Index
  python run_pipeline.py --evaluate       # Execute 15-Query TREC Benchmark
  python run_pipeline.py --serve          # Launch FastAPI & Web GUI
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

from src.evaluation.benchmark import run_benchmark_evaluation
from src.indexing.index_builder import build_pipeline_a_index
from src.indexing.inverted_index import PositionalInvertedIndex
from src.ingestion.ingest import ingest_finfact
from src.retrieval.search_engine import SearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finfact.runner")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "pipeline_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def step_ingest(cfg: dict):
    logger.info("=== STEP 1: INGESTION & CONTENT-UNIT EXTRACTION ===")
    t0 = time.time()
    summary = ingest_finfact(
        raw_json_path=cfg["paths"]["raw_dataset"],
        output_corpus_jsonl=BASE_DIR / cfg["paths"]["structured_corpus"],
        output_claim_registry=BASE_DIR / cfg["paths"]["claim_registry"],
        output_unit_registry=BASE_DIR / cfg["paths"]["unit_registry"],
        output_summary_json=BASE_DIR / cfg["paths"]["summary_stats"],
    )
    logger.info(f"Ingestion completed in {time.time() - t0:.2f}s | Summary: {summary}")
    return summary


def step_index(cfg: dict, max_units: int = 0):
    logger.info("=== STEP 2: PIPELINE A PREPROCESSING & POSITIONAL INDEXING ===")
    t0 = time.time()
    stats = build_pipeline_a_index(
        corpus_jsonl_path=BASE_DIR / cfg["paths"]["structured_corpus"],
        output_index_path=BASE_DIR / cfg["paths"]["index_file"],
        output_unit_store_path=BASE_DIR / cfg["paths"]["unit_store"],
        max_units=max_units,
    )
    logger.info(f"Indexing completed in {time.time() - t0:.2f}s | Stats: {stats}")
    return stats


def step_evaluate(cfg: dict):
    logger.info("=== STEP 3: BENCHMARK EVALUATION (15 CANONICAL QUERIES) ===")
    t0 = time.time()
    index_path = BASE_DIR / cfg["paths"]["index_file"]
    store_path = BASE_DIR / cfg["paths"]["unit_store"]

    if not index_path.exists() or not store_path.exists():
        logger.error("Index not found. Running indexing first...")
        step_index(cfg)

    index = PositionalInvertedIndex.load(index_path)
    with open(store_path, "r", encoding="utf-8") as f:
        unit_store = json.load(f)

    engine = SearchEngine(
        index=index,
        unit_store=unit_store,
        term_weight=cfg["retrieval"]["term_match_weight"],
        phrase_bonus=cfg["retrieval"]["exact_phrase_bonus"],
        log_tf_weight=cfg["retrieval"]["log_tf_weight"],
    )

    report = run_benchmark_evaluation(
        engine=engine,
        output_json_path=BASE_DIR / cfg["paths"]["benchmark_results"],
        output_csv_path=BASE_DIR / cfg["paths"]["benchmark_csv"],
    )
    logger.info(f"Benchmark completed in {time.time() - t0:.2f}s | MAP: {report['summary']['MAP']} | MRR: {report['summary']['MRR']}")
    return report


def step_serve(cfg: dict):
    logger.info("=== STEP 4: LAUNCHING FASTAPI SERVER & WEB GUI ===")
    import uvicorn
    host = cfg["server"]["host"]
    port = cfg["server"]["port"]
    logger.info(f"Access Web GUI at http://{host}:{port}")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(description="FinFact NLP Pipeline Runner")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: ingest, index, evaluate")
    parser.add_argument("--ingest", action="store_true", help="Run dataset ingestion")
    parser.add_argument("--index", action="store_true", help="Run Pipeline A indexing")
    parser.add_argument("--evaluate", action="store_true", help="Run 15-query benchmark")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI web server")
    parser.add_argument("--max-units", type=int, default=0, help="Cap units for fast smoke testing")

    args = parser.parse_args()
    cfg = load_config()

    if not any([args.all, args.ingest, args.index, args.evaluate, args.serve]):
        # Default behavior: run all
        args.all = True

    if args.all:
        step_ingest(cfg)
        step_index(cfg, max_units=args.max_units)
        step_evaluate(cfg)
        logger.info("Full Pipeline Executed Successfully!")
        return

    if args.ingest:
        step_ingest(cfg)

    if args.index:
        step_index(cfg, max_units=args.max_units)

    if args.evaluate:
        step_evaluate(cfg)

    if args.serve:
        step_serve(cfg)


if __name__ == "__main__":
    main()
