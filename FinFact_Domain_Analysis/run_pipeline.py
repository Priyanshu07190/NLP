"""Master Pipeline Orchestrator for FinFact Domain-Specific IR System.

Usage:
  python run_pipeline.py --all            # Run Ingestion, Indexing, Benchmark & Comparison
  python run_pipeline.py --ingest         # Run Ingestion only
  python run_pipeline.py --index          # Build POS-Guided Positional Inverted Index
  python run_pipeline.py --evaluate       # Execute 15-Query TREC Benchmark
  python run_pipeline.py --compare        # Compare Without-POS vs With-POS side-by-side
  python run_pipeline.py --serve          # Launch FastAPI & Web GUI (POS-Guided)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
import shutil

import yaml

from src.evaluation.benchmark import compare_pipelines, run_benchmark_evaluation
from src.indexing.index_builder import build_pipeline_a_index
from src.indexing.inverted_index import PositionalInvertedIndex
from src.ingestion.ingest import ingest_finfact
from src.preprocessing.pipeline import pipeline_a_baseline, pipeline_a_pos
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


def step_index_pos(cfg: dict, max_units: int = 0):
    logger.info("=== STEP 2: PIPELINE A+ PREPROCESSING & POSITIONAL INDEXING (WITH POS TAGGING) ===")
    t0 = time.time()
    index_pos_path = BASE_DIR / "results/index/positional_index_pos.json"
    store_pos_path = BASE_DIR / "results/index/unit_store_pos.json"
    active_index_path = BASE_DIR / cfg["paths"]["index_file"]
    active_store_path = BASE_DIR / cfg["paths"]["unit_store"]

    stats = build_pipeline_a_index(
        corpus_jsonl_path=BASE_DIR / cfg["paths"]["structured_corpus"],
        output_index_path=index_pos_path,
        output_unit_store_path=store_pos_path,
        processor=pipeline_a_pos,
        max_units=max_units,
    )

    # Make the POS-guided index the active default index for the engine and dashboard
    shutil.copyfile(index_pos_path, active_index_path)
    shutil.copyfile(store_pos_path, active_store_path)
    logger.info(f"POS Indexing completed in {time.time() - t0:.2f}s | Active default set to POS index | Stats: {stats}")
    return stats


def step_evaluate(cfg: dict):
    logger.info("=== STEP 3: BENCHMARK EVALUATION (15 CANONICAL QUERIES) ===")
    t0 = time.time()
    index_path = BASE_DIR / cfg["paths"]["index_file"]
    store_path = BASE_DIR / cfg["paths"]["unit_store"]

    if not index_path.exists() or not store_path.exists():
        logger.error("Index not found. Running indexing first...")
        step_index_pos(cfg)

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
        pipeline_name="Pipeline A+ (With POS Tagging)",
    )
    logger.info(f"Benchmark completed in {time.time() - t0:.2f}s | MAP: {report['summary']['MAP']} | MRR: {report['summary']['MRR']}")
    return report


def step_compare(cfg: dict):
    logger.info("=== STEP 4: SIDE-BY-SIDE PIPELINE COMPARISON (WITHOUT POS vs WITH POS) ===")
    t0 = time.time()
    base_idx_path = BASE_DIR / "results/index/positional_index_baseline.json"
    base_store_path = BASE_DIR / "results/index/unit_store_baseline.json"
    pos_idx_path = BASE_DIR / "results/index/positional_index_pos.json"
    pos_store_path = BASE_DIR / "results/index/unit_store_pos.json"

    # If POS index hasn't been built yet, build it
    if not pos_idx_path.exists() or not pos_store_path.exists():
        step_index_pos(cfg)

    # If baseline index doesn't exist, build baseline
    if not base_idx_path.exists() or not base_store_path.exists():
        logger.info("Building baseline index for comparison...")
        build_pipeline_a_index(
            corpus_jsonl_path=BASE_DIR / cfg["paths"]["structured_corpus"],
            output_index_path=base_idx_path,
            output_unit_store_path=base_store_path,
            processor=pipeline_a_baseline,
        )

    index_base = PositionalInvertedIndex.load(base_idx_path)
    with open(base_store_path, "r", encoding="utf-8") as f:
        store_base = json.load(f)

    engine_base = SearchEngine(
        index=index_base,
        unit_store=store_base,
        term_weight=cfg["retrieval"]["term_match_weight"],
        phrase_bonus=cfg["retrieval"]["exact_phrase_bonus"],
        log_tf_weight=cfg["retrieval"]["log_tf_weight"],
    )

    index_pos = PositionalInvertedIndex.load(pos_idx_path)
    with open(pos_store_path, "r", encoding="utf-8") as f:
        store_pos = json.load(f)

    engine_pos = SearchEngine(
        index=index_pos,
        unit_store=store_pos,
        term_weight=cfg["retrieval"]["term_match_weight"],
        phrase_bonus=cfg["retrieval"]["exact_phrase_bonus"],
        log_tf_weight=cfg["retrieval"]["log_tf_weight"],
    )

    comp_report = compare_pipelines(
        engine_baseline=engine_base,
        engine_pos=engine_pos,
        output_json_path=BASE_DIR / "results/evaluation/pos_comparison_report.json",
        output_csv_path=BASE_DIR / "results/evaluation/pos_comparison_table.csv",
    )
    logger.info(f"Comparison completed in {time.time() - t0:.2f}s!")
    return comp_report


def step_serve(cfg: dict):
    logger.info("=== LAUNCHING FASTAPI SERVER & WEB GUI (WITH POS TAGGING) ===")
    import uvicorn
    host = cfg["server"]["host"]
    port = cfg["server"]["port"]
    logger.info(f"Access Web GUI at http://{host}:{port}")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(description="FinFact NLP Pipeline Runner")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: ingest, index POS, evaluate, compare")
    parser.add_argument("--ingest", action="store_true", help="Run dataset ingestion")
    parser.add_argument("--index", action="store_true", help="Run Pipeline A+ POS indexing")
    parser.add_argument("--evaluate", action="store_true", help="Run 15-query benchmark")
    parser.add_argument("--compare", action="store_true", help="Run side-by-side comparison (without POS vs with POS)")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI web server")
    parser.add_argument("--max-units", type=int, default=0, help="Cap units for fast smoke testing")

    args = parser.parse_args()
    cfg = load_config()

    if not any([args.all, args.ingest, args.index, args.evaluate, args.compare, args.serve]):
        args.all = True

    if args.all:
        step_ingest(cfg)
        step_index_pos(cfg, max_units=args.max_units)
        step_evaluate(cfg)
        step_compare(cfg)
        logger.info("Full Pipeline with POS Tagging & Comparison Executed Successfully!")
        return

    if args.ingest:
        step_ingest(cfg)

    if args.index:
        step_index_pos(cfg, max_units=args.max_units)

    if args.evaluate:
        step_evaluate(cfg)

    if args.compare:
        step_compare(cfg)

    if args.serve:
        step_serve(cfg)


if __name__ == "__main__":
    main()
