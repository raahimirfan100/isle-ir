"""Retrieval ablation study.

Owner:   Abubakar Ahmed (SSOT §2)
Phase:   6 — Evaluation (SSOT §11)
STATUS:  core

Runs the five system variants over evaluation/test_queries.json and writes
evaluation/results/ablation_table.csv with P@5, P@10, MRR and MAP.

Variants (SSOT §11):
  1. BM25 only
  2. Dense only (FAISS)
  3. Hybrid (RRF)
  4. Hybrid + Reranker
  5. Full system (+ query expansion)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.config import CORPUS_PATH, EVAL_RESULTS_DIR, RERANK_TOP_K
from retrieval.bm25_index import BM25Index
from retrieval.faiss_index import FAISSIndex
from retrieval.pipeline import full_retrieval_pipeline

EVAL_TOP_K = RERANK_TOP_K


def _load_queries() -> list[dict]:
  queries_path = Path(__file__).with_name("test_queries.json")
  with queries_path.open("r", encoding="utf-8") as handle:
    return json.load(handle)


def _load_corpus_lookup() -> tuple[pd.DataFrame, dict[str, dict]]:
  corpus_df = pd.read_parquet(CORPUS_PATH)
  papers_lookup = corpus_df.set_index("arxiv_id").to_dict("index")
  return corpus_df, papers_lookup


def _precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
  if k <= 0:
    return 0.0
  return sum(pid in relevant_ids for pid in ranked_ids[:k]) / k


def _reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
  for rank, pid in enumerate(ranked_ids, start=1):
    if pid in relevant_ids:
      return 1.0 / rank
  return 0.0


def _average_precision(ranked_ids: list[str], relevant_ids: set[str]) -> float:
  if not relevant_ids:
    return 0.0

  hits = 0
  precision_sum = 0.0
  for rank, pid in enumerate(ranked_ids, start=1):
    if pid in relevant_ids:
      hits += 1
      precision_sum += hits / rank

  return precision_sum / len(relevant_ids)


def _evaluate_variant(
  query_rows: list[dict],
  variant_name: str,
  ranker,
) -> dict[str, float | str]:
  p5_scores: list[float] = []
  p10_scores: list[float] = []
  mrr_scores: list[float] = []
  map_scores: list[float] = []

  for query_row in query_rows:
    relevant_ids = set(query_row["relevant"])
    ranked_ids = ranker(query_row["query"])

    p5_scores.append(_precision_at_k(ranked_ids, relevant_ids, 5))
    p10_scores.append(_precision_at_k(ranked_ids, relevant_ids, 10))
    mrr_scores.append(_reciprocal_rank(ranked_ids, relevant_ids))
    map_scores.append(_average_precision(ranked_ids, relevant_ids))

  return {
    "System": variant_name,
    "P@5": sum(p5_scores) / len(p5_scores),
    "P@10": sum(p10_scores) / len(p10_scores),
    "MRR": sum(mrr_scores) / len(mrr_scores),
    "MAP": sum(map_scores) / len(map_scores),
  }


def run_ablation() -> None:
  """Evaluate all variants on the labelled test queries; write the CSV."""
  query_rows = _load_queries()
  corpus_df, papers_lookup = _load_corpus_lookup()

  if not query_rows:
    raise ValueError("No test queries found in evaluation/test_queries.json.")

  bm25_index = BM25Index()
  bm25_index.load()

  faiss_index = FAISSIndex()
  faiss_index.load()

  variant_runners = [
    ("BM25 only", lambda query: [pid for pid, _ in bm25_index.search(query, top_k=EVAL_TOP_K)]),
    ("Dense only", lambda query: [pid for pid, _ in faiss_index.search(query, top_k=EVAL_TOP_K)]),
    (
      "Hybrid (RRF)",
      lambda query: [
        pid
        for pid, _ in full_retrieval_pipeline(
          query,
          bm25_index,
          faiss_index,
          papers_lookup,
          expand_query=False,
          rerank=False,
        )
      ],
    ),
    (
      "Hybrid + Reranker",
      lambda query: [
        pid
        for pid, _ in full_retrieval_pipeline(
          query,
          bm25_index,
          faiss_index,
          papers_lookup,
          expand_query=False,
          rerank=True,
        )
      ],
    ),
    (
      "Full system",
      lambda query: [
        pid
        for pid, _ in full_retrieval_pipeline(
          query,
          bm25_index,
          faiss_index,
          papers_lookup,
          expand_query=True,
          rerank=True,
        )
      ],
    ),
  ]

  rows = [
    _evaluate_variant(query_rows, variant_name, ranker)
    for variant_name, ranker in variant_runners
  ]

  output_dir = EVAL_RESULTS_DIR
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "ablation_table.csv"

  result_df = pd.DataFrame(rows, columns=["System", "P@5", "P@10", "MRR", "MAP"])
  result_df.to_csv(output_path, index=False)
  print(f"Wrote {output_path} with {len(result_df)} rows using {len(corpus_df)} corpus documents.")


if __name__ == "__main__":
  run_ablation()
