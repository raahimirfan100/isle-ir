"""Retrieval ablation study.

Owner:   Abubakar Ahmed (SSOT §2)
Phase:   6 — Evaluation (SSOT §11)
STATUS:  core

Runs the five system variants over evaluation/test_queries.json and writes
evaluation/results/ablation_table.csv with P@5, P@10, MRR, MAP and ILD@10.

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

import numpy as np
import pandas as pd

from app.config import CORPUS_PATH, EVAL_RESULTS_DIR, FAISS_DIR, RERANK_TOP_K
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


def _ild_at_k(
  ranked_ids: list[str],
  pid_to_idx: dict[str, int],
  embeddings: np.ndarray,
  k: int = 10,
) -> float:
  """Intra-List Diversity at k — mean pairwise cosine distance in embedding space.

  ILD@k = (2 / k(k−1)) · Σ_{i<j} (1 − cosine_sim(emb_i, emb_j))
  """
  ids = [pid for pid in ranked_ids[:k] if pid in pid_to_idx]
  n = len(ids)
  if n < 2:
    return 0.0

  vecs = np.array([embeddings[pid_to_idx[pid]] for pid in ids], dtype=np.float32)
  norms = np.linalg.norm(vecs, axis=1, keepdims=True)
  norms = np.where(norms == 0, 1.0, norms)
  vecs = vecs / norms

  total = 0.0
  pairs = 0
  for i in range(n):
    for j in range(i + 1, n):
      total += 1.0 - float(np.dot(vecs[i], vecs[j]))
      pairs += 1

  return total / pairs if pairs else 0.0


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
  pid_to_idx: dict[str, int] | None = None,
  embeddings: np.ndarray | None = None,
) -> dict[str, float | str]:
  p5_scores: list[float] = []
  p10_scores: list[float] = []
  mrr_scores: list[float] = []
  map_scores: list[float] = []
  ild_scores: list[float] = []

  for query_row in query_rows:
    relevant_ids = set(query_row["relevant"])
    ranked_ids = ranker(query_row["query"])

    p5_scores.append(_precision_at_k(ranked_ids, relevant_ids, 5))
    p10_scores.append(_precision_at_k(ranked_ids, relevant_ids, 10))
    mrr_scores.append(_reciprocal_rank(ranked_ids, relevant_ids))
    map_scores.append(_average_precision(ranked_ids, relevant_ids))
    if pid_to_idx is not None and embeddings is not None:
      ild_scores.append(_ild_at_k(ranked_ids, pid_to_idx, embeddings, k=10))

  row: dict[str, float | str] = {
    "System": variant_name,
    "P@5": sum(p5_scores) / len(p5_scores),
    "P@10": sum(p10_scores) / len(p10_scores),
    "MRR": sum(mrr_scores) / len(mrr_scores),
    "MAP": sum(map_scores) / len(map_scores),
  }
  if ild_scores:
    row["ILD@10"] = sum(ild_scores) / len(ild_scores)
  return row


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

  # Load embeddings for ILD metric (B-A2)
  embeddings_path = FAISS_DIR / "embeddings.npy"
  embeddings: np.ndarray | None = None
  pid_to_idx: dict[str, int] | None = None
  if embeddings_path.exists():
    embeddings = np.load(str(embeddings_path))
    paper_ids_path = FAISS_DIR / "paper_ids.json"
    if paper_ids_path.exists():
      import json as _json
      with paper_ids_path.open() as f:
        paper_ids = _json.load(f)
      pid_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

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
    _evaluate_variant(query_rows, variant_name, ranker, pid_to_idx, embeddings)
    for variant_name, ranker in variant_runners
  ]

  output_dir = EVAL_RESULTS_DIR
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "ablation_table.csv"

  columns = ["System", "P@5", "P@10", "MRR", "MAP"]
  if embeddings is not None:
    columns.append("ILD@10")
  result_df = pd.DataFrame(rows, columns=columns)
  result_df.to_csv(output_path, index=False)
  print(f"Wrote {output_path} with {len(result_df)} rows using {len(corpus_df)} corpus documents.")


if __name__ == "__main__":
  run_ablation()
