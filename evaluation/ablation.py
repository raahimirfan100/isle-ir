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
  4. Hybrid + cross-encoder reranker
  5. Full system (+ query expansion)
"""

from __future__ import annotations


def run_ablation() -> None:
    """Evaluate all variants on the labelled test queries; write the CSV."""
    # TODO(abubakar, phase-6): load test_queries.json (relevance must be
    #   filled in first); compute P@5, P@10, MRR, MAP per variant; write
    #   evaluation/results/ablation_table.csv.
    raise NotImplementedError


if __name__ == "__main__":
    run_ablation()
