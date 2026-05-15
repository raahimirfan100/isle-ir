"""Cross-encoder reranker.

Owner:    Raahim Irfan (SSOT §2)
Contract: D — consumed by pipeline.full_retrieval_pipeline (SSOT §6D)
Phase:    2 — Hybrid Retrieval (SSOT §10 Above-and-Beyond)
STATUS:   enhancement — schedule after the core pipeline works

Re-scores (query, abstract) pairs with a cross-encoder for a final precision
pass over the RRF top-N. Model name: config.RERANKER_MODEL.
"""

from __future__ import annotations


def rerank(
    query: str,
    candidates: list[tuple[str, float]],
    papers_lookup: dict[str, dict],
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Re-score ``candidates`` with the cross-encoder; return the top_k."""
    # TODO(raahim, enhancement): load config.RERANKER_MODEL, score (query,
    #   abstract) pairs, re-sort. Latency target < 3s for 50 pairs (SSOT §10).
    raise NotImplementedError
