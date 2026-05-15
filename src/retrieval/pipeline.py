"""End-to-end hybrid retrieval pipeline.

Owner:    Raahim Irfan (SSOT §2)
Contract: D — full_retrieval_pipeline (SSOT §6D)
Phase:    2 — Hybrid Retrieval
STATUS:   core

Wires the retrieval stages: query expansion → BM25 + FAISS → RRF → cross-encoder
rerank. The core path is BM25 + FAISS + RRF; expansion and reranking are
enhancement stages gated by the boolean flags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.bm25_index import BM25Index
    from retrieval.faiss_index import FAISSIndex


def full_retrieval_pipeline(
    query: str,
    bm25_index: BM25Index,
    faiss_index: FAISSIndex,
    papers_lookup: dict[str, dict],
    expand_query: bool = True,
    rerank: bool = True,
) -> list[tuple[str, float]]:
    """Return the top-20 [(arxiv_id, final_score), ...] for ``query``.

    Contract D (SSOT §6D). Core stages: BM25 + FAISS + RRF.
    """
    # TODO(raahim, phase-2): wire the core stages — BM25 + FAISS + RRF — first.
    # TODO(raahim, enhancement): gate query_expansion / reranker on the flags;
    #   until those modules land, callers should pass expand_query=rerank=False.
    raise NotImplementedError
