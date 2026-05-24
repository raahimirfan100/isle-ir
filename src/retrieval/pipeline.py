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

import logging
from typing import TYPE_CHECKING

from app.config import RERANK_TOP_K, RRF_TOP_N, TOP_K_EACH
from retrieval.rrf import reciprocal_rank_fusion

if TYPE_CHECKING:
    from retrieval.bm25_index import BM25Index
    from retrieval.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


def full_retrieval_pipeline(
    query: str,
    bm25_index: BM25Index,
    faiss_index: FAISSIndex,
    papers_lookup: dict[str, dict],
    expand_query: bool = True,
    rerank: bool = True,
) -> list[tuple[str, float]]:
    """Return the top-20 [(arxiv_id, final_score), ...] for ``query``.

    Contract D (SSOT §6D). Core: BM25 + FAISS → RRF.
    expand_query and rerank flags wire in the enhancement stages.
    """
    bm25_results = bm25_index.search(query, top_k=TOP_K_EACH)
    faiss_results = faiss_index.search(query, top_k=TOP_K_EACH)
    logger.debug("BM25: %d results, FAISS: %d results", len(bm25_results), len(faiss_results))

    fused = reciprocal_rank_fusion(bm25_results, faiss_results)
    candidates = fused[:RRF_TOP_N]

    # TODO(raahim, enhancement): if expand_query, expand query before search
    # TODO(raahim, enhancement): if rerank, re-score candidates with cross-encoder
    _ = expand_query
    _ = rerank

    return candidates[:RERANK_TOP_K]
