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

    Contract D (SSOT §6D). Core stages: BM25 + FAISS + RRF.
    Enhancement stages (query expansion, cross-encoder rerank) are gated by
    the boolean flags — pass expand_query=False, rerank=False until those
    modules are ready (TODO items below).
    """
    # TODO(raahim, phase-2): wire BM25 + FAISS + RRF core stages.
    #   bm25_index.search(query, top_k=TOP_K_EACH) → bm25_results
    #   faiss_index.search(query, top_k=TOP_K_EACH) → faiss_results
    #   from retrieval.rrf import reciprocal_rank_fusion
    #   fused = reciprocal_rank_fusion([bm25_results, faiss_results], top_n=RRF_TOP_N)
    #   return fused[:RERANK_TOP_K]
    #
    # TODO(raahim, enhancement): if expand_query, call query_expansion.expand()
    #   first and pass expanded query to both indexes.
    # TODO(raahim, enhancement): if rerank, pass fused results through
    #   reranker.rerank(query, candidates, papers_lookup) before returning.
    #
    # Blocked on: Abubakar's bm25_index.py and rrf.py (SSOT §2).
    raise NotImplementedError
