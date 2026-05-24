"""Reciprocal Rank Fusion.

Owner:    Abubakar Ahmed (SSOT §2)
Contract: D — reciprocal_rank_fusion (SSOT §6D)
Phase:    2 — Hybrid Retrieval (IMPLEMENTATION.md §2.3)
STATUS:   core

Fuses ranked lists from BM25 and FAISS by rank position (Cormack et al.,
SIGIR 2009). Course tie-in: Week 7 evaluation / rank aggregation.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Returns merged [(arxiv_id, rrf_score), ...] sorted descending."""

    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
