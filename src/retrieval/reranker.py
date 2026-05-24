"""Cross-encoder reranker.

Owner:    Raahim Irfan (SSOT §2)
Contract: D — consumed by pipeline.full_retrieval_pipeline (SSOT §6D)
Phase:    2 — Hybrid Retrieval (SSOT §10 Above-and-Beyond)
STATUS:   enhancement — schedule after the core pipeline works

Re-scores (query, abstract) pairs with a cross-encoder for a final precision
pass over the RRF top-N. Model name: config.RERANKER_MODEL.
"""

from __future__ import annotations

import logging
import time

import torch
from sentence_transformers import CrossEncoder

from app.config import RERANK_TOP_K, RERANKER_MODEL

logger = logging.getLogger(__name__)

_model: CrossEncoder | None = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info("Loading CrossEncoder '%s' on %s.", RERANKER_MODEL, _device)
        _model = CrossEncoder(RERANKER_MODEL, device=_device)
    return _model


def rerank(
    query: str,
    candidates: list[tuple[str, float]],
    papers_lookup: dict[str, dict],
    top_k: int = RERANK_TOP_K,
) -> list[tuple[str, float]]:
    """Re-score ``candidates`` with the cross-encoder; return the top_k.

    Each candidate is a (arxiv_id, rrf_score) pair from RRF top-N.
    Returns (arxiv_id, ce_score) sorted descending by cross-encoder score.
    Latency target: <3s for 50 pairs on GPU (SSOT §10).
    """
    if not candidates:
        return []

    arxiv_ids = [aid for aid, _ in candidates]
    sentence_pairs = [
        (query, papers_lookup[aid].get("abstract", "") if aid in papers_lookup else "")
        for aid in arxiv_ids
    ]

    t0 = time.perf_counter()
    scores = _get_model().predict(sentence_pairs, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    logger.debug("CrossEncoder scored %d pairs in %.3fs.", len(sentence_pairs), elapsed)

    ranked = sorted(
        zip(arxiv_ids, (float(s) for s in scores), strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]
