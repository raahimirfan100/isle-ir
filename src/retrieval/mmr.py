"""Maximal Marginal Relevance diversity re-ranking.

Owner:    Raahim Irfan (SSOT §2)
Phase:    Bonus (B-R1)
STATUS:   enhancement

After cross-encoder reranking, applies MMR to balance relevance vs. diversity.
Uses precomputed FAISS embeddings — no new model inference required.
Course tie-in: diversity-relevance tradeoff (Manning Ch. 9).
"""

from __future__ import annotations

import numpy as np


def mmr(
    query_emb: np.ndarray,
    candidates: list[tuple[str, float, np.ndarray]],
    lam: float = 0.5,
    k: int = 20,
) -> list[tuple[str, float]]:
    """Greedy MMR selection balancing relevance and diversity.

    MMR = argmax[ λ · sim(d, q) − (1−λ) · max_{s∈S} sim(d, s) ]

    Parameters
    ----------
    query_emb:
        L2-normalised 768-dim query vector.
    candidates:
        List of (arxiv_id, relevance_score, embedding) — embeddings must be
        L2-normalised float32 arrays of the same dimension as query_emb.
    lam:
        Trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
    k:
        Maximum results to return.

    Returns
    -------
    [(arxiv_id, original_relevance_score), ...] in MMR order.
    """
    if not candidates:
        return []

    k = min(k, len(candidates))
    q = query_emb.flatten().astype(np.float32)

    selected: list[tuple[str, float, np.ndarray]] = []
    remaining = [
        (arxiv_id, score, emb.flatten().astype(np.float32))
        for arxiv_id, score, emb in candidates
    ]

    while len(selected) < k and remaining:
        best_idx = -1
        best_mmr = float("-inf")

        for i, (_arxiv_id, _rel_score, emb) in enumerate(remaining):
            sim_q = float(np.dot(q, emb))

            if selected:
                sim_sel = max(float(np.dot(emb, s_emb)) for _, _, s_emb in selected)
            else:
                sim_sel = 0.0

            score = lam * sim_q - (1.0 - lam) * sim_sel
            if score > best_mmr:
                best_mmr = score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return [(arxiv_id, rel_score) for arxiv_id, rel_score, _ in selected]
