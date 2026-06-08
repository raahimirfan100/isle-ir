"""Rocchio relevance feedback.

Owner:    Abubakar Ahmed (SSOT §2)
Phase:    Bonus (B-A1)
STATUS:   enhancement

Implements the Rocchio update rule so users can refine results by marking
papers relevant / not-relevant in the UI. Course tie-in: interactive relevance
feedback (Manning Ch. 9 — explicit, not pseudo).
"""

from __future__ import annotations

import numpy as np


def rocchio(
    q_emb: np.ndarray,
    rel_embs: list[np.ndarray],
    nrel_embs: list[np.ndarray],
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> np.ndarray:
    """Return updated query embedding after Rocchio update.

    q_new = α · q_orig + β · mean(rel_embs) − γ · mean(nrel_embs)

    Parameters
    ----------
    q_emb:
        Original L2-normalised query embedding (768-dim float32).
    rel_embs:
        Embeddings of papers marked relevant. May be empty.
    nrel_embs:
        Embeddings of papers marked non-relevant. May be empty.

    Returns
    -------
    Updated query embedding, L2-normalised float32 array.
    """
    updated = alpha * q_emb.flatten().astype(np.float32)

    if rel_embs:
        rel_mean = np.mean([e.flatten().astype(np.float32) for e in rel_embs], axis=0)
        updated = updated + beta * rel_mean

    if nrel_embs:
        nrel_mean = np.mean([e.flatten().astype(np.float32) for e in nrel_embs], axis=0)
        updated = updated - gamma * nrel_mean

    norm = float(np.linalg.norm(updated))
    if norm > 0:
        updated = updated / norm

    return updated
