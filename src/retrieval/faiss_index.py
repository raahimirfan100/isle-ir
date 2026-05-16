"""Dense semantic retrieval index (FAISS).

Owner:    Raahim Irfan (SSOT §2)
Contract: C — FAISSIndex (SSOT §6C)
Phase:    2 — Hybrid Retrieval (IMPLEMENTATION.md §2.2)
STATUS:   core

Encodes papers with the configured embedding model and serves
nearest-neighbour search via a FAISS index. The model and dimension come from
src/app/config.py (EMBEDDING_MODEL, EMBEDDING_DIM) — currently SPECTER2,
768-dim. SPECTER2 loads via the `adapters` library, not a plain
SentenceTransformer.
"""

from __future__ import annotations

import numpy as np


class FAISSIndex:
    """Dense index. Contract C — the public signatures are fixed (SSOT §6C)."""

    def build(self, papers: list[dict], batch_size: int = 128) -> None:
        """Encode papers and build + persist the FAISS index."""
        # TODO(raahim, phase-2): batch-encode; build IndexIVFFlat (train it
        #   before adding!); persist vectors.index, paper_ids.json, embeddings.npy.
        raise NotImplementedError

    def load(self) -> None:
        """Load the persisted index from config.FAISS_DIR."""
        # TODO(raahim, phase-2): read the index; optionally move it to GPU.
        raise NotImplementedError

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return [(arxiv_id, cosine_similarity), ...] sorted descending."""
        # TODO(raahim, phase-2): encode query, search, drop -1 slots.
        raise NotImplementedError

    def search_by_embedding(
        self, embedding: np.ndarray, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """'More Like This' — nearest neighbours of a precomputed vector."""
        # TODO(raahim, enhancement): powers the More Like This feature (SSOT §10).
        raise NotImplementedError
