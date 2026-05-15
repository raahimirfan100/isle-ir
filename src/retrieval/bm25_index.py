"""BM25 lexical retrieval index.

Owner:    Abubakar Ahmed (SSOT §2)
Contract: B — BM25Index (SSOT §6B)
Phase:    2 — Hybrid Retrieval (IMPLEMENTATION.md §2.1)
STATUS:   core

Pure-Python BM25 over the corpus using ``bm25s``. Course tie-in: Week 9
probabilistic retrieval model.
"""

from __future__ import annotations


class BM25Index:
    """Lexical index. Contract B — the public signatures are fixed (SSOT §6B)."""

    def build(self, papers: list[dict]) -> None:
        """Build and persist the BM25 index from Contract A paper dicts."""
        # TODO(abubakar, phase-2): tokenize (stopwords + PyStemmer), index with
        #   bm25s, persist to config.BM25_DIR. See IMPLEMENTATION.md §2.1.
        raise NotImplementedError

    def load(self) -> None:
        """Load the persisted index from config.BM25_DIR."""
        # TODO(abubakar, phase-2): load the bm25s index and paper_ids.
        raise NotImplementedError

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return [(arxiv_id, bm25_score), ...] sorted by descending score."""
        # TODO(abubakar, phase-2): tokenize query, retrieve top_k, map to ids.
        raise NotImplementedError
