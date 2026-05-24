"""BM25 lexical retrieval index.

Owner:    Abubakar Ahmed (SSOT §2)
Contract: B — BM25Index (SSOT §6B)
Phase:    2 — Hybrid Retrieval (IMPLEMENTATION.md §2.1)
STATUS:   core

Pure-Python BM25 over the corpus using ``bm25s``. Course tie-in: Week 9
probabilistic retrieval model.
"""

from __future__ import annotations

from pathlib import Path

import bm25s
import joblib
import Stemmer

from app.config import BM25_B, BM25_DIR, BM25_K1


class BM25Index:
    """Lexical index. Contract B — the public signatures are fixed (SSOT §6B)."""

    def __init__(self) -> None:
        self.index_dir = BM25_DIR
        self.stemmer = Stemmer.Stemmer("porter")
        self.retriever: bm25s.BM25 | None = None
        self.paper_ids: list[str] = []

    def _retriever_path(self) -> Path:
        return self.index_dir / "retriever.joblib"

    def _paper_ids_path(self) -> Path:
        return self.index_dir / "paper_ids.joblib"

    def build(self, papers: list[dict]) -> None:
        """Build and persist the BM25 index from Contract A paper dicts."""
        corpus_texts = [paper["text"] for paper in papers]
        self.paper_ids = [paper["arxiv_id"] for paper in papers]

        tokenized = bm25s.tokenize(
            corpus_texts,
            stopwords="en",
            stemmer=self.stemmer,
        )

        self.retriever = bm25s.BM25(k1=BM25_K1, b=BM25_B)
        self.retriever.index(tokenized)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.retriever, self._retriever_path())
        joblib.dump(self.paper_ids, self._paper_ids_path())

    def load(self) -> None:
        """Load the persisted index from config.BM25_DIR."""
        retriever_path = self._retriever_path()
        paper_ids_path = self._paper_ids_path()
        if not retriever_path.exists() or not paper_ids_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self.index_dir}. Run build() first."
            )

        self.retriever = joblib.load(retriever_path)
        self.paper_ids = joblib.load(paper_ids_path)

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return [(arxiv_id, bm25_score), ...] sorted by descending score."""
        if self.retriever is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        query_tokens = bm25s.tokenize(
            query,
            stopwords="en",
            stemmer=self.stemmer,
        )
        results = self.retriever.retrieve(query_tokens, k=top_k)
        return [
            (self.paper_ids[idx], float(score))
            for idx, score in zip(results.documents[0], results.scores[0], strict=False)
            if idx != -1
        ]
