"""Query expansion — WordNet synonyms + pseudo-relevance feedback.

Owner:    Abubakar Ahmed (SSOT §2)
Contract: D — consumed by pipeline.full_retrieval_pipeline (SSOT §6D)
Phase:    2 — Hybrid Retrieval (SSOT §10 Above-and-Beyond)
STATUS:   enhancement — schedule after the core pipeline works

Course tie-in: Week 8 relevance feedback (PRF / Rocchio).
"""

from __future__ import annotations


def expand_query(query: str, bm25_index=None) -> str:
    """Return ``query`` expanded with synonyms / PRF feedback terms."""
    # TODO(abubakar, enhancement): step 1 — WordNet synonyms; step 2 — PRF
    #   using the top-3 BM25 results' TF-IDF terms.
    raise NotImplementedError
