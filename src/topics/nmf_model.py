"""NMF topic model — CPU baseline.

Owner:    Abubakar Ahmed (SSOT §2)
Contract: E — run_nmf (SSOT §6E)
Phase:    3 — Topic Modeling (IMPLEMENTATION.md §3.1)
STATUS:   core

TF-IDF document representation + scikit-learn NMF. Course tie-in: Week 5
(Vector Space Model / TF-IDF), Week 12 (text clustering).
"""

from __future__ import annotations


def find_optimal_k(docs: list[str], k_range: range = range(5, 21)) -> int:
    """Sweep k, return the value with the best c_v coherence (gensim)."""
    # TODO(abubakar, phase-3): coherence sweep — see IMPLEMENTATION.md §3.1.
    raise NotImplementedError


def run_nmf(
    docs: list[str], n_topics: int
) -> tuple[list[dict], list[int], list[float]]:
    """Fit NMF. Returns ``(topic_info, doc_topics, doc_probs)`` — Contract E.

    topic_info: [{"topic_id", "keywords", "doc_count", "label"}, ...]
    doc_topics: one topic id per document. doc_probs: assigned-topic weight.
    """
    # TODO(abubakar, phase-3): TF-IDF + NMF; emit the exact Contract E shape
    #   so graph.builder can consume it unchanged.
    raise NotImplementedError
