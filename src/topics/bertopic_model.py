"""BERTopic neural topic model.

Owner:    Raahim Irfan (SSOT §2)
Contract: E — run_bertopic (SSOT §6E)
Phase:    3 — Topic Modeling (IMPLEMENTATION.md §3.2)
STATUS:   core

Embedding-based topic modeling (UMAP + HDBSCAN + c-TF-IDF). Reuses the
precomputed embeddings from the FAISS build (embeddings.npy) — do not encode
the corpus twice. Course tie-in: Week 12-13 (clustering / neural IR).
"""

from __future__ import annotations

import numpy as np


def run_bertopic(
    docs: list[str], embeddings: np.ndarray
) -> tuple[list[dict], list[int], list[float]]:
    """Fit BERTopic. Returns ``(topic_info, doc_topics, doc_probs)`` — Contract E.

    IMPLEMENTATION.md §3.2 splits this into a model object plus a separate
    get_topic_info(); this wrapper must still return the Contract E tuple so
    graph.builder consumes NMF and BERTopic output identically. Outlier topic
    -1 stays in doc_topics but is excluded from topic_info.
    """
    # TODO(raahim, phase-3): build the BERTopic pipeline with precomputed
    #   embeddings; map get_topic_info() to the Contract E shape.
    raise NotImplementedError
