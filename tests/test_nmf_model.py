from __future__ import annotations

import logging

from topics.nmf_model import find_optimal_k, run_nmf

DOCS = [
    "neural network deep learning model representation",
    "deep neural networks for vision and representation learning",
    "neural model learning with embeddings and attention",
    "information retrieval search ranking bm25 query",
    "retrieval ranking search engine and lexical bm25 methods",
    "hybrid search information retrieval and dense ranking",
]


def test_find_optimal_k_logs_scores_and_returns_candidate(caplog) -> None:
    caplog.set_level(logging.INFO)

    best_k = find_optimal_k(DOCS, k_range=range(2, 4))

    assert best_k in {2, 3}
    assert "NMF coherence k=2" in caplog.text
    assert "NMF coherence k=3" in caplog.text
    assert "Selected NMF k=" in caplog.text


def test_run_nmf_matches_contract_e_shape() -> None:
    topic_info, doc_topics, doc_probs = run_nmf(DOCS, n_topics=2)

    assert len(topic_info) == 2
    assert len(doc_topics) == len(DOCS)
    assert len(doc_probs) == len(DOCS)
    assert sum(topic["doc_count"] for topic in topic_info) == len(DOCS)

    for topic in topic_info:
        assert set(topic) == {"topic_id", "keywords", "doc_count", "label"}
        assert isinstance(topic["topic_id"], int)
        assert isinstance(topic["keywords"], list)
        assert isinstance(topic["doc_count"], int)
        assert isinstance(topic["label"], str)
        assert topic["keywords"]

    assert all(isinstance(topic_id, int) for topic_id in doc_topics)
    assert all(prob >= 0 for prob in doc_probs)