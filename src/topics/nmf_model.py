"""NMF topic model — CPU baseline.

Owner:    Abubakar Ahmed (SSOT §2)
Contract: E — run_nmf (SSOT §6E)
Phase:    3 — Topic Modeling (IMPLEMENTATION.md §3.1)
STATUS:   core

TF-IDF document representation + scikit-learn NMF. Course tie-in: Week 5
(Vector Space Model / TF-IDF), Week 12 (text clustering).
"""

from __future__ import annotations

import logging

import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import NMF_K_RANGE

logger = logging.getLogger(__name__)


def _tokenize_docs(docs: list[str]) -> list[list[str]]:
    return [doc.lower().split() for doc in docs]


def _build_tfidf_matrix(docs: list[str]) -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        max_features=15_000,
        stop_words="english",
        min_df=2,
        max_df=0.90,
    )
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


def _topic_keywords(
    components: np.ndarray,
    vocabulary: np.ndarray,
    top_n: int = 10,
) -> list[list[str]]:
    topics: list[list[str]] = []
    for row in components:
        top_indices = row.argsort()[-top_n:][::-1]
        topics.append([str(vocabulary[index]) for index in top_indices])
    return topics


def find_optimal_k(docs: list[str], k_range: range = NMF_K_RANGE) -> int:
    """Sweep k, return the value with the best c_v coherence (gensim)."""
    if not docs:
        raise ValueError("docs must not be empty")

    tokenized = _tokenize_docs(docs)
    dictionary = Dictionary(tokenized)
    vectorizer, matrix = _build_tfidf_matrix(docs)
    vocabulary = vectorizer.get_feature_names_out()

    best_k: int | None = None
    best_score = float("-inf")

    for k in k_range:
        model = NMF(n_components=k, random_state=42, max_iter=500, init="nndsvda")
        model.fit(matrix)
        topics_words = _topic_keywords(model.components_, vocabulary)
        coherence = CoherenceModel(
            topics=topics_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        score = float(coherence.get_coherence())
        logger.info("NMF coherence k=%d c_v=%.4f", k, score)

        if score > best_score:
            best_k = k
            best_score = score

    if best_k is None:
        raise RuntimeError("No valid k values were evaluated")

    logger.info("Selected NMF k=%d with c_v=%.4f", best_k, best_score)
    return best_k


def run_nmf(
    docs: list[str], n_topics: int
) -> tuple[list[dict], list[int], list[float]]:
    """Fit NMF. Returns ``(topic_info, doc_topics, doc_probs)`` — Contract E.

    topic_info: [{"topic_id", "keywords", "doc_count", "label"}, ...]
    doc_topics: one topic id per document. doc_probs: assigned-topic weight.
    """
    if not docs:
        raise ValueError("docs must not be empty")

    vectorizer, matrix = _build_tfidf_matrix(docs)
    vocabulary = vectorizer.get_feature_names_out()

    model = NMF(n_components=n_topics, random_state=42, max_iter=1000, init="nndsvda")
    doc_topic_matrix = model.fit_transform(matrix)

    doc_topics = doc_topic_matrix.argmax(axis=1).tolist()
    doc_probs = doc_topic_matrix.max(axis=1).tolist()

    topic_info: list[dict] = []
    for topic_id, row in enumerate(model.components_):
        top_indices = row.argsort()[-10:][::-1]
        keywords = [str(vocabulary[index]) for index in top_indices]
        doc_count = int(sum(1 for assignment in doc_topics if assignment == topic_id))
        topic_info.append(
            {
                "topic_id": int(topic_id),
                "keywords": keywords,
                "doc_count": doc_count,
                "label": ", ".join(keywords[:3]),
            }
        )

    return topic_info, doc_topics, doc_probs
