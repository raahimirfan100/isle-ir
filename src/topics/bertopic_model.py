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

import logging

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from app.config import (
    BERTOPIC_MIN_CLUSTER,
    BERTOPIC_N_COMPONENTS,
    BERTOPIC_N_NEIGHBORS,
)

logger = logging.getLogger(__name__)


def _build_bertopic_model() -> BERTopic:
    """Construct a reproducible BERTopic model from config hyperparameters."""
    umap_model = UMAP(
        n_neighbors=BERTOPIC_N_NEIGHBORS,
        n_components=BERTOPIC_N_COMPONENTS,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=BERTOPIC_MIN_CLUSTER,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
    )

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=10,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True,
    )


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
