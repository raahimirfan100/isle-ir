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
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from app.config import (
    BERTOPIC_DIR,
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


def _get_topic_info(topic_model: BERTopic) -> list[dict]:
    """Map BERTopic's get_topic_info() DataFrame to Contract E shape.

    Outlier cluster -1 is excluded — downstream KG builder only processes
    real topics.
    """
    info = topic_model.get_topic_info()
    return [
        {
            "topic_id": int(row["Topic"]),
            "keywords": row["Representation"] if "Representation" in row else [],
            "doc_count": int(row["Count"]),
            "label": row.get("Name", f"Topic_{row['Topic']}"),
        }
        for _, row in info.iterrows()
        if row["Topic"] != -1
    ]


def _log_coherence(topic_info: list[dict], docs: list[str]) -> None:
    """Log C_V coherence score via gensim. Side-effect only — not in the contract."""
    try:
        tokenised = [d.lower().split() for d in docs]
        dictionary = Dictionary(tokenised)
        keywords = [t["keywords"] for t in topic_info if t["keywords"]]
        if not keywords:
            return
        cm = CoherenceModel(
            topics=keywords,
            texts=tokenised,
            dictionary=dictionary,
            coherence="c_v",
        )
        score = cm.get_coherence()
        logger.info("BERTopic C_V coherence: %.4f", score)
    except Exception:
        logger.warning("Coherence scoring failed.", exc_info=True)


def run_bertopic(
    docs: list[str], embeddings: np.ndarray
) -> tuple[list[dict], list[int], list[float]]:
    """Fit BERTopic. Returns ``(topic_info, doc_topics, doc_probs)`` — Contract E.

    IMPLEMENTATION.md §3.2 splits this into a model object plus a separate
    get_topic_info(); this wrapper must still return the Contract E tuple so
    graph.builder consumes NMF and BERTopic output identically. Outlier topic
    -1 stays in doc_topics but is excluded from topic_info.
    """
    logger.info("Fitting BERTopic on %d documents (precomputed embeddings).", len(docs))

    topic_model = _build_bertopic_model()
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    BERTOPIC_DIR.mkdir(parents=True, exist_ok=True)
    try:
        topic_model.save(
            str(BERTOPIC_DIR / "model"),
            serialization="pickle",
            save_ctfidf=True,
            save_embedding_model=False,
        )
        logger.info("BERTopic model saved to %s.", BERTOPIC_DIR / "model")
    except Exception:
        logger.warning("Could not save BERTopic model.", exc_info=True)

    topic_info = _get_topic_info(topic_model)
    logger.info("Discovered %d topics (excluding outliers).", len(topic_info))
    _log_coherence(topic_info, docs)

    # probs is 2D [n_docs × n_topics] when calculate_probabilities=True;
    # take the max probability across topics as the per-doc confidence score.
    if probs is not None and np.ndim(probs) == 2:
        doc_probs: list[float] = probs.max(axis=1).tolist()
    elif probs is not None:
        doc_probs = [float(p) for p in probs]
    else:
        doc_probs = [0.0] * len(docs)

    return topic_info, list(topics), doc_probs
