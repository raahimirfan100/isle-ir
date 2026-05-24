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
    BERTOPIC_HDBSCAN_METRIC,
    BERTOPIC_HDBSCAN_MIN_SAMPLES,
    BERTOPIC_HDBSCAN_SELECTION,
    BERTOPIC_MAX_DF,
    BERTOPIC_MIN_CLUSTER,
    BERTOPIC_MIN_DF,
    BERTOPIC_N_COMPONENTS,
    BERTOPIC_N_NEIGHBORS,
    BERTOPIC_NR_TOPICS,
    BERTOPIC_RANDOM_STATE,
    BERTOPIC_TOP_N_WORDS,
    BERTOPIC_UMAP_METRIC,
    BERTOPIC_UMAP_MIN_DIST,
)

logger = logging.getLogger(__name__)


def _build_bertopic_model() -> BERTopic:
    """Construct a reproducible BERTopic model from config hyperparameters."""
    umap_model = UMAP(
        n_neighbors=BERTOPIC_N_NEIGHBORS,
        n_components=BERTOPIC_N_COMPONENTS,
        min_dist=BERTOPIC_UMAP_MIN_DIST,
        metric=BERTOPIC_UMAP_METRIC,
        random_state=BERTOPIC_RANDOM_STATE,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=BERTOPIC_MIN_CLUSTER,
        min_samples=BERTOPIC_HDBSCAN_MIN_SAMPLES,
        metric=BERTOPIC_HDBSCAN_METRIC,
        cluster_selection_method=BERTOPIC_HDBSCAN_SELECTION,
        prediction_data=True,
    )

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=BERTOPIC_MIN_DF,
        max_df=BERTOPIC_MAX_DF,
    )

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=BERTOPIC_TOP_N_WORDS,
        nr_topics=BERTOPIC_NR_TOPICS,
        calculate_probabilities=True,
        verbose=True,
    )


def _get_topic_info(topic_model: BERTopic) -> list[dict]:
    """Map BERTopic's get_topic_info() DataFrame to Contract E shape.

    Outlier cluster -1 is excluded — downstream KG builder only processes
    real topics. Keywords are fetched via get_topic() to guarantee list[str]
    regardless of BERTopic version; label falls back to a string when Name is
    missing or NaN.
    """
    info = topic_model.get_topic_info()
    result = []
    for _, row in info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue
        # get_topic() returns [(word, score), ...] — always a list
        word_score_pairs = topic_model.get_topic(topic_id) or []
        keywords = [word for word, _score in word_score_pairs]
        raw_label = row.get("Name", None)
        label = str(raw_label) if raw_label and str(raw_label) != "nan" else f"Topic_{topic_id}"
        result.append(
            {
                "topic_id": topic_id,
                "keywords": keywords,
                "doc_count": int(row["Count"]),
                "label": label,
            }
        )
    return result


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

    # Select the probability for each document's *assigned* topic.
    # probs is 2D [n_docs × n_topics] when calculate_probabilities=True;
    # column j corresponds to topic ID j. Outlier docs (topic == -1) get 0.0.
    if probs is not None and np.ndim(probs) == 2:
        doc_probs: list[float] = [
            float(probs[i, topics[i]]) if topics[i] != -1 else 0.0
            for i in range(len(topics))
        ]
    elif probs is not None:
        doc_probs = [float(p) for p in probs]
    else:
        doc_probs = [0.0] * len(docs)

    return topic_info, list(topics), doc_probs
