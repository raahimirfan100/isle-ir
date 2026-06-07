"""Freshness-boosted re-ranking.

Owner:    Mujtaba Kamran (SSOT §2)
Phase:    Bonus (B-M1)
STATUS:   enhancement

Re-scores RRF results by a temporal freshness signal. Papers in fast-moving
topics (e.g. LLMs) decay fast; foundational topics decay slowly.
"""

from __future__ import annotations

import math

import pandas as pd


def estimate_decay_rates(
    corpus_df: pd.DataFrame,
    topic_assignments_full: list[int],
    current_year: int = 2025,
) -> dict[int, float]:
    """Estimate a citation-velocity decay rate per topic.

    For each topic, compute mean citations-per-year-old across its papers.
    Topics where recent papers get proportionally more citations → higher decay rate.
    Returns {topic_id: decay_rate} where decay_rate > 0.
    """
    if not topic_assignments_full or all(t == -1 for t in topic_assignments_full):
        # No real topic assignments — return uniform low decay
        return {}

    papers = corpus_df.to_dict("records")
    topic_citations: dict[int, list[float]] = {}

    for paper, tid in zip(papers, topic_assignments_full, strict=True):
        if tid == -1:
            continue
        year = paper.get("year", 0)
        age = max(1, current_year - year)
        cited = paper.get("cited_by_count", 0)
        rate = cited / age
        topic_citations.setdefault(tid, []).append(rate)

    decay_rates: dict[int, float] = {}
    for tid, rates in topic_citations.items():
        mean_rate = sum(rates) / len(rates)
        decay_rates[tid] = max(0.01, min(1.0, mean_rate / 10.0))  # normalise to 0.01–1.0

    return decay_rates


def apply_freshness(
    results: list[tuple[str, float]],
    papers_lookup: dict[str, dict],
    topic_assignments: dict[str, int],
    decay_rates: dict[int, float],
    lam: float = 0.3,
    current_year: int = 2025,
) -> list[tuple[str, float]]:
    """Re-score and re-sort results with a freshness boost.

    final_score = rrf_score × (1 + λ · e^(−decay_rate × years_old))

    Returns re-sorted [(arxiv_id, boosted_score), ...].
    """
    if not decay_rates or not results:
        return results

    boosted: list[tuple[str, float]] = []
    for arxiv_id, score in results:
        paper = papers_lookup.get(arxiv_id, {})
        year = paper.get("year", 0)
        years_old = max(0, current_year - year)
        tid = topic_assignments.get(arxiv_id, -1)
        decay = decay_rates.get(tid, 0.1)
        freshness = math.exp(-decay * years_old)
        new_score = score * (1.0 + lam * freshness)
        boosted.append((arxiv_id, new_score))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted
