"""Unit tests for freshness-boosted re-ranking."""

from __future__ import annotations

from retrieval.freshness import apply_freshness


def test_recent_paper_ranks_first():
    """A 2024 paper should outrank a 2020 paper when RRF scores are equal."""
    results = [("paper_2020", 0.5), ("paper_2024", 0.5)]
    papers_lookup = {
        "paper_2020": {"year": 2020},
        "paper_2024": {"year": 2024},
    }
    topic_assignments = {"paper_2020": 0, "paper_2024": 0}
    decay_rates = {0: 0.5}

    boosted = apply_freshness(results, papers_lookup, topic_assignments, decay_rates, lam=0.3)

    assert len(boosted) == 2
    assert boosted[0][0] == "paper_2024"
    assert boosted[0][1] > boosted[1][1]


def test_empty_decay_rates_returns_original():
    """With no decay rates, results should be returned unchanged."""
    results = [("a", 0.8), ("b", 0.6)]
    boosted = apply_freshness(results, {}, {}, {}, lam=0.3)
    assert boosted == results


def test_empty_results_returns_empty():
    """Empty input should return empty output."""
    boosted = apply_freshness([], {}, {}, {0: 0.5}, lam=0.3)
    assert boosted == []
