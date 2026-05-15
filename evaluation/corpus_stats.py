"""Corpus statistics — Heaps' Law and Zipf's Law.

Owner:   Abubakar Ahmed (SSOT §2)
Phase:   6 — Evaluation (SSOT §13)
STATUS:  core

Produces the vocabulary-growth (Heaps) and term-frequency (Zipf) plots over
corpus.parquet for the report's Corpus Statistics section. Course tie-in:
Week 4 (Heaps' / Zipf's law).
"""

from __future__ import annotations


def compute_corpus_stats() -> None:
    """Compute and save the Heaps' and Zipf's law plots to evaluation/results/."""
    # TODO(abubakar, phase-6): token/vocab growth curve + rank-frequency plot.
    #   See docs/IMPLEMENTATION.md / SSOT §13 for the reference implementation.
    raise NotImplementedError


if __name__ == "__main__":
    compute_corpus_stats()
