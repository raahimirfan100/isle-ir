"""arXiv metadata loader.

Owner:    Mujtaba Kamran (SSOT §2)
Contract: A — produces data/processed/corpus.parquet (SSOT §6A)
Phase:    1 — Data Acquisition (IMPLEMENTATION.md §1.1)
STATUS:   core

Loads the arXiv Kaggle snapshot, filters to cs.* categories, and emits the
unified corpus DataFrame. Every row must satisfy the Contract A column schema
defined in SSOT §6A.
"""

from __future__ import annotations

import pandas as pd


def load_arxiv_snapshot(
    path: str,
    categories: list[str],
    max_records: int = 100_000,
) -> pd.DataFrame:
    """Load the arXiv Kaggle snapshot, filtered to ``categories``.

    Writes data/processed/corpus.parquet and returns the DataFrame. Each row
    must match the Contract A schema (SSOT §6A), including the pre-joined
    ``text`` field.
    """
    # TODO(mujtaba, phase-1): parse the JSONL snapshot, filter cs.* categories,
    #   build the `text` field, persist parquet, return the Contract A frame.
    raise NotImplementedError
