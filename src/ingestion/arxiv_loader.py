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

import json

import numpy as np
import pandas as pd

from app.config import CORPUS_PATH, DATA_PROCESSED

# Contract A columns (SSOT §6A) — used for construction and validation.
CONTRACT_A_COLUMNS: list[str] = [
    "arxiv_id",
    "title",
    "abstract",
    "text",
    "authors",
    "year",
    "categories",
    "doi",
    "cited_by_count",
    "institutions",
    "countries",
    "references",
]


def _parse_authors(authors_parsed: list[list[str]]) -> list[str]:
    """Convert Kaggle authors_parsed ([last, first, middle]) to full names."""
    names: list[str] = []
    for parts in authors_parsed:
        last = parts[0].strip() if len(parts) > 0 else ""
        first = parts[1].strip() if len(parts) > 1 else ""
        middle = parts[2].strip() if len(parts) > 2 else ""
        full = " ".join(filter(None, [first, middle, last]))
        if full:
            names.append(full)
    return names


def load_arxiv_snapshot(
    path: str,
    categories: list[str],
    max_records: int = 100_000,
) -> pd.DataFrame:
    """Load the arXiv Kaggle snapshot, filtered to ``categories``.

    Writes data/processed/corpus.parquet and returns the DataFrame. Each row
    matches the Contract A schema (SSOT §6A), including the pre-joined
    ``text`` field. OpenAlex columns are set to defaults (0 / empty list)
    and filled in later by ``openalex_enrich``.
    """
    target_cats = set(categories)
    records: list[dict] = []

    with open(path) as f:
        for line in f:
            if len(records) >= max_records:
                break
            paper = json.loads(line)
            paper_cats = set(paper.get("categories", "").split())
            if not (paper_cats & target_cats):
                continue

            title = paper["title"].replace("\n", " ").strip()
            abstract = paper["abstract"].replace("\n", " ").strip()

            records.append(
                {
                    "arxiv_id": paper["id"],
                    "title": title,
                    "abstract": abstract,
                    "text": f"{title} {abstract}",
                    "authors": _parse_authors(paper.get("authors_parsed", [])),
                    "year": int(paper.get("update_date", "2020")[:4]),
                    "categories": paper.get("categories", "").split(),
                    "doi": paper.get("doi") or "",
                    # OpenAlex defaults — filled by enrich_batch
                    "cited_by_count": 0,
                    "institutions": [],
                    "countries": [],
                    "references": [],
                }
            )

    df = pd.DataFrame(records, columns=CONTRACT_A_COLUMNS)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CORPUS_PATH, index=False)
    print(f"Saved {len(df)} papers to {CORPUS_PATH}")
    return df


def validate_contract_a(df: pd.DataFrame) -> None:
    """Validate that df matches the Contract A schema (SSOT §6A).

    Raises AssertionError if any column is missing, has the wrong type,
    or if the text field doesn't match title + abstract.
    """
    missing = set(CONTRACT_A_COLUMNS) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    for col in ("arxiv_id", "title", "abstract", "text", "doi"):
        assert pd.api.types.is_string_dtype(
            df[col]
        ), f"'{col}' expected string, got {df[col].dtype}"

    assert pd.api.types.is_integer_dtype(df["year"]), "'year' expected int"
    assert pd.api.types.is_integer_dtype(df["cited_by_count"]), "'cited_by_count' expected int"

    for col in ("authors", "categories", "institutions", "countries", "references"):
        non_null = df[col].dropna()
        assert (non_null.apply(lambda v: isinstance(v, (list, np.ndarray)))).all(), (
            f"'{col}' has non-list values"
        )
        # Check element types on a sample (full scan is expensive).
        for val in non_null.head(200):
            assert all(isinstance(el, str) for el in val), (
                f"'{col}' contains non-string elements"
            )

    mismatches = (df["text"] != df["title"] + " " + df["abstract"]).sum()
    assert mismatches == 0, f"text field mismatch in {mismatches}/{len(df)} rows"

    for col in ("title", "abstract", "text"):
        assert not df[col].str.contains("\n", na=False).any(), f"'{col}' contains newlines"

    print(f"Contract A validation passed — {len(df)} rows, {len(df.columns)} columns")
