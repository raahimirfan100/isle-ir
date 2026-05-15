"""OpenAlex metadata enrichment.

Owner:    Mujtaba Kamran (SSOT §2)
Contract: A — adds citation/affiliation fields to corpus.parquet (SSOT §6A)
Phase:    1 — Data Acquisition (IMPLEMENTATION.md §1.3)
STATUS:   core

Enriches papers with cited_by_count, institutions, countries and references
via the OpenAlex API. A free API key is required as of Feb 2026 — configure
it in src/app/config.py (OA_EMAIL, OA_API_KEY).
"""

from __future__ import annotations


def enrich_batch(doi_list: list[str]) -> dict[str, dict]:
    """Return per-DOI OpenAlex metadata keyed by DOI.

    Each value carries cited_by_count, institutions, countries and references,
    which feed the corresponding Contract A columns (SSOT §6A).
    """
    # TODO(mujtaba, phase-1): batch-query OpenAlex via pyalex; cache responses
    #   locally as JSON; respect the polite-pool rate limit.
    raise NotImplementedError
