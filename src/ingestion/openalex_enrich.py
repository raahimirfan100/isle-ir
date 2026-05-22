"""OpenAlex metadata enrichment.

Owner:    Mujtaba Kamran (SSOT §2)
Contract: A — adds citation/affiliation fields to corpus.parquet (SSOT §6A)
Phase:    1 — Data Acquisition (IMPLEMENTATION.md §1.3)
STATUS:   core

Enriches papers with cited_by_count, institutions, countries and references
via the OpenAlex API. A free API key is required as of Feb 2026 — configure
it via OPENALEX_EMAIL / OPENALEX_API_KEY environment variables (config.py).
"""

from __future__ import annotations

import json
import time

import pandas as pd
import pyalex
from pyalex import Works
from tqdm import tqdm

from app.config import CORPUS_PATH, DATA_RAW, OA_API_KEY, OA_EMAIL


def _cache_dir() -> str:
    d = DATA_RAW / "openalex_cache"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _configure_pyalex() -> None:
    pyalex.config.email = OA_EMAIL
    pyalex.config.api_key = OA_API_KEY
    pyalex.config.max_retries = 3
    pyalex.config.retry_backoff_factor = 0.5
    pyalex.config.retry_http_codes = [429, 500, 503]


def enrich_batch(doi_list: list[str]) -> dict[str, dict]:
    """Return per-DOI OpenAlex metadata keyed by DOI.

    Each value carries cited_by_count, institutions, countries and references,
    which feed the corresponding Contract A columns (SSOT §6A). Responses are
    cached as JSON in data/raw/openalex_cache/ so re-runs skip already-fetched
    DOIs.
    """
    _configure_pyalex()
    cache_dir = _cache_dir()
    results: dict[str, dict] = {}

    for doi in tqdm(doi_list, desc="Enriching DOIs"):
        if not doi:
            continue

        cache_file = f"{cache_dir}/{doi.replace('/', '_')}.json"
        try:
            with open(cache_file) as f:
                results[doi] = json.load(f)
            continue
        except FileNotFoundError:
            pass

        try:
            work = Works()[f"doi:{doi}"]
        except Exception:
            # DOI not found or API error — cache empty result
            entry = {
                "cited_by_count": 0,
                "institutions": [],
                "countries": [],
                "references": [],
            }
            results[doi] = entry
            with open(cache_file, "w") as f:
                json.dump(entry, f)
            continue

        institutions: list[str] = []
        countries: list[str] = []
        for authorship in work.get("authorships", []):
            for inst in authorship.get("institutions", []):
                name = inst.get("display_name", "")
                code = inst.get("country_code", "")
                if name:
                    institutions.append(name)
                if code:
                    countries.append(code)

        entry = {
            "cited_by_count": work.get("cited_by_count", 0),
            "institutions": sorted(set(institutions)),
            "countries": sorted(set(countries)),
            "references": work.get("referenced_works", []),
        }
        results[doi] = entry
        with open(cache_file, "w") as f:
            json.dump(entry, f)
        time.sleep(0.05)  # polite pool: ~20 req/sec

    return results


def merge_enrichment(df: pd.DataFrame, enrichment: dict[str, dict]) -> pd.DataFrame:
    """Merge OpenAlex enrichment data into the corpus DataFrame.

    Updates cited_by_count, institutions, countries, references columns
    where the DOI has enrichment data. Saves the result to corpus.parquet.
    """
    for col in ("cited_by_count", "institutions", "countries", "references"):
        df[col] = [
            enrichment.get(doi, {}).get(col, current)
            for doi, current in zip(df["doi"], df[col], strict=True)
        ]

    df.to_parquet(CORPUS_PATH, index=False)
    print(f"Enriched {len(enrichment)} DOIs — saved to {CORPUS_PATH}")
    return df
