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
import logging
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import pandas as pd
import pyalex
import requests
from pyalex import Works
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from app.config import CORPUS_PATH, DATA_RAW, OA_API_KEY, OA_EMAIL

logger = logging.getLogger(__name__)

_EMPTY_ENTRY = {
    "cited_by_count": 0,
    "institutions": [],
    "countries": [],
    "references": [],
}

_OA_FIELDS = ["id", "doi", "cited_by_count", "authorships", "referenced_works"]
_BATCH_SIZE = 50
_MAX_WORKERS = 10


def _cache_dir() -> str:
    d = DATA_RAW / "openalex_cache"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _configure_pyalex() -> None:
    if not OA_API_KEY:
        raise ValueError(
            "OPENALEX_API_KEY is not set. Register a free key at "
            "https://openalex.org/api-key and export it in your environment."
        )
    pyalex.config.email = OA_EMAIL
    pyalex.config.api_key = OA_API_KEY
    pyalex.config.max_retries = 3
    pyalex.config.retry_backoff_factor = 0.5
    pyalex.config.retry_http_codes = [429, 500, 503]


class _TimeoutAdapter(HTTPAdapter):
    """HTTPAdapter that injects a 30-second timeout into every request."""
    def send(self, request, **kwargs):
        kwargs.setdefault("timeout", 30)
        return super().send(request, **kwargs)


def _normalize_work_id(raw: str) -> str:
    """Extract bare OpenAlex work ID (e.g. 'W2964149567') from a URL or raw string."""
    if raw.startswith("https://openalex.org/"):
        return raw.rsplit("/", 1)[-1]
    return raw


def _cache_path(cache_dir: str, doi: str) -> str:
    """Filesystem-safe cache path derived from the DOI."""
    return f"{cache_dir}/{quote(doi, safe='')}.json"


def _extract_entry(work) -> dict:
    """Build an enrichment entry dict from an OpenAlex work record."""
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

    raw_refs = work.get("referenced_works", [])
    return {
        "cited_by_count": work.get("cited_by_count", 0),
        "institutions": sorted(set(institutions)),
        "countries": sorted(set(countries)),
        "references": [_normalize_work_id(r) for r in raw_refs],
    }


def _fetch_one(doi: str):
    """Fetch a single DOI, returning the work dict or None if not found."""
    try:
        return Works()[f"doi:{doi}"]
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if status == 404 or "not found" in str(exc).lower():
            return None
        raise


def _session_with_timeout() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=pyalex.config.max_retries,
        backoff_factor=pyalex.config.retry_backoff_factor,
        status_forcelist=pyalex.config.retry_http_codes,
    )
    s.mount("https://", _TimeoutAdapter(max_retries=retries))
    return s


def enrich_batch(doi_list: list[str]) -> dict[str, dict]:
    """Return per-DOI OpenAlex metadata keyed by DOI.

    Uses the ``filter_or`` batch endpoint (50 DOIs/request) with a
    ``ThreadPoolExecutor`` (10 workers) for concurrent fetching. Batches that
    fail (e.g. special-character DOIs) fall back to individual per-DOI requests.

    Each value carries cited_by_count, institutions, countries and references,
    which feed the corresponding Contract A columns (SSOT §6A). Responses are
    cached as JSON in data/raw/openalex_cache/ so re-runs skip already-fetched
    DOIs.

    Confirmed "not found" DOIs are cached as empty results. Transient errors
    (network, rate-limit) are logged and skipped — no cache file is written so
    the next run will retry them automatically.
    """
    _configure_pyalex()
    pyalex.api._get_requests_session = _session_with_timeout  # type: ignore[attr-defined]

    cache_dir = _cache_dir()
    results: dict[str, dict] = {}

    # Load cached results first
    uncached: list[str] = []
    for doi in tqdm(doi_list, desc="Checking cache"):
        if not doi:
            continue
        cache_file = _cache_path(cache_dir, doi)
        try:
            with open(cache_file) as f:
                results[doi] = json.load(f)
        except FileNotFoundError:
            uncached.append(doi)

    if not uncached:
        return results

    batches = [uncached[i:i + _BATCH_SIZE] for i in range(0, len(uncached), _BATCH_SIZE)]

    def _process_batch(batch: list[str]) -> None:
        try:
            works = Works().filter_or(doi=batch).select(_OA_FIELDS).get()
            found_dois: set[str] = set()
            for work in works:
                raw_doi = (work.get("doi") or "").replace("https://doi.org/", "")  # type: ignore[union-attr]
                if not raw_doi:
                    continue
                found_dois.add(raw_doi)
                entry = _extract_entry(work)
                results[raw_doi] = entry
                with open(_cache_path(cache_dir, raw_doi), "w") as f:
                    json.dump(entry, f)
            # DOIs not returned by the API are "not found"
            for doi in batch:
                if doi not in found_dois and doi not in results:
                    results[doi] = dict(_EMPTY_ENTRY)
                    with open(_cache_path(cache_dir, doi), "w") as f:
                        json.dump(_EMPTY_ENTRY, f)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Batch failed (%d DOIs), falling back to individual requests: %s",
                len(batch), exc,
            )
            for doi in batch:
                if doi in results:
                    continue
                try:
                    work = _fetch_one(doi)
                    if work is None:
                        results[doi] = dict(_EMPTY_ENTRY)
                        with open(_cache_path(cache_dir, doi), "w") as f:
                            json.dump(_EMPTY_ENTRY, f)
                    else:
                        entry = _extract_entry(work)
                        results[doi] = entry
                        with open(_cache_path(cache_dir, doi), "w") as f:
                            json.dump(entry, f)
                except Exception as inner_exc:  # noqa: BLE001
                    logger.warning("Individual DOI %s also failed: %s", doi, inner_exc)

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        list(tqdm(
            executor.map(_process_batch, batches),
            total=len(batches),
            desc="Enriching DOIs",
        ))

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
