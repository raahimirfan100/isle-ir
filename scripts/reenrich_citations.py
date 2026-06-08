"""Re-enrich citation counts for papers that have cited_by_count == 0.

Uses the canonical arXiv DOI format (10.48550/arXiv.ID) which OpenAlex
indexes for all arXiv papers, regardless of whether the paper has a
publisher DOI. Also retries papers that have a publisher DOI but got 0.

Run from repo root:
    source .venv/bin/activate
    python scripts/reenrich_citations.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from app.config import CORPUS_PATH, OA_EMAIL  # noqa: E402

_BATCH = 50
_FIELDS = "id,doi,cited_by_count,authorships,referenced_works"
_EMAIL = OA_EMAIL or "work.raahim@gmail.com"


def _extract(work: dict) -> dict:
    institutions, countries = [], []
    for a in work.get("authorships", []):
        for inst in a.get("institutions", []):
            if inst.get("display_name"):
                institutions.append(inst["display_name"])
            if inst.get("country_code"):
                countries.append(inst["country_code"])
    raw_refs = work.get("referenced_works", [])
    refs = [r.rsplit("/", 1)[-1] for r in raw_refs]
    return {
        "cited_by_count": work.get("cited_by_count", 0) or 0,
        "institutions": sorted(set(institutions)),
        "countries": sorted(set(countries)),
        "references": refs,
    }


def _fetch_batch(dois: list[str]) -> dict[str, dict]:
    """Return {doi_suffix -> enrichment_dict} for a batch of DOIs."""
    doi_filter = "|".join(dois)
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={
                "filter": f"doi:{doi_filter}",
                "mailto": _EMAIL,
                "per_page": len(dois),
                "select": _FIELDS,
            },
            timeout=30,
        )
        r.raise_for_status()
    except Exception as exc:
        print(f"  batch error: {exc}", file=sys.stderr)
        return {}

    results: dict[str, dict] = {}
    for work in r.json().get("results", []):
        raw_doi = (work.get("doi") or "").replace("https://doi.org/", "").lower()
        if raw_doi:
            results[raw_doi] = _extract(work)
    return results


def main() -> None:
    df = pd.read_parquet(str(CORPUS_PATH))
    total = len(df)
    print(f"Corpus: {total} papers")
    print(f"Currently with citations: {(df['cited_by_count'] > 0).sum()}")

    # Candidates: papers with 0 cited_by_count
    zero_mask = df["cited_by_count"] == 0
    candidates = df[zero_mask].copy()
    print(f"Re-enriching {len(candidates)} papers with cited_by_count == 0 …")

    # Build lookup: lowercase_doi -> row_index in df
    # For papers with no DOI, fabricate the arxiv DOI
    doi_to_idx: dict[str, int] = {}
    for idx, row in candidates.iterrows():
        raw_doi = str(row.get("doi", "")).strip()
        if raw_doi:
            doi_to_idx[raw_doi.lower()] = idx
        else:
            arxiv_doi = f"10.48550/arxiv.{row['arxiv_id']}"
            doi_to_idx[arxiv_doi] = idx

    # Also build arxiv-doi -> idx for papers that have a publisher DOI but still 0
    # (we'll try the arxiv doi as a fallback after the main batch)
    arxiv_fallback: dict[str, int] = {}
    for idx, row in candidates.iterrows():
        raw_doi = str(row.get("doi", "")).strip().lower()
        arxiv_doi = f"10.48550/arxiv.{row['arxiv_id']}"
        if raw_doi and raw_doi != arxiv_doi:
            arxiv_fallback[arxiv_doi] = idx

    all_dois = list(doi_to_idx.keys())
    batches = [all_dois[i:i + _BATCH] for i in range(0, len(all_dois), _BATCH)]

    updated = 0
    for batch in tqdm(batches, desc="Fetching"):
        found = _fetch_batch(batch)
        for doi_key, enrichment in found.items():
            if enrichment["cited_by_count"] <= 0:
                continue
            # Match back to df index
            idx = doi_to_idx.get(doi_key)
            if idx is None:
                # Try stripping the arxiv prefix
                short = doi_key.replace("10.48550/arxiv.", "")
                idx = doi_to_idx.get(f"10.48550/arxiv.{short}")
            if idx is None:
                continue
            df.at[idx, "cited_by_count"] = enrichment["cited_by_count"]
            if not df.at[idx, "institutions"] or len(df.at[idx, "institutions"]) == 0:
                df.at[idx, "institutions"] = enrichment["institutions"]
            if not df.at[idx, "countries"] or len(df.at[idx, "countries"]) == 0:
                df.at[idx, "countries"] = enrichment["countries"]
            updated += 1
        time.sleep(0.05)  # polite rate-limiting

    # Second pass: arxiv-doi fallback for papers that had publisher DOI but 0
    if arxiv_fallback:
        print(f"\nFallback pass for {len(arxiv_fallback)} papers with publisher DOI …")
        fb_dois = list(arxiv_fallback.keys())
        fb_batches = [fb_dois[i:i + _BATCH] for i in range(0, len(fb_dois), _BATCH)]
        for batch in tqdm(fb_batches, desc="Fallback"):
            found = _fetch_batch(batch)
            for doi_key, enrichment in found.items():
                if enrichment["cited_by_count"] <= 0:
                    continue
                idx = arxiv_fallback.get(doi_key)
                if idx is None:
                    continue
                if df.at[idx, "cited_by_count"] == 0:
                    df.at[idx, "cited_by_count"] = enrichment["cited_by_count"]
                    updated += 1
            time.sleep(0.05)

    print(f"\nUpdated {updated} papers with new citation counts")
    print(f"Now with citations: {(df['cited_by_count'] > 0).sum()}")
    df.to_parquet(str(CORPUS_PATH), index=False)
    print(f"Saved to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
