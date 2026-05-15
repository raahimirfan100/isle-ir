# Phase 1 — Data Acquisition

> Copy everything below the divider into your agent. Owner: Mujtaba.

---

You are working on **ISLE-IR**. Read `AGENTS.md` first, then do **Phase 1:
Data Acquisition**.

## Read first
- `docs/IMPLEMENTATION.md` §1 — arXiv + OpenAlex ingestion (reference code).
- `docs/SSOT.md` §6 Contract A — the exact `corpus.parquet` column schema.
- `docs/SSOT.md` §7 Week 1 — the timeline and dependency this unblocks.

## Goal
Produce `data/processed/corpus.parquet` — the corpus every other module
depends on. Nothing downstream can start without it.

## Deliverables
- `src/ingestion/arxiv_loader.py` — load the arXiv Kaggle snapshot, filter to
  `cs.*` categories, build the unified `text` field.
- `src/ingestion/openalex_enrich.py` — enrich papers with `cited_by_count`,
  `institutions`, `countries`, `references` via OpenAlex.
- `data/processed/corpus.parquet` — committed to the shared Drive, **not** git.

## Contract to honor
Contract A (`docs/SSOT.md` §6A): every row of `corpus.parquet` must have
exactly the specified columns and types. Downstream code trusts this schema.

## Steps
1. Implement `load_arxiv_snapshot()` per IMPLEMENTATION.md §1.1; cache responses.
2. Implement `enrich_batch()` per §1.3; respect the OpenAlex polite-pool rate
   limit and cache every response locally as JSON.
3. Produce a **5k-paper sample first** so retrieval work can start (Day 3),
   then the full corpus (≥ 50k papers).
4. Validate the output against the Contract A schema before sharing it.

## Exit criteria — verify every box before reporting done
- [ ] `corpus.parquet` exists and every row matches the Contract A schema.
- [ ] ≥ 50k papers; `text` field is `title + " " + abstract`, newline-free.
- [ ] OpenAlex fields populated where available (0 / empty where not).
- [ ] A 5k sample was shared early so retrieval is unblocked.
- [ ] `ruff check .` clean; ingestion modules import without error.

## Commit & branch
Branch `feat/ingestion` off `dev`. Small commits:
`feat(ingestion): load arXiv snapshot`, `feat(ingestion): add OpenAlex enrichment`.
