# Phase 2 — Hybrid Retrieval Engine

> Copy everything below the divider into your agent. Owners: Abubakar
> (BM25, RRF) and Raahim (FAISS, pipeline). This is the core of the project.

---

You are working on **ISLE-IR**. Read `AGENTS.md` first, then do **Phase 2:
Hybrid Retrieval Engine**.

## Read first
- `docs/IMPLEMENTATION.md` §2 — BM25, FAISS, RRF (reference code + pitfalls).
- `docs/SSOT.md` §6 Contracts B, C, D.

## Goal
A working hybrid retrieval pipeline: BM25 + dense retrieval, fused with
Reciprocal Rank Fusion, returning a ranked result list for any query string.

## Deliverables
- `src/retrieval/bm25_index.py` — `BM25Index` (Contract B).
- `src/retrieval/faiss_index.py` — `FAISSIndex` (Contract C).
- `src/retrieval/rrf.py` — `reciprocal_rank_fusion` (Contract D).
- `src/retrieval/pipeline.py` — `full_retrieval_pipeline` (Contract D).

## Contracts to honor
Contracts B, C, D (`docs/SSOT.md` §6). The public signatures are fixed — match
them exactly so the UI and evaluation code can call them unchanged.

## Steps (build order matters)
1. **BM25 first.** Build the index, run a few queries, eyeball the results.
2. **FAISS next.** Encode with the model from `config.EMBEDDING_MODEL`; build
   `IndexIVFFlat` (train before adding!); persist `embeddings.npy` — Phase 3
   reuses it, do not encode twice.
3. **RRF.** Implement it; unit-test it with dummy ranked lists in `tests/`.
4. **pipeline.py.** Wire the core path: BM25 + FAISS → RRF. Keep
   `expand_query` / `rerank` as flags; they call enhancement modules — leave a
   `# TODO(..., enhancement)` and default callers to `False` until Phase
   enhancements land.

## Exit criteria — verify every box before reporting done
- [ ] `BM25Index.search()` and `FAISSIndex.search()` each return ranked
      results on the 5k sample.
- [ ] BM25 and FAISS agree on at least a few top results (sanity check).
- [ ] `reciprocal_rank_fusion` has a passing unit test in `tests/`.
- [ ] `full_retrieval_pipeline()` returns a ranked list for any query string.
- [ ] On the full corpus, one query runs in under ~15 seconds.
- [ ] `ruff check .` clean; all signatures match Contracts B/C/D exactly.

## Commit & branch
Branches `feat/bm25-rrf` and `feat/faiss-index` off `dev`.
Small commits: `feat(retrieval): add BM25 index`, `test(retrieval): RRF unit
tests`, `feat(retrieval): wire hybrid pipeline`.
