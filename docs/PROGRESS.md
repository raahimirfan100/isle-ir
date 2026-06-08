# Project Status

> Session-handoff tracker — read this and `AGENTS.md` at the start of any
> session. It records *state*; the *plan* lives in `docs/SSOT.md`. Update the
> phase table, the date, and the open items whenever state changes.

**Last updated:** 2026-06-08 · **Stage:** Phases 0–6 done. Bonus features largely complete — MMR, Rocchio, freshness, citation paths, export, corpus stats all implemented.

## Phase status

| Phase | Scope | Status |
|---|---|---|
| Bootstrap | repo scaffold, docs, CI, branch protection | done |
| 0 — Setup | env, GPU, embedding-model decision | done (Mujtaba) |
| 1 — Data | arXiv + OpenAlex → corpus.parquet | done (Mujtaba) — 100k papers, enriched; `data-v2` release |
| 2 — Retrieval | BM25 + FAISS + RRF + pipeline | done — FAISS ✅ BM25 ✅ RRF ✅ pipeline ✅ reranker ✅ query expansion ✅ |
| 3 — Topics | NMF + BERTopic | done — BERTopic ✅ (Raahim, 657 topics C_V 0.6623), NMF ✅ (Abubakar) |
| 4 — Knowledge graph | builder + analytics + collab network + visualization | done (Mujtaba) — PR #56 merged; builder ✅ analytics ✅ collab network ✅ PyVis ✅ PageRank ✅ |
| 5 — Frontend | Streamlit app | done — all tabs implemented (results, topics, graph, analytics, collab, semantic map) |
| 6 — Evaluation | ablation, coherence, corpus stats | done — ablation.py ✅ corpus_stats.py ✅ test_queries.json labeled ✅ |
| Bonus | MMR, Rocchio, freshness, citation paths, export, UMAP stub | done — MMR ✅ Rocchio ✅ freshness ✅ paths ✅ export ✅ UMAP stub ✅ |

Each phase has a copy-paste prompt in `docs/agent-prompts/`. Mark a phase done
only when its exit-criteria checklist is fully met.

## Starting a new session

1. Read `AGENTS.md` (how to work here) and this file (where things stand).
2. Take the current phase from the table above; open its
   `docs/agent-prompts/phase-N-*.md` prompt and follow it.
3. Spec and contracts: `docs/SSOT.md`. Code and pitfalls: `docs/IMPLEMENTATION.md`.

## Infrastructure state

- **Branches:** `main` and `dev` — no force-push or deletion; direct pushes allowed on both. Feature work: `feat/<feature>` off `dev`.
- **CI:** ruff lint + syntax + package-layout check on every PR and push.
- **Dependabot:** weekly pip and GitHub Actions update PRs.
- **Visibility:** private. Intended public after the project concludes — secret
  scanning and push protection auto-enable then.
- **Large artifacts:** GitHub Release assets, never the git tree (SSOT §15).

## Key decisions — index (detail lives where noted)

| Decision | Lives in |
|---|---|
| MIT license | `LICENSE` |
| Conventional Commits, no machine enforcement (no hooks) | `AGENTS.md` |
| Branch names by feature, not developer | `CONTRIBUTING.md` |
| Both `dev` and `main` accept direct pushes; no PR required | `CONTRIBUTING.md`, SSOT §4 |
| Embedding model: **SPECTER2** (`allenai/specter2_base`, 768-dim); MiniLM kept as comparison baseline | `src/app/config.py` |
| Environment: **uv** (`uv.lock`, `cpu`/`gpu` extras) — conda dropped; PyTorch `cu126` + `faiss-gpu-cu12` | `pyproject.toml`, `setup.sh` |
| No Claude git attribution; no `claude/*` branch names | team preference |
| Canonical docs: `docs/SSOT.md` + `docs/IMPLEMENTATION.md` (older drafts dropped) | — |
| Committed scope first; enhancements deferred and TODO-marked | `AGENTS.md`, SSOT §10 |

## Open items

- [x] Abubakar: implement `bm25_index.py` (Contract B) ✅
- [x] Abubakar: implement `rrf.py` (Contract D) ✅
- [x] Raahim: wire `pipeline.py` once BM25/RRF land — #38 ✅
- [x] Raahim: implement `bertopic_model.py` — #40 ✅ 657 topics, C_V 0.6623; model in `bertopic-index` release
- [x] Mujtaba: upload `corpus.parquet` to `data-v2` release ✅
- [x] Raahim: implement `reranker.py` — #52 ✅ CrossEncoder lazy-load, 0.4s/50 pairs on GPU
- [x] Mujtaba: implement knowledge graph Phase 4 — PR #56 ✅
- [x] Mujtaba: implement Phase 5 Streamlit frontend — PR #58 ✅
- [x] Abubakar: Phase 6 — relevance labels ✅ ablation.py ✅ corpus_stats.py ✅
- [x] Raahim: MMR (B-R1) ✅ — `src/retrieval/mmr.py`, wired in main.py
- [x] Abubakar: Rocchio (B-A1) ✅ — `src/retrieval/rocchio.py`, wired in main.py
- [ ] Raahim: SPECTER2 vs MiniLM comparison study → report Section 8
- [ ] Raahim: UMAP 2D scatter (B-R2) — stub exists, needs implementation
- [ ] Raahim: Topics-over-time (B-R3) — not yet started
- [ ] Abubakar: ILD metric in ablation (B-A2) — not yet started
- [ ] Abubakar: Citation flow Sankey (B-A3) — not yet started
