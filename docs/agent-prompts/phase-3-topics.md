# Phase 3 — Topic Modeling

> Copy everything below the divider into your agent. Owners: Abubakar (NMF)
> and Raahim (BERTopic).

---

You are working on **ISLE-IR**. Read `AGENTS.md` first, then do **Phase 3:
Topic Modeling**.

## Read first
- `docs/IMPLEMENTATION.md` §3 — NMF and BERTopic (reference code + pitfalls).
- `docs/SSOT.md` §6 Contract E.

## Goal
Cluster a retrieved document set into coherent topics, with two
interchangeable implementations (NMF and BERTopic) behind one contract.

## Deliverables
- `src/topics/nmf_model.py` — `find_optimal_k`, `run_nmf` (Contract E).
- `src/topics/bertopic_model.py` — `run_bertopic` (Contract E).

## Contract to honor
Contract E (`docs/SSOT.md` §6E): `run_nmf` and `run_bertopic` **return the
same shape** — `(topic_info, doc_topics, doc_probs)`. `topic_info` items are
`{"topic_id", "keywords", "doc_count", "label"}`. The KG builder consumes
either output unchanged, so the shapes must be identical.

## Steps
1. **NMF.** TF-IDF + sklearn NMF. Sweep `k` over `config.NMF_K_RANGE`, pick the
   best by `c_v` coherence (gensim), log the scores.
2. **BERTopic.** Build the UMAP + HDBSCAN pipeline. **Reuse the precomputed
   `embeddings.npy`** from Phase 2 — do not re-encode the corpus. Set
   `random_state` for reproducibility; keep outlier topic `-1` out of
   `topic_info` but present in `doc_topics`.
3. Map BERTopic's native output to the exact Contract E tuple.
4. Compute and log `c_v` coherence for both; expect BERTopic ≥ NMF on
   scientific text.

## Exit criteria — verify every box before reporting done
- [ ] `run_nmf` and `run_bertopic` both return the Contract E tuple shape.
- [ ] NMF `c_v` coherence computed and logged for the selected `k`.
- [ ] BERTopic `c_v` coherence computed; compared against NMF.
- [ ] BERTopic reuses `embeddings.npy` (no second encoding pass).
- [ ] `ruff check .` clean; output consumable by `graph.builder` unchanged.

## Commit & branch
Branches `feat/abubakar-nmf-eval` and `feat/raahim-bertopic-reranker` off
`dev`. Small commits: `feat(topics): add NMF k-sweep`,
`feat(topics): add BERTopic pipeline`.
