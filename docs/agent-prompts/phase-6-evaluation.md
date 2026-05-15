# Phase 6 — Evaluation

> Copy everything below the divider into your agent. Owner: Abubakar.

---

You are working on **ISLE-IR**. Read `AGENTS.md` first, then do **Phase 6:
Evaluation**.

## Read first
- `docs/IMPLEMENTATION.md` §6 — retrieval and topic-coherence evaluation.
- `docs/SSOT.md` §11 (ablation study), §12 (test queries), §13 (corpus stats).

## Goal
Quantitative evidence the system works: a retrieval ablation table, topic
coherence scores, and corpus statistics for the report.

## Deliverables
- `evaluation/test_queries.json` — relevance labels filled in (the 20 queries
  are already scaffolded; add the relevant `arxiv_id`s).
- `evaluation/ablation.py` — runs the 5 system variants, writes
  `evaluation/results/ablation_table.csv`.
- `evaluation/corpus_stats.py` — Heaps' and Zipf's law plots.

## Steps
1. Label relevance for the top results of each of the 20 test queries (binary;
   `docs/SSOT.md` §11 suggests a majority vote across teammates).
2. Implement `ablation.py`: evaluate variants 1–5 (`docs/SSOT.md` §11) on
   P@5, P@10, MRR, MAP; write the CSV.
3. Implement `corpus_stats.py`: vocabulary-growth (Heaps) and rank-frequency
   (Zipf) plots over `corpus.parquet`.
4. Compute topic coherence (`c_v`, NPMI) for NMF vs BERTopic.

## Exit criteria — verify every box before reporting done
- [ ] All 20 queries in `test_queries.json` have relevance labels.
- [ ] `ablation_table.csv` exists with real P@5 / P@10 / MRR / MAP numbers for
      all 5 variants.
- [ ] Heaps' and Zipf's law plots generated.
- [ ] NMF and BERTopic coherence scores computed and recorded.
- [ ] `ruff check .` clean.

## Commit & branch
Branch `feat/evaluation` off `dev`. Small
commits: `feat(eval): add ablation harness`, `feat(eval): add corpus stats`,
`docs(eval): record final ablation numbers`.
