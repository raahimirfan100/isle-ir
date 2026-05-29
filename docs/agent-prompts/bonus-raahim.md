# Bonus Features — Raahim

> Copy everything below the divider into your agent. Owner: Raahim.
> Prerequisite: Phase 5 (Streamlit) must be wired before the UI tasks here
> can be exercised. B-R1 (MMR) and the SPECTER2 comparison can be started
> independently.

---

You are working on **ISLE-IR**. Read `AGENTS.md` first, then implement
**Raahim's bonus features** as described in `docs/BONUS_FEATURES.md`.

## Read first
- `docs/BONUS_FEATURES.md` — full spec for B-R1, B-R2, B-R3.
- `src/retrieval/pipeline.py` — see how `rerank` flag is wired; MMR follows the same pattern.
- `src/retrieval/faiss_index.py` — `search_by_embedding` and the embeddings layout.
- `src/topics/bertopic_model.py` — the fitted `BERTopic` object exposes `umap_model` and `topics_over_time`.
- `src/app/config.py` — all paths and constants.

## Goal
Three GPU-accelerated features that improve retrieval quality and visualization:
MMR diversity re-ranking, UMAP 2D result scatter, and topics-over-time chart.
Plus finish the SPECTER2 vs MiniLM comparison study for the report.

---

## B-R1 — MMR Diversity Re-ranking

**New file:** `src/retrieval/mmr.py`

After cross-encoder reranking, apply Maximal Marginal Relevance to select a
diverse top-k from the candidate pool. Uses precomputed embeddings from
`embeddings.npy` — no new model inference.

### Interface
```python
def mmr(
    query_emb: np.ndarray,
    candidates: list[tuple[str, float, np.ndarray]],  # (arxiv_id, score, embedding)
    lam: float = 0.5,
    k: int = 20,
) -> list[tuple[str, float]]:
    """
    Greedy MMR selection.

    MMR = argmax[ λ · sim(d, q) − (1−λ) · max_{s∈S} sim(d, s) ]

    lam=1.0 → pure relevance (same as top-k); lam=0.0 → pure diversity.
    Returns [(arxiv_id, mmr_score), ...] length k.
    """
```

### Steps
1. Implement the greedy loop: maintain a `selected` set; at each step pick the
   candidate that maximises the MMR score; `sim` is cosine (dot product on
   L2-normalised vectors, which FAISS already guarantees).
2. Wire into `pipeline.py`: add `mmr: bool = False, mmr_lambda: float = 0.5`
   to `full_retrieval_pipeline`. When `mmr=True`, fetch embeddings for the
   reranked candidates via index into `embeddings.npy`, then call `mmr()`.
3. Add `MMR_LAMBDA = 0.5` and `MMR_TOP_K = 20` to `src/app/config.py`.
4. Mujtaba will add a λ slider to the sidebar — leave a
   `# TODO(mujtaba, phase-5): wire mmr_lambda from sidebar` in `pipeline.py`.
5. Write a unit test in `tests/test_mmr.py` with a tiny synthetic case:
   create 5 embeddings where two are near-duplicates; confirm MMR drops one.

### Exit criteria
- [ ] `mmr()` returns exactly `k` results.
- [ ] With `lam=0.5`, the result set is measurably more diverse than top-k by
      cosine distance (compute mean pairwise similarity before/after).
- [ ] `full_retrieval_pipeline(query, ..., mmr=True)` runs without error.
- [ ] Unit test passes.
- [ ] `ruff check .` clean.

---

## B-R2 — UMAP 2D Embedding Visualization

**New file:** `src/app/components/embedding_viz_tab.py`

Project the top-50 retrieved papers into 2D using the already-fitted BERTopic
UMAP model. Render as a Plotly scatter coloured by topic cluster.

### Interface
```python
def render(
    result_ids: list[str],
    topic_assignments: list[int],
    all_embeddings: np.ndarray,
    pid_to_idx: dict[str, int],
    topic_info: list[dict],
    papers_lookup: dict[str, dict],
) -> None:
    """
    Render a 2D UMAP scatter of retrieved results coloured by topic.
    Call umap_model.transform() on the result subset — do NOT refit.
    """
```

### Steps
1. Retrieve `umap_model` from the fitted BERTopic object (stored as
   `topic_model.umap_model` after `fit_transform`). Use `transform()`, not
   `fit_transform()`, to project the result subset — this is instant on 50 points.
2. Build a `pd.DataFrame` with columns `x`, `y`, `topic`, `title`, `year`,
   `cited_by_count`; use topic keyword strings from `topic_info` as colour labels.
3. Plot with `plotly.express.scatter`; set `hover_data=["title", "year"]`.
4. Handle the case where the BERTopic model isn't fitted yet (e.g. user runs
   with NMF): fall back to a `st.info("Run with BERTopic to see this view.")`.
5. Mujtaba adds `"🗺 Semantic Map"` as a 5th tab in `main.py`; leave a
   `# TODO(mujtaba, phase-5): add Semantic Map tab` comment for now.

### Exit criteria
- [ ] Scatter renders with correct topic colours and hover text.
- [ ] `umap_model.transform()` used (not `fit_transform`).
- [ ] Graceful fallback when model is unavailable.
- [ ] `ruff check .` clean.

---

## B-R3 — Topics-Over-Time Temporal Visualization

**Extend:** `src/app/components/topics_tab.py`

Use BERTopic's built-in `topics_over_time()` to produce a Plotly line chart
showing how topic proportions shift across publication years in the result set.
This mirrors ISLE Figure 6 and directly maps to Week 12 (text clustering /
temporal analysis).

### Steps
1. After `run_bertopic()` returns the fitted `topic_model`, call:
   ```python
   timestamps = [str(papers_lookup[pid]["year"]) for pid in result_ids]
   tot = topic_model.topics_over_time(docs, timestamps, nr_bins=6, global_tuning=True)
   fig = topic_model.visualize_topics_over_time(tot, top_n_topics=8)
   st.plotly_chart(fig, use_container_width=True)
   ```
2. Guard: only render if `len(set(timestamps)) >= 3` (need at least 3 distinct
   years for a meaningful chart); otherwise show `st.info("Not enough year
   diversity in this result set for temporal analysis.")`.
3. Add a subheader `"Topic Trends Over Time"` above the chart in `topics_tab.py`.
4. This is CPU-bound and fast (<1 s on 50 docs) — no spinner needed.

### Exit criteria
- [ ] Chart renders for a query that returns papers spanning ≥ 3 years.
- [ ] Graceful fallback for single-year result sets.
- [ ] Only called when BERTopic (not NMF) is active.
- [ ] `ruff check .` clean.

---

## SPECTER2 vs MiniLM Comparison (SSOT §10 remaining item)

**No new file** — this is a notebook + report section task.

### Steps
1. In `notebooks/02_retrieval_sanity.ipynb`, build a **10k-paper** FAISS
   `IndexFlatIP` for each model:
   - `allenai/specter2_base` (768-dim, already used for the main index)
   - `sentence-transformers/all-MiniLM-L6-v2` (384-dim, the baseline)
2. Run the same 10 queries from `evaluation/test_queries.json` on both indexes.
3. Compute per-query: mean intra-result cosine similarity of top-10 (higher =
   more semantically focused results), MRR against your relevance labels.
4. Record build time and query latency for both.
5. Write up findings in report Section 8 — expected conclusion: SPECTER2 wins
   on scientific query focus; MiniLM is 2× faster.

### Exit criteria
- [ ] Notebook cell produces a comparison table (model | intra-sim | MRR | build time | query time).
- [ ] Conclusion written in report Section 8 with quantitative support.

---

## Commit & branch
Branch `feat/bonus-retrieval` off `dev`.

Small commits:
- `feat(retrieval): implement MMR diversity re-ranking`
- `test(retrieval): add MMR unit test`
- `feat(app): add UMAP 2D semantic map tab`
- `feat(app): add topics-over-time chart to topics tab`
- `docs(eval): SPECTER2 vs MiniLM comparison notebook`
