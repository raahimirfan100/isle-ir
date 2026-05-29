# Bonus Features — Abubakar

> Copy everything below the divider into your agent. Owner: Abubakar.
> Prerequisite: Phase 6 (evaluation / ablation) must be done before B-A2 (ILD
> metric) can be added to the ablation table. B-A1 (Rocchio) and B-A3
> (citation flow) can be started in parallel with Phase 6.

---

You are working on **ISLE-IR**. Read `AGENTS.md` first, then implement
**Abubakar's bonus features** as described in `docs/BONUS_FEATURES.md`.

## Read first
- `docs/BONUS_FEATURES.md` — full spec for B-A1, B-A2, B-A3.
- `src/retrieval/faiss_index.py` — `search_by_embedding` is used by Rocchio's second-pass retrieval.
- `evaluation/ablation.py` — extend this for B-A2.
- `src/graph/analytics.py` — betweenness centrality pattern used in B-A3.
- `docs/SSOT.md` §11 — ablation study design.
- `src/app/config.py` — all paths and constants.

## Goal
Three CPU-friendly features that improve interactivity and evaluation rigor:
interactive Rocchio relevance feedback, a diversity metric in the ablation
table, and a citation flow visualization.

---

## B-A1 — Interactive Rocchio Relevance Feedback

**New file:** `src/retrieval/rocchio.py`

Implement the Rocchio update rule so users can refine search results by marking
papers relevant or not-relevant. The UI is wired by Mujtaba; your job is the
algorithm and the interface it exposes.

### Interface
```python
def rocchio(
    q_emb: np.ndarray,
    rel_embs: list[np.ndarray],
    nrel_embs: list[np.ndarray],
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> np.ndarray:
    """
    Rocchio relevance feedback update (Manning Ch. 9).

    q_new = α · q_orig + β · mean(rel_embs) − γ · mean(nrel_embs)

    If rel_embs or nrel_embs is empty, skip that term.
    Returns a unit-normalised embedding of the same dimension as q_emb.
    """
```

### Steps
1. Implement the formula above. Guard against empty lists (skip the term if
   `rel_embs` or `nrel_embs` is empty rather than erroring).
2. L2-normalise the output so it can be passed directly to
   `FAISSIndex.search_by_embedding()` (which expects unit vectors).
3. Add constants to `src/app/config.py`:
   `ROCCHIO_ALPHA = 1.0`, `ROCCHIO_BETA = 0.75`, `ROCCHIO_GAMMA = 0.15`.
4. Leave a `# TODO(mujtaba, phase-5): thumbs-up/down buttons + Refine Search`
   comment in `results_tab.py` at the point where per-card buttons would go.
   Mujtaba owns the Streamlit session-state plumbing that accumulates feedback
   and calls `FAISSIndex.search_by_embedding(rocchio(...))`.
5. Write unit tests in `tests/test_rocchio.py`:
   - With one relevant embedding that matches the query: output should move
     closer to it.
   - With one non-relevant embedding: output should move away.
   - With empty lists: should return the original (normalised) query.

### Exit criteria
- [ ] `rocchio()` output is unit-normalised.
- [ ] Three unit tests pass (closer, farther, empty-list fallback).
- [ ] `ruff check .` clean.

---

## B-A2 — ILD Diversity Metric in Ablation Table

**Extend:** `evaluation/ablation.py`

Add Intra-List Diversity (ILD) as a 6th metric column. ILD quantifies how
spread out the result set is in embedding space — a higher ILD with similar
MRR confirms that MMR (B-R1) improves diversity without hurting relevance.

### Formula
```
ILD@k = (2 / k(k−1)) · Σ_{i<j} (1 − cosine_sim(emb_i, emb_j))
```

### Steps
1. After computing the result set for each system variant and query, load the
   embeddings for the retrieved papers from `embeddings.npy` using the
   paper-ID-to-index map (already built during FAISS index load).
2. Implement `ild_at_k(result_ids: list[str], pid_to_idx: dict, embeddings: np.ndarray, k: int) -> float`.
3. Add `ILD@10` as a 6th column to `ablation_table.csv` alongside P@5/P@10/MRR/MAP.
4. Run with `mmr=True` for the "Full system" variant and `mmr=False` for all
   others — confirm ILD is higher in the MMR row.

### Exit criteria
- [ ] `ablation_table.csv` has 6 columns (P@5, P@10, MRR, MAP, ILD@10, ILD@10-mmr).
- [ ] ILD is higher for the MMR variant than the non-MMR baseline.
- [ ] `ruff check .` clean.

---

## B-A3 — Citation Flow & Temporal Influence Analysis

**New files:**
- `evaluation/citation_flow.py` — builds the temporal citation flow data.
- Add a "Citation Flow" chart to `src/app/components/analytics_tab.py` (Mujtaba wires the call).

Use citation edges and year metadata already in the KG to show how influence
propagates through the result set over time. Produces a Sankey-style flow
diagram: year slices as nodes, citation volume between year groups as flows.

### Interface
```python
def build_citation_flow(
    G: nx.DiGraph,
    result_ids: list[str],
) -> dict:
    """
    Build year-binned citation flow data for a Plotly Sankey diagram.

    Returns:
        {
          "labels": list[str],   # ["2019", "2020", "2021", ...]
          "source": list[int],   # index into labels (citing year)
          "target": list[int],   # index into labels (cited year)
          "value":  list[int],   # citation count between those years
        }
    """

def find_bridge_papers(
    G: nx.DiGraph,
    result_ids: list[str],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """
    Return the top_n papers by betweenness centrality in the citation
    subgraph of result_ids. These are the 'bridge' papers connecting
    disparate sub-communities.
    Returns [(arxiv_id, betweenness_score), ...].
    """
```

### Steps
1. Build the **citation subgraph**: extract only `result_ids` nodes and `cites`
   edges between them from the full KG `G`.
2. Group papers by year; count directed citations between year bins (year Y →
   year Y+1, Y+2, etc.) to produce the Sankey source/target/value arrays.
3. Compute betweenness centrality (`nx.betweenness_centrality`) on the citation
   subgraph; return top-N papers.
4. In `analytics_tab.py`, render:
   - The Sankey diagram via `plotly.graph_objects.Figure(go.Sankey(...))`.
   - A "Bridge Papers" table showing the top-5 betweenness-centrality papers
     with their title, year, and score.
5. Guard: if the result set has fewer than 5 papers with citation edges, show
   `st.info("Not enough intra-set citation links for flow analysis.")`.

### Exit criteria
- [ ] `build_citation_flow` returns a dict with the four keys above.
- [ ] `find_bridge_papers` returns a non-empty list for a query whose results
      include papers that cite each other.
- [ ] Sankey diagram renders in the analytics tab without error.
- [ ] Bridge papers table appears below the Sankey.
- [ ] Graceful fallback when there are too few citation links.
- [ ] `ruff check .` clean.

---

## Commit & branch
Branch `feat/bonus-evaluation` off `dev`.

Small commits:
- `feat(retrieval): implement Rocchio relevance feedback`
- `test(retrieval): add Rocchio unit tests`
- `feat(eval): add ILD diversity metric to ablation table`
- `feat(eval): add citation flow and bridge paper analysis`
- `feat(app): wire citation flow Sankey to analytics tab`
