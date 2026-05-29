# ISLE-IR — Bonus Feature Plan

> **Owner:** All three teammates  
> **Added:** 2026-05-29  
> **Status:** Planned — implement after Phase 5 and Phase 6 are complete

This document defines the "above and beyond" features that extend the committed scope. All features are new — not listed in `SSOT.md §10`. Each is assigned to one owner based on hardware fit and existing module ownership.

For context on what is already done from `SSOT.md §10`:

| Feature | Owner | Status |
|---|---|---|
| Cross-encoder reranking | Raahim | ✅ Done |
| Pseudo-relevance feedback query expansion | Abubakar | ✅ Done |
| "More Like This" via `FAISSIndex.search_by_embedding()` | Raahim | ✅ Done |
| Author collaboration network | Mujtaba | ✅ Done (Phase 4) |
| PageRank on citation graph | Mujtaba | ✅ Done (Phase 4) |
| SPECTER2 vs MiniLM comparison study | Raahim | ⏳ Report Section 8 only |

---

## Remaining Committed Scope (finish before bonus work)

| Task | Owner | Phase |
|---|---|---|
| Wire `main.py`: sidebar, `@st.cache_resource`, `@st.cache_data`, tab routing | Mujtaba | 5 |
| `results_tab.py render()` — result cards with title, score, abstract, "More Like This" button | Abubakar | 5 |
| `topics_tab.py render()` — topic pie chart + BERTopic word cloud barchart | Abubakar | 5 |
| `graph_tab.py render()` — PyVis HTML iframe embed | Mujtaba | 5 |
| `analytics_tab.py render()` — year histogram, top-cited table, PageRank vs citation side-by-side | Mujtaba | 5 |
| Fill relevance labels in `evaluation/test_queries.json` (20 queries, majority vote) | All | 6 |
| `evaluation/ablation.py` — 5 system variants × 20 queries, P@5/P@10/MRR/MAP, write CSV | Abubakar | 6 |
| `evaluation/corpus_stats.py` — Heaps' Law + Zipf's Law plots | Abubakar | 6 |

---

## Raahim's Bonus Tasks (GPU — RTX 3060)

### B-R1: MMR Diversity Re-ranking
**New file:** `src/retrieval/mmr.py`

After cross-encoder reranking, apply Maximal Marginal Relevance to balance relevance vs. diversity:

```
MMR = argmax[ λ · sim(d, q) − (1−λ) · max_{s∈S} sim(d, s) ]
```

Uses precomputed FAISS embeddings — no new model inference.

```python
def mmr(
    query_emb: np.ndarray,
    candidates: list[tuple[str, float, np.ndarray]],  # (arxiv_id, score, embedding)
    lam: float = 0.5,
    k: int = 20,
) -> list[tuple[str, float]]:
    """Greedy MMR selection. Returns [(arxiv_id, mmr_score), ...]."""
```

- Wire into `pipeline.py` as `mmr: bool = False` flag (same pattern as `rerank`)
- Mujtaba adds a λ slider in the Streamlit sidebar
- Abubakar adds ILD metric to the ablation table (B-A2) to quantify the improvement
- **Course tie-in:** Diversity-relevance tradeoff (Manning Ch. 9)

**Effort:** ~1 day

---

### B-R2: UMAP 2D Embedding Visualization
**New file:** `src/app/components/embedding_viz_tab.py`

Project the top-50 retrieved papers into 2D using the already-fitted BERTopic UMAP model. Color by topic cluster. Interactive Plotly scatter with hover showing title, year, citation count.

```python
def render(
    result_ids: list[str],
    topic_assignments: list[int],
    all_embeddings: np.ndarray,
    pid_to_idx: dict[str, int],
    topic_info: list[dict],
) -> None:
    """Render a 2D UMAP scatter plot of retrieved results coloured by topic."""
```

- Calls `umap_model.transform()` on the result subset — fast (transform, not fit)
- Mujtaba adds a "Semantic Map" tab to `main.py` routing to this component
- Zero new dependencies — UMAP already fitted, embeddings already in memory
- **Demo appeal:** Most visually striking output; shows semantic clustering at a glance

**Effort:** ~0.5 days

---

### B-R3: Topics-Over-Time Temporal Visualization
**Extend:** `src/app/components/topics_tab.py`

BERTopic has a built-in `topics_over_time()` method. Call it with paper publication years as timestamps to show how topic proportions shift across years (mirrors ISLE Figure 6).

```python
timestamps = [str(paper["year"]) for paper in result_papers]
tot = topic_model.topics_over_time(docs, timestamps, nr_bins=6, global_tuning=True)
fig = topic_model.visualize_topics_over_time(tot, top_n_topics=8)
st.plotly_chart(fig, use_container_width=True)
```

- Mentioned in `IMPLEMENTATION.md §3.3` but never assigned; Raahim owns BERTopic so owns this
- **Course tie-in:** Temporal IR; links naturally to Year nodes in the KG

**Effort:** ~0.5 days

---

## Abubakar's Bonus Tasks (Google Colab — CPU)

### B-A1: Interactive Rocchio Relevance Feedback
**New file:** `src/retrieval/rocchio.py`

Implement the Rocchio update rule so users can refine results by marking papers relevant/not-relevant in the UI:

```
q_new = α · q_orig + β · mean(relevant_embs) − γ · mean(non_relevant_embs)
```

```python
def rocchio(
    q_emb: np.ndarray,
    rel_embs: list[np.ndarray],
    nrel_embs: list[np.ndarray],
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> np.ndarray:
    """Return updated query embedding after Rocchio update."""
```

- Mujtaba adds thumbs-up/down buttons per result card and a "Refine Search" button in `results_tab.py` using `st.session_state`
- On click, calls `FAISSIndex.search_by_embedding()` with the updated embedding
- **Course tie-in:** Interactive relevance feedback (Manning Ch. 9 — Week 8); PRF is already done; Rocchio closes the loop with explicit user input

**Effort:** ~1 day

---

### B-A2: ILD Diversity Metric in Ablation Table
**Extend:** `evaluation/ablation.py`

Add Intra-List Diversity (ILD) as a 6th metric column alongside P@5/P@10/MRR/MAP. Quantifies the effect of MMR (B-R1) in the evaluation table.

```
ILD = (2 / k(k−1)) · Σ_{i≠j} (1 − cosine_sim(emb_i, emb_j))
```

- Fetch precomputed embeddings for each result set from `embeddings.npy` using the paper-ID-to-index map
- Expected result: MMR row has higher ILD than baseline rows; MRR drops < 10%
- **Course tie-in:** Diversity in IR (Week 7 evaluation concepts)

**Effort:** ~0.5 days

---

### B-A3: Citation Flow & Temporal Influence Analysis
**New files:** `evaluation/citation_flow.py` (logic) + add display to `src/app/components/analytics_tab.py`

Use citation edges and year metadata already in the KG to show how influence propagates through the result set over time.

- Build a temporal citation subgraph from the result set: for each year Y, sum outgoing citations to papers published in later years
- Identify "bridge papers": highest betweenness centrality in the citation subgraph (papers that bridge topic clusters)
- Visualize as a Plotly Sankey diagram (`plotly.graph_objects.Sankey`) — year slices as nodes, citation volume as flows

```python
def build_citation_flow(
    G: nx.DiGraph,
    result_ids: list[str],
) -> dict:
    """Return year-binned citation flow data for Sankey plot."""
```

- Mujtaba wires the Sankey chart into `analytics_tab.py`
- **Data requirement:** `references` (OpenAlex IDs) and `year` already in corpus; citation edges already in the KG from Phase 4
- **Course tie-in:** Bibliometric analysis + link analysis (Week 10)

**Effort:** ~1 day

---

## Mujtaba's Bonus Tasks (M4 Mac — Frontend/Graph)

### B-M1: Freshness-Boosted Re-ranking
**New file:** `src/retrieval/freshness.py`

Weight final RRF scores by a temporal freshness signal. Estimate per-topic citation decay rates from the corpus (fast-moving topics like LLMs decay quickly; foundational topics decay slowly):

```
freshness_boost(paper) = e^(−decay_rate_topic × years_old)
final_score = rrf_score × (1 + λ · freshness_boost)
```

```python
def apply_freshness(
    results: list[tuple[str, float]],
    papers_lookup: dict[str, dict],
    topic_assignments: dict[str, int],
    decay_rates: dict[int, float],  # topic_id → decay rate (precomputed from corpus)
    lam: float = 0.3,
) -> list[tuple[str, float]]:
    """Re-score results with freshness boost. Returns re-sorted list."""
```

- Wire as a post-RRF step in `pipeline.py` with `freshness: bool = False` flag
- Mujtaba adds a λ slider in the Streamlit sidebar
- Decay rates precomputed once at index-build time from corpus citation-year statistics
- **Course tie-in:** Temporal IR; motivates why retrieval for CS must account for field velocity

**Effort:** ~1 day

---

### B-M2: Multi-Hop Citation Path Explorer
**New file:** `src/graph/paths.py` + extend `src/app/components/graph_tab.py`

Given any two papers in the current result set, find the shortest citation path through the knowledge graph and highlight it.

```python
def find_citation_path(
    G: nx.DiGraph,
    source_id: str,
    target_id: str,
    max_hops: int = 4,
) -> list[str] | None:
    """Return list of arxiv_ids forming the shortest citation path, or None."""
```

- UI: Two `st.selectbox` dropdowns (from result set) + "Find Connection" button
- PyVis: re-render the subgraph with path edges highlighted in a distinct colour
- Falls back gracefully when no path exists within `max_hops`
- **Course tie-in:** Graph path ranking, link analysis (Week 10)

**Effort:** ~0.5 days

---

### B-M3: BibTeX / CSV Export
**New file:** `src/app/export.py`

One-click export of the current result set in researcher-friendly formats.

```python
def to_bibtex(results: list[dict]) -> str:
    """Render results as a valid BibTeX string (@article entries)."""

def to_csv(results: list[dict]) -> bytes:
    """Render results as UTF-8 CSV bytes for st.download_button."""
```

- BibTeX entry format: `@article{arxiv_id, title={...}, author={...}, year={...}, journal={arXiv}, url={https://arxiv.org/abs/arxiv_id}}`
- Add two `st.download_button` calls at the bottom of `results_tab.py`
- **Practical value:** The feature researchers would actually use — memorable in demos

**Effort:** ~0.5 days

---

## Task Summary

| ID | Task | Owner | New files | Effort |
|---|---|---|---|---|
| B-R1 | MMR diversity re-ranking | Raahim | `src/retrieval/mmr.py` | 1 day |
| B-R2 | UMAP 2D result scatter | Raahim | `src/app/components/embedding_viz_tab.py` | 0.5 days |
| B-R3 | Topics-over-time viz | Raahim | extend `topics_tab.py` | 0.5 days |
| B-A1 | Interactive Rocchio feedback | Abubakar | `src/retrieval/rocchio.py` | 1 day |
| B-A2 | ILD metric in ablation | Abubakar | extend `evaluation/ablation.py` | 0.5 days |
| B-A3 | Citation flow Sankey | Abubakar | `evaluation/citation_flow.py` | 1 day |
| B-M1 | Freshness-boosted re-ranking | Mujtaba | `src/retrieval/freshness.py` | 1 day |
| B-M2 | Multi-hop citation path explorer | Mujtaba | `src/graph/paths.py` | 0.5 days |
| B-M3 | BibTeX / CSV export | Mujtaba | `src/app/export.py` | 0.5 days |

**Per-person bonus effort:** Raahim 2 days · Abubakar 2.5 days · Mujtaba 2 days

---

## Execution Order

1. Mujtaba finishes Phase 5 (Streamlit wired) — unblocks all UI bonus tasks
2. Abubakar finishes Phase 6 (evaluation + labeling) — unblocks B-A2
3. Raahim finishes SPECTER2 vs MiniLM comparison (SSOT §10 remaining item)
4. Parallel bonus sprint:
   - Raahim: B-R2 → B-R3 → B-R1
   - Abubakar: B-A2 → B-A1 → B-A3
   - Mujtaba: B-M3 → B-M2 → B-M1

---

## Course Concept Coverage

Every bonus feature ties to a specific course week — this matters for the report:

| Feature | Course week / concept |
|---|---|
| B-R1 MMR | Ch. 9 — diversity-relevance tradeoff |
| B-R2 UMAP viz | Wk 13–14 — neural IR / representation learning |
| B-R3 Topics-over-time | Wk 12 — text clustering; temporal analysis |
| B-A1 Rocchio | Wk 8 — relevance feedback (interactive, not just PRF) |
| B-A2 ILD metric | Wk 7 — IR evaluation beyond relevance |
| B-A3 Citation flow | Wk 10 — link analysis + bibliometrics |
| B-M1 Freshness | Temporal IR; field velocity awareness |
| B-M2 Citation paths | Wk 10 — graph path ranking |
| B-M3 BibTeX export | Practical system quality |
