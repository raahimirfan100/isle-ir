# Bonus Features — Mujtaba

> Copy everything below the divider into your agent. Owner: Mujtaba.
> Prerequisite: Phase 5 (Streamlit) must be wired before B-M1 and B-M2 can
> be exercised in the UI. B-M3 (BibTeX export) can be added once
> `results_tab.py` has a working `render()`.

---

You are working on **ISLE-IR**. Read `AGENTS.md` first, then implement
**Mujtaba's bonus features** as described in `docs/BONUS_FEATURES.md`.

## Read first
- `docs/BONUS_FEATURES.md` — full spec for B-M1, B-M2, B-M3.
- `src/graph/builder.py` and `src/graph/analytics.py` — the KG structure and
  existing analytics patterns.
- `src/retrieval/pipeline.py` — see how `rerank` and `expand_query` flags are
  wired; freshness follows the same pattern.
- `src/app/main.py` and `src/app/components/` — where to add UI wiring.
- `src/app/config.py` — all paths and constants.

## Goal
Three frontend and graph features: freshness-boosted re-ranking, a multi-hop
citation path explorer in the KG visualization, and BibTeX/CSV export of
results.

---

## B-M1 — Freshness-Boosted Re-ranking

**New file:** `src/retrieval/freshness.py`

Re-score results after RRF by multiplying by a temporal freshness signal.
Papers in fast-moving topics (e.g. LLMs) decay fast; foundational topics decay
slowly. Decay rates are precomputed from the corpus once.

### Interface
```python
def estimate_decay_rates(
    corpus_df: pd.DataFrame,
    topic_assignments_full: list[int],
    current_year: int = 2025,
) -> dict[int, float]:
    """
    Estimate a citation-velocity decay rate per topic.

    For each topic, compute mean citations-per-year-old across its papers.
    Topics where recent papers get proportionally more citations → higher decay rate.
    Returns {topic_id: decay_rate} where decay_rate > 0.
    """

def apply_freshness(
    results: list[tuple[str, float]],
    papers_lookup: dict[str, dict],
    topic_assignments: dict[str, int],
    decay_rates: dict[int, float],
    lam: float = 0.3,
    current_year: int = 2025,
) -> list[tuple[str, float]]:
    """
    Re-score and re-sort results with a freshness boost.

    final_score = rrf_score × (1 + λ · e^(−decay_rate × years_old))

    Returns re-sorted [(arxiv_id, boosted_score), ...].
    """
```

### Steps
1. Implement `estimate_decay_rates`: for each BERTopic cluster, compute
   `mean(cited_by_count / max(1, current_year - year))` across papers in that
   cluster. Use `current_year=2025`. Clip to a sensible range (0.01–1.0).
   This runs once at startup; cache the result in the Streamlit app with
   `@st.cache_data`.
2. Implement `apply_freshness` with the exponential decay formula above.
   `years_old = current_year - paper["year"]`; clip at 0 to handle pre-dated
   papers.
3. Wire into `pipeline.py`: add `freshness: bool = False, freshness_lambda: float = 0.3`
   to `full_retrieval_pipeline`; call `apply_freshness` as a post-RRF step
   (before reranking, so the cross-encoder still refines the boosted set).
4. Add `FRESHNESS_LAMBDA = 0.3` to `src/app/config.py`.
5. In `main.py` sidebar, add:
   ```python
   use_freshness = st.toggle("Freshness boost", value=False)
   freshness_lambda = st.slider("Freshness λ", 0.0, 1.0, 0.3) if use_freshness else 0.3
   ```
6. Write a unit test in `tests/test_freshness.py`: create two fake results,
   one from 2020 and one from 2024 with equal RRF scores; confirm the 2024
   paper ranks first after `apply_freshness`.

### Exit criteria
- [ ] `estimate_decay_rates` returns a non-empty dict with positive values.
- [ ] `apply_freshness` returns the same number of results, re-sorted.
- [ ] A recent paper with equal RRF score outranks an older paper when
      `freshness=True`.
- [ ] Unit test passes.
- [ ] Sidebar toggle and slider render; toggling changes result order.
- [ ] `ruff check .` clean.

---

## B-M2 — Multi-Hop Citation Path Explorer

**New file:** `src/graph/paths.py`
**Extend:** `src/app/components/graph_tab.py`

Given any two papers in the current result set, find the shortest citation
path between them in the KG and highlight it in the PyVis graph.

### Interface
```python
def find_citation_path(
    G: nx.DiGraph,
    source_id: str,
    target_id: str,
    max_hops: int = 4,
) -> list[str] | None:
    """
    Find the shortest directed citation path from source_id to target_id.

    Only traverses 'cites' edges. Returns a list of arxiv_ids forming the
    path (inclusive of source and target), or None if no path exists within
    max_hops.
    """
```

### Steps
1. Build a **citation-only subgraph** from `G`:
   ```python
   cites_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("rel") == "cites"]
   citation_sub = nx.DiGraph(cites_edges)
   ```
2. Use `nx.shortest_path(citation_sub, source=source_id, target=target_id)`.
   Catch `nx.NetworkXNoPath` and `nx.NodeNotFound`; return `None` for both.
   Apply the `max_hops` limit by checking `len(path) - 1 <= max_hops`.
3. In `graph_tab.py`, add below the main PyVis graph:
   ```
   st.subheader("Citation Path Explorer")
   col_a, col_b = st.columns(2)
   paper_a = col_a.selectbox("From paper", result_titles)
   paper_b = col_b.selectbox("To paper", result_titles)
   if st.button("Find connection"):
       path = find_citation_path(G, id_a, id_b, max_hops=4)
       if path:
           # Re-export PyVis with path edges highlighted in #FFD700
       else:
           st.warning("No citation path found within 4 hops.")
   ```
4. For the highlighted PyVis: call `export_pyvis` as usual, then iterate the
   returned `Network` object and colour the path edges differently before
   calling `net.save_graph()`.
5. Fallback: if the citation subgraph has fewer than 2 edges in the result set,
   hide the UI with `st.info("Too few intra-set citation links for path
   exploration.")`.

### Exit criteria
- [ ] `find_citation_path` returns a valid path for two papers that do cite
      each other (verify manually on one known pair).
- [ ] Returns `None` when no path exists within 4 hops.
- [ ] UI dropdowns populate from the current result set.
- [ ] Highlighted path renders visibly in the PyVis graph.
- [ ] `ruff check .` clean.

---

## B-M3 — BibTeX / CSV Export

**New file:** `src/app/export.py`

One-click export of the current result set in researcher-friendly formats.

### Interface
```python
def to_bibtex(results: list[dict]) -> str:
    """
    Render results as a BibTeX string.

    Each entry: @article{arxiv_id,
      title = {...},
      author = {First Last and First Last},
      year = {2023},
      journal = {arXiv},
      url = {https://arxiv.org/abs/arxiv_id},
    }
    """

def to_csv(results: list[dict]) -> bytes:
    """
    Render results as UTF-8 CSV bytes (for st.download_button).
    Columns: arxiv_id, title, authors, year, categories, cited_by_count, url.
    """
```

### Steps
1. Implement `to_bibtex`: escape curly braces in titles with `{{}}` if
   they appear; join authors with `" and "`. Keep it valid — test with an
   online BibTeX linter.
2. Implement `to_csv`: use `pd.DataFrame(results)[columns].to_csv(index=False).encode("utf-8")`.
3. In `results_tab.py`, add at the bottom of the `render()` function:
   ```python
   col1, col2 = st.columns(2)
   bibtex_str = to_bibtex(papers_list)
   col1.download_button("⬇ Export BibTeX", bibtex_str, "results.bib", "text/plain")
   csv_bytes = to_csv(papers_list)
   col2.download_button("⬇ Export CSV", csv_bytes, "results.csv", "text/csv")
   ```
4. Write a unit test in `tests/test_export.py`:
   - `to_bibtex` output starts with `@article{` and contains the paper's arxiv_id.
   - `to_csv` output is valid UTF-8 and the first row is a header.

### Exit criteria
- [ ] Clicking "Export BibTeX" downloads a `.bib` file parseable by BibTeX
      (test with `bibtex` CLI or an online tool).
- [ ] Clicking "Export CSV" downloads a readable CSV.
- [ ] Both buttons appear in the Results tab after a search.
- [ ] Unit tests pass.
- [ ] `ruff check .` clean.

---

## UI wiring for Raahim and Abubakar's bonus features

While implementing Phase 5, also wire these hooks that Raahim and Abubakar
own algorithmically but that require Streamlit session state on your side:

### Rocchio thumbs (for B-A1)
In `results_tab.py`, per result card:
```python
cols = st.columns([6, 1, 1])
with cols[1]:
    if st.button("👍", key=f"rel_{pid}"):
        st.session_state.setdefault("rel_ids", []).append(pid)
with cols[2]:
    if st.button("👎", key=f"nrel_{pid}"):
        st.session_state.setdefault("nrel_ids", []).append(pid)
```
Add a "🔄 Refine Search" button in the sidebar; on click, fetch embeddings for
`rel_ids` and `nrel_ids`, call `rocchio()`, then `faiss_index.search_by_embedding()`.

### MMR λ slider (for B-R1)
In `main.py` sidebar:
```python
use_mmr = st.toggle("MMR diversity", value=False)
mmr_lambda = st.slider("MMR λ (1=relevance, 0=diversity)", 0.0, 1.0, 0.5) if use_mmr else 0.5
```
Pass `mmr=use_mmr, mmr_lambda=mmr_lambda` to `full_retrieval_pipeline`.

### Semantic Map tab (for B-R2)
In `main.py`, add `"🗺 Semantic Map"` as a 5th tab and route it to
`embedding_viz_tab.render(...)`.

---

## Commit & branch
Branch `feat/bonus-frontend` off `dev`.

Small commits:
- `feat(retrieval): implement freshness-boosted re-ranking`
- `test(retrieval): add freshness unit test`
- `feat(graph): implement citation path finder`
- `feat(app): add citation path explorer to graph tab`
- `feat(app): add BibTeX and CSV export`
- `test(app): add export unit tests`
- `feat(app): wire Rocchio, MMR, and semantic map in sidebar`
