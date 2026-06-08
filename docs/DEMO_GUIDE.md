# ISLE-IR — Demo & Evaluation Guide

Quick reference for running and presenting the system.

---

## Prerequisites

```bash
# From repo root — one-time setup
uv sync
# Ensure data artifacts are present:
#   data/processed/corpus.parquet
#   data/faiss/faiss.index  (+ embeddings.npy, paper_ids.json)
#   data/bertopic/model/    (pre-fit BERTopic directory)
```

---

## 1. Launch the app

```bash
uv run streamlit run src/app/main.py
```

Opens at **http://localhost:8501**.

---

## 2. Core demo flow

| Step | What to do | What to show |
|------|-----------|--------------|
| Basic search | Query: *"transformer attention mechanism"* | 20 ranked results, scores, abstracts |
| Query expansion | Toggle **Query expansion** ON/OFF | Count of results changes; show synonym enrichment in BM25 recall |
| Reranker effect | Toggle **Cross-encoder rerank** ON/OFF | Order of top results shifts — CE scores vs RRF scores |
| Topics tab | Click **Topics** tab | BERTopic clusters, keyword clouds |
| Knowledge graph | Click **Knowledge Graph** tab | Interactive PyVis graph; paper/author/institution/country nodes |
| Analytics | Click **Analytics** tab | Year histogram, top cited papers, citation flow Sankey |
| Collaboration | Click **Collaboration Network** | Author co-authorship network, betweenness centrality |
| Semantic map | Click **Semantic Map** | UMAP/PCA 2-D projection of retrieved papers |

### Bonus features to highlight

- **MMR diversity** — enable sidebar toggle, move λ slider (1 = relevance, 0 = diversity); re-run query to show diversified set
- **Freshness boost** — enable toggle; papers from recent years bubble up
- **Rocchio feedback** — run a query → click 👍/👎 on result cards → hit **Refine Search** → result set updates
- **Export** — Results tab has **Download BibTeX** and **Download CSV** buttons

---

## 3. Evaluation run

```bash
# From repo root
source .venv/bin/activate
python evaluation/ablation.py
# Writes: evaluation/results/ablation_table.csv
```

Expected output (5 rows × 6 cols — System / P@5 / P@10 / MRR / MAP / ILD@10):

| System | P@5 | P@10 | MRR | MAP | ILD@10 |
|--------|-----|------|-----|-----|--------|
| BM25 only | — | — | — | — | — |
| Dense only | — | — | — | — | — |
| Hybrid (RRF) | — | — | — | — | — |
| Hybrid + Reranker | ↑ | ↑ | ↑ | ↑ | — |
| Full system | ↑↑ | ↑↑ | ↑↑ | ↑↑ | ↑ |

*(Actual values depend on your labelled queries in `evaluation/test_queries.json`.)*

### Corpus statistics

```bash
python evaluation/corpus_stats.py
# Writes PNG plots to evaluation/results/ for Heaps' Law and Zipf's Law
```

---

## 4. Smoke test (non-UI)

```bash
source .venv/bin/activate
python - <<'EOF'
import sys; sys.path.insert(0, "src")
import pandas as pd
from retrieval.bm25_index import BM25Index
from retrieval.faiss_index import FAISSIndex
from retrieval.pipeline import full_retrieval_pipeline

df = pd.read_parquet("data/processed/corpus.parquet")
papers_lookup = {r["arxiv_id"]: r for r in df.to_dict("records")}
bm25 = BM25Index(); bm25.load()
faiss = FAISSIndex(); faiss.load()

results = full_retrieval_pipeline(
    "transformer attention mechanism",
    bm25, faiss, papers_lookup, expand_query=True, rerank=True
)
print(f"Results: {len(results)}, top: {results[0]}")
assert len(results) <= 20
print("PASS")
EOF
```

---

## 5. Key artifacts

| Artifact | Path |
|----------|------|
| Corpus | `data/processed/corpus.parquet` |
| BM25 index | `data/bm25/` |
| FAISS index + embeddings | `data/faiss/faiss.index`, `data/faiss/embeddings.npy` |
| BERTopic model | `data/bertopic/model/` |
| Evaluation queries | `evaluation/test_queries.json` |
| Ablation results | `evaluation/results/ablation_table.csv` |

---

## 6. Stopping

`Ctrl-C` in the terminal running Streamlit.
