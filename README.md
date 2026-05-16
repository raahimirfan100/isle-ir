# ISLE-IR — Hybrid Information Retrieval Explorer

[![CI](https://github.com/raahimirfan100/isle-ir/actions/workflows/ci.yml/badge.svg)](https://github.com/raahimirfan100/isle-ir/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

A query-driven scientific-literature search system that fuses **lexical** and
**semantic** retrieval, clusters results into **topics**, and lets you explore
them through an interactive **knowledge graph** — built on the ideas in the
ISLE paper ([arXiv:2512.12760](https://arxiv.org/abs/2512.12760)).

Given a natural-language query, ISLE-IR runs BM25 and dense-vector retrieval in
parallel, fuses the rankings with Reciprocal Rank Fusion, models the topics of
the retrieved set with BERTopic / NMF, builds a heterogeneous knowledge graph
(papers · authors · institutions · countries · topics · years), and presents it
all in a Streamlit app.

> University course project (Information Retrieval). Reference docs:
> [`docs/SSOT.md`](docs/SSOT.md) (project spec) ·
> [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) (technical guide).

## Architecture

```
User query
   │
   ▼  Query expansion (WordNet + pseudo-relevance feedback)
   ├──────────────┬───────────────────┐
   ▼              ▼                   │
BM25 lexical   Dense retrieval        │
(bm25s)        (SentenceTransformers  │
   │            + FAISS)              │
   └──────┬───────┘                   │
          ▼  Reciprocal Rank Fusion   │
          ▼  Cross-encoder rerank     │
   ┌──────┴───────┐
   ▼              ▼
Topic modeling   Knowledge graph
(BERTopic / NMF) (NetworkX + PageRank)
   └──────┬───────┘
          ▼
   Streamlit UI  ·  Results · Topics · Graph · Analytics
```

See [`docs/SSOT.md` §3](docs/SSOT.md) for the full diagram and stage ownership.

## Tech stack

| Layer | Tools |
|---|---|
| Retrieval | `bm25s`, `sentence-transformers`, `faiss` |
| Topic modeling | `bertopic`, `umap-learn`, `hdbscan`, `scikit-learn` (NMF) |
| Knowledge graph | `networkx`, `pyvis`, `streamlit-agraph` |
| Data | `arxiv`, `pyalex` (OpenAlex), `pandas`, `pyarrow` |
| Frontend | `streamlit`, `plotly` |
| Evaluation | `gensim` (topic coherence) |

## Repository layout

```
src/ingestion/   arXiv + OpenAlex ingestion        → corpus.parquet
src/retrieval/   BM25, FAISS, RRF, reranker, query expansion, pipeline
src/topics/      NMF and BERTopic topic models
src/graph/       knowledge-graph builder, analytics, visualization
src/app/         Streamlit app + shared config.py
evaluation/      ablation study, corpus stats, test queries
docs/            SSOT.md, IMPLEMENTATION.md, agent-prompts/, papers/
```

## Quickstart

ISLE-IR uses [`uv`](https://docs.astral.sh/uv/) for environment management.
Install uv once, then run `setup.sh` — it creates `.venv` (Python 3.11, pinned
in `.python-version`) and picks the GPU or CPU build automatically.

```bash
# 1. Install uv (once per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS / Linux / WSL
#   Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

# 2. Set up the environment — picks GPU (Linux/WSL + NVIDIA) or CPU
./setup.sh                       # or force it:  ./setup.sh gpu | ./setup.sh cpu

# 3. Run the app (after the data + index phases — see docs/agent-prompts/)
uv run streamlit run src/app/main.py
```

The two hardware variants — `uv sync` installs the project editable, so the
`src/` packages import without any `PYTHONPATH` hack:

```bash
uv sync --extra gpu     # torch (CUDA 12.6) + faiss-gpu   — RTX rig, Google Colab
uv sync --extra cpu     # torch (CPU / Apple MPS) + faiss-cpu — M4 Mac, Intel laptop
```

GPU mode is **Linux/WSL only** — `faiss-gpu-cu12` publishes Linux x86-64 wheels
only, so on Windows and macOS `setup.sh` always selects `cpu` (and an explicit
`./setup.sh gpu` there exits with an error).

No `uv` on the machine? `pip install -r requirements-gpu.txt` (or
`requirements-cpu.txt`) are uv-exported fallbacks; `uv.lock` is the source of
truth for reproducible installs.

Data and index artifacts (`corpus.parquet`, `embeddings.npy`, FAISS/BM25
indexes) are **not** in git — they are distributed as GitHub Release assets.
See [`docs/SSOT.md` §15](docs/SSOT.md).

## Project status

Bootstrap phase — repository scaffolded, modules are contract stubs awaiting
implementation. Work proceeds phase by phase; each phase has a ready-to-use
prompt in [`docs/agent-prompts/`](docs/agent-prompts/).

## Team

| Member | ID | Owns |
|---|---|---|
| Abubakar Ahmed | 23k-0801 | BM25 · RRF · NMF · query expansion · evaluation |
| Raahim Irfan | 23k-0837 | Embeddings · FAISS · BERTopic · reranker · integration |
| Mujtaba Kamran | 23k-0715 | Data ingestion · knowledge graph · Streamlit UI |

## Contributing

Working on this repo — human or AI agent? Read [`CONTRIBUTING.md`](CONTRIBUTING.md)
and [`AGENTS.md`](AGENTS.md) first.

## License

[MIT](LICENSE) © 2026 Abubakar Ahmed, Raahim Irfan, Mujtaba Kamran
