# ISLE-IR Project — Single Source of Truth (SSOT)
> **Last updated:** May 2026 · **Status:** Active development
> **Course weight:** Class Project = 10% of final grade
> **Deliverables:** Code + Report + Presentation/Demo · **Timeline:** 4+ weeks
>
> **Canonical project spec** — team roles, repo layout, interface contracts,
> timeline, report/presentation division, evaluation. Phase-by-phase code,
> library choices and pitfalls live in [`IMPLEMENTATION.md`](IMPLEMENTATION.md);
> the two are complementary, not redundant. Reference papers: [`papers/`](papers/).

---

## ⚠️ Proposal Situation — Read This First

### What was submitted vs. what was intended

| | Intended Proposal (Raahim) | Submitted Proposal (teammates) |
|---|---|---|
| **Reference** | "a system introduced in arXiv:2512.12760" | "inspired by the paper ISLE (2025)" |
| **Retrieval** | "BM25 + dense embeddings via RRF" | "BM25 + dense vector embeddings via RRF" |
| **Topics** | "BERTopic for query-driven topic clustering" | "BERTopic **or** NMF" |
| **KG** | "heterogeneous KG connecting papers, authors, institutions, topics" | "lightweight KG representing relationships between documents, topics, metadata" |
| **Evaluation** | "NDCG, MAP, topic coherence, reproducing paper's benchmarks" | *(no specific metrics mentioned)* |
| **Data** | "arXiv and OpenAlex public APIs" | *(not mentioned)* |

### What this actually means for you

The submitted proposal is **less specific but not a problem** — it gives you more flexibility. Three things to note:

1. **"BERTopic or NMF"** — you implement both (which you were going to anyway). This is fine, arguably better scope management.
2. **"Lightweight KG"** — the submitted proposal set lower expectations on the KG. You can exceed them visibly, which looks better than falling short of ambitious promises.
3. **No evaluation metrics committed** — this means you can choose whichever metrics you implement best. NDCG requires ground-truth relevance grades (0–3 scale), which is more work than binary judgments for MAP/MRR. Stick with **MAP, MRR, P@5, P@10** and topic coherence (C_V). Easier to compute, equally valid academically.

**One thing you should do:** In your report introduction, write: *"We implement the full system as proposed, and additionally evaluate retrieval quality via MAP, MRR, P@5, and P@10 metrics, with topic quality via C_V coherence scores."* This frames exceeding the proposal as intentional.

---

## 📚 Course Outline Coverage Analysis

The project is worth 10% — the teacher almost certainly cares that it demonstrates command of course concepts, not just that it works. Here is every week mapped to your system:

### ✅ Strongly Covered (explicitly implemented)

| Week | Topic | Where in your system | How to emphasize in report |
|---|---|---|---|
| **Wk 2** | Document Processing, Term vocabulary, Stemming/Lemmatization | BM25 preprocessing uses English stemmer (`PyStemmer`) + stopword removal before indexing | Report Section 5: mention stemming choice explicitly, show vocabulary size before/after |
| **Wk 2** | Posting lists | `bm25s` internally builds inverted index with posting lists | One sentence in report: "BM25S constructs an inverted index with posting lists..." |
| **Wk 4** | Index construction | BM25 index build (in-memory, single-pass). FAISS index = vector index construction | Report Section 5: describe both as index construction, contrast document vs. vector index |
| **Wk 4** | Heaps' Law / Zipf's Law | **Run corpus statistics notebook** — plot vocabulary growth curve (Heaps) and term frequency distribution (Zipf) on your 100k paper corpus | Adds half a page to report with real plots. Trivial to implement. |
| **Wk 5** | Vector Space Model / TF-IDF | NMF pipeline uses TF-IDF as document representation. BM25 is a probabilistic extension of TF-IDF | Report Section 6 (NMF): "Documents are represented using TF-IDF, consistent with the Vector Space Model (Manning et al., Ch. 6)" |
| **Wk 7** | Evaluation of IR (MAP, MRR, P@k) | Your entire ablation study section | Report Section 9: full metrics table. This is your strongest course-concept tie-in |
| **Wk 8** | Relevance Feedback | **Pseudo-Relevance Feedback (PRF)** is explicitly in your query expansion | Report Section 5: "Query expansion uses PRF (Rocchio, Ch. 9) — top-3 BM25 results provide feedback terms" |
| **Wk 9** | Probabilistic Language Model / BM25 | BM25 is a probabilistic retrieval model | Report Section 5: cite Manning Ch. 11, derive BM25 equation, explain k1 and b parameters |
| **Wk 10** | Link Analysis | **PageRank on citation graph** is explicitly implemented | Report Section 7: "We apply PageRank (Page et al., 1999) to the citation subgraph, treating citations as directed links analogous to web hyperlinks" |
| **Wk 12** | Text Clustering | BERTopic + NMF are both clustering approaches on retrieved text | Report Section 6: entire section |
| **Wk 13** | Neural Information Retrieval | SPECTER2 dense retrieval, cross-encoder reranker, BERTopic neural pipeline | Report Section 5 + 8: neural retrieval as contrast to lexical BM25 |
| **Wk 14** | Learning Word Representation | SPECTER2 embeddings = learned scientific paper representations | Report Section 5: describe SPECTER2 training objective (citation-aware) |

### 🟡 Partially / Implicitly Covered (mention briefly)

| Week | Topic | How to get credit |
|---|---|---|
| **Wk 1** | Boolean IR | One sentence in Related Work: "Traditional systems relied on Boolean retrieval; our system moves beyond this via probabilistic and semantic retrieval" |
| **Wk 3** | Spelling Correction / Edit Distance | `bm25s` supports fuzzy matching. Mention it as a built-in feature: "BM25S incorporates fuzzy matching to handle morphological variation" |
| **Wk 10** | Web Crawling / Indexing | arXiv + OpenAlex ingestion is analogous to crawling and indexing. Frame it: "Rather than a live web crawler, we use the arXiv API as a structured crawl target" |
| **Wk 15** | RAG | Knowledge graph + retrieval is architecturally adjacent to RAG. One sentence: "The system's retrieve-then-analyze pipeline shares architectural principles with Retrieval-Augmented Generation" |
| **Wk 16** | Contrastive Learning | SPECTER2 is trained with contrastive objectives (citation pairs as positive examples). One sentence in Section 8 covers this |

### ❌ Not Covered (genuinely out of scope — don't stretch)

| Week | Topic | Why skip |
|---|---|---|
| **Wk 3** | Permuterm / k-gram index for wildcard | Your system doesn't support wildcard queries — don't claim it does |
| **Wk 4** | Variable-length coding (gamma, delta codes) | Implementation detail of compressed indexes, not relevant to your system |
| **Wk 15** | Full RAG pipeline | Out of scope unless you add LLM answer generation — not worth the effort at 10% course weight |

### Coverage Score: 13/16 weeks touched by your project

This is excellent for a 10% project. The key is that your report explicitly calls out the course concepts by name and chapter reference so the teacher sees the connections.

---

## 2. Team Overview

| | Abubakar Ahmed · 23k-0801 | Raahim Irfan · 23k-0837 | Mujtaba Kamran · 23k-0715 |
|---|---|---|---|
| **Hardware** | Google Colab (T4 GPU free tier) | RTX 3060 12GB · 64GB DDR5 · Ryzen 7 7700 · 990 Pro NVMe | M4 MacBook Air |
| **Primary role** | Retrieval core + Evaluation | Embeddings + GPU models + Final integration | Data pipeline + KG + Frontend |
| **Owns** | BM25 · RRF · NMF · Query expansion · Ablation study · Report sections 1,2,7,9,11 | SPECTER2 · FAISS · BERTopic · Cross-encoder · More Like This · pipeline.py · Report sections 5,8 | arXiv/OA ingestion · KG · Author network · PageRank · Streamlit UI · Report sections 3,4,6,10 |
| **Presentation** | Evaluation results + key findings | Live demo narration | Architecture diagrams + KG walkthrough |

**Why this split.** GPU-heavy, time-sensitive work runs on Raahim's RTX 3060
(SPECTER2/dense encoding, FAISS-GPU index build, BERTopic UMAP+HDBSCAN,
cross-encoder reranking) — it cannot finish on a Colab free session or an M4
within the time budget, and final integration runs there too. The data
pipeline, knowledge graph (NetworkX is CPU-bound, not GPU-bound) and Streamlit
UI run well on Mujtaba's M4. BM25, NMF and RRF are pure CPU/Python and suit
Abubakar's Colab, which disconnects after ~12 h — so his jobs stay short and
notebook-driven.

---

## 3. System Architecture

```
User Query (natural language)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  QUERY EXPANSION                    [Abubakar]       │
│  WordNet synonyms + Pseudo-Relevance Feedback        │
│  Course tie-in: Week 8 — Relevance Feedback (Ch.9)  │
└──────────────────────┬──────────────────────────────┘
                       │ expanded query
          ┌────────────▼────────────┐
          │                         │
┌─────────▼────────┐   ┌────────────▼─────────────┐
│   BM25 Lexical   │   │  SPECTER2 Dense Retrieval │
│   [Abubakar]     │   │  [Raahim]                 │
│  Week 9: Prob.   │   │  Week 13-14: Neural IR +  │
│  Language Model  │   │  Word Representation      │
└─────────┬────────┘   └────────────┬─────────────┘
          │ top-100                  │ top-100
          └────────────┬────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  RECIPROCAL RANK FUSION (k=60)      [Abubakar]       │
│  Week 7: Evaluation of IR — rank aggregation         │
└──────────────────────┬──────────────────────────────┘
                       │ top-50
┌──────────────────────▼──────────────────────────────┐
│  CROSS-ENCODER RERANKER             [Raahim]         │
│  ms-marco-MiniLM-L-6-v2 scores (query, abstract)    │
│  Week 13: Neural IR                                  │
└──────────────────────┬──────────────────────────────┘
                       │ top-20 final results
          ┌────────────▼────────────┐
          │                         │
┌─────────▼────────┐   ┌────────────▼──────────────┐
│  TOPIC MODELING  │   │  KNOWLEDGE GRAPH           │
│  NMF [Abubakar]  │   │  [Mujtaba]                 │
│  BERTopic[Raahim]│   │  Papers·Authors·Topics·    │
│  Week 12:        │   │  Institutions·Countries    │
│  Text Clustering │   │  Week 10: Link Analysis    │
└─────────┬────────┘   └────────────┬──────────────┘
          │                         │
          └────────────┬────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  STREAMLIT FRONTEND                 [Mujtaba]        │
│  Results · Topics · KG · Analytics · Collab Network  │
└─────────────────────────────────────────────────────┘
```

---

## 4. Repository Structure

```
isle-ir/
├── data/
│   ├── raw/                         # Mujtaba populates
│   │   └── arxiv-metadata-oai-snapshot.json   # Kaggle download
│   ├── processed/
│   │   └── corpus.parquet           # CONTRACT: see Section 6A
│   └── indexes/
│       ├── bm25/                    # Abubakar builds
│       └── faiss/
│           ├── vectors.index        # Raahim builds
│           ├── paper_ids.json       # Raahim builds
│           └── embeddings.npy       # Raahim builds — shared with BERTopic+Mujtaba
│
├── src/
│   ├── ingestion/
│   │   ├── arxiv_loader.py          # Mujtaba
│   │   └── openalex_enrich.py       # Mujtaba
│   ├── retrieval/
│   │   ├── bm25_index.py            # Abubakar
│   │   ├── faiss_index.py           # Raahim
│   │   ├── rrf.py                   # Abubakar
│   │   ├── reranker.py              # Raahim
│   │   ├── query_expansion.py       # Abubakar
│   │   └── pipeline.py              # Raahim — wires all retrieval stages
│   ├── topics/
│   │   ├── nmf_model.py             # Abubakar
│   │   └── bertopic_model.py        # Raahim
│   ├── graph/
│   │   ├── builder.py               # Mujtaba
│   │   ├── analytics.py             # Mujtaba — PageRank, impact
│   │   ├── collab_network.py        # Mujtaba — co-authorship graph
│   │   └── visualization.py         # Mujtaba — PyVis + streamlit-agraph
│   └── app/
│       ├── main.py                  # Mujtaba (layout) — Raahim reviews
│       ├── config.py                # ALL — shared constants/paths
│       └── components/
│           ├── results_tab.py       # Abubakar
│           ├── topics_tab.py        # Abubakar
│           ├── graph_tab.py         # Mujtaba
│           └── analytics_tab.py     # Mujtaba
│
├── evaluation/
│   ├── test_queries.json            # Abubakar — 20 labeled queries
│   ├── ablation.py                  # Abubakar
│   ├── corpus_stats.py              # Abubakar — Heaps+Zipf plots (Week 4 tie-in)
│   └── results/
│       └── ablation_table.csv       # produced by ablation.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Mujtaba
│   ├── 02_retrieval_sanity.ipynb    # Raahim + Abubakar
│   └── 03_topic_eval.ipynb          # Abubakar
│
├── report/                          # See Section 9
├── pyproject.toml                   # dependencies + uv config — Raahim maintains
├── uv.lock                          # pinned, reproducible install — committed
├── setup.sh                         # one-command environment setup
└── README.md                        # Mujtaba drafts
```

### Git Workflow

```
main  ← protected, merge via PR (no required review)
  └── dev  ← integration branch
        ├── feat/bm25-rrf
        ├── feat/nmf
        ├── feat/faiss-index
        ├── feat/bertopic
        ├── feat/ingestion
        ├── feat/knowledge-graph
        └── feat/streamlit-ui
```

Branches are named `feat/<feature>` (also `fix/`, `docs/`, `chore/`) — feature,
not developer. The repo is public; branch names should not carry personal names.

**Rules:** Feature branches merge into `dev` directly — no PR required. `main`
changes go through a PR too, but no review is required — the author can
self-merge. Never force-push or delete `dev` or `main`. Raahim owns the final
integration merge to `main`.

---

## 5. Shared Config

```python
# src/app/config.py — everyone imports from here, no hardcoded paths anywhere else

from pathlib import Path

ROOT            = Path(__file__).parent.parent.parent
DATA_RAW        = ROOT / "data" / "raw"
CORPUS_PATH     = ROOT / "data" / "processed" / "corpus.parquet"
BM25_DIR        = ROOT / "data" / "indexes" / "bm25"
FAISS_DIR       = ROOT / "data" / "indexes" / "faiss"

# Models
EMBEDDING_MODEL = "allenai/specter2_base"        # 768-dim, scientific text
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM   = 768

# BM25 — from ISLE paper Section 3.2.1
BM25_K1         = 1.5
BM25_B          = 0.75

# Retrieval
TOP_K_EACH      = 100    # BM25 and FAISS each retrieve this many
RRF_K           = 60     # Cormack et al. 2009 standard constant
RRF_TOP_N       = 50     # after fusion
RERANK_TOP_K    = 20     # cross-encoder final output

# FAISS IVFFlat
FAISS_NLIST     = 1024
FAISS_NPROBE    = 64

# BERTopic
BERTOPIC_MIN_CLUSTER  = 15
BERTOPIC_N_COMPONENTS = 5   # UMAP output dims
BERTOPIC_N_NEIGHBORS  = 15

# NMF
NMF_K_RANGE     = range(5, 21)

# OpenAlex — register free key at openalex.org/api-key
OA_EMAIL        = "your-team-email@nu.edu.pk"
OA_API_KEY      = "YOUR_FREE_KEY"
```

---

## 6. Interface Contracts

These are the function signatures that connect each person's work. Your internal implementation is your own choice. Your output shape is not.

### Contract A — Corpus Schema (Mujtaba → Everyone)
**DEADLINE: End of Week 1. Nothing else starts without this.**

```python
# Every row of corpus.parquet must have exactly these columns:
{
    "arxiv_id":       str,        # "2301.07041"
    "title":          str,        # cleaned, no newlines
    "abstract":       str,        # cleaned, no newlines
    "text":           str,        # = title + " " + abstract — pre-joined
    "authors":        list[str],  # ["Yann LeCun", "Geoffrey Hinton"]
    "year":           int,        # 2023
    "categories":     list[str],  # ["cs.LG", "cs.AI"]
    "doi":            str,        # may be empty ""
    "cited_by_count": int,        # from OpenAlex, 0 if not found
    "institutions":   list[str],  # from OpenAlex
    "countries":      list[str],  # ISO codes, from OpenAlex
    "references":     list[str],  # OpenAlex work IDs ["W2964149567", ...]
}
```

### Contract B — BM25 (Abubakar)
```python
class BM25Index:
    def build(self, papers: list[dict]) -> None: ...
    def load(self) -> None: ...
    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Returns [(arxiv_id, bm25_score), ...] sorted descending."""
```

### Contract C — FAISS (Raahim)
```python
class FAISSIndex:
    def build(self, papers: list[dict], batch_size: int = 128) -> None: ...
    def load(self) -> None: ...
    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Returns [(arxiv_id, cosine_similarity), ...] sorted descending."""
    def search_by_embedding(self, embedding: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """For 'More Like This' — takes precomputed 768-dim vector."""
```

### Contract D — Full Retrieval Pipeline (Raahim)
```python
def full_retrieval_pipeline(
    query: str,
    bm25_index: BM25Index,
    faiss_index: FAISSIndex,
    papers_lookup: dict[str, dict],
    expand_query: bool = True,
    rerank: bool = True,
) -> list[tuple[str, float]]:
    """
    Returns [(arxiv_id, final_score), ...] top-20, reranked.
    Internally calls: query_expansion → BM25 → FAISS → RRF → cross-encoder.
    """
```

### Contract E — Topic Modeling (Abubakar + Raahim → Mujtaba's KG)
```python
# Both NMF and BERTopic return identical shapes:
def run_nmf(docs: list[str], n_topics: int) -> tuple[list[dict], list[int], list[float]]: ...
def run_bertopic(docs: list[str], embeddings: np.ndarray) -> tuple[list[dict], list[int], list[float]]: ...

# topic_info list shape:
# [{"topic_id": int, "keywords": list[str], "doc_count": int, "label": str}, ...]
# doc_topics: list[int]    — one per document, -1 = outlier (BERTopic only)
# doc_probs:  list[float]  — confidence/weight for assigned topic
```

### Contract F — Knowledge Graph (Mujtaba)
```python
def build_knowledge_graph(
    papers: list[dict],
    topic_assignments: list[int],
    topic_probs: list[float],
    topic_info: list[dict],
) -> nx.DiGraph:
    """
    Node types (node_type attr): paper, author, institution, country, topic, year
    Edge types (rel attr): authored, affiliated, from_country,
                           published_in, belongs_to, cites
    """
```

---

## 7. Week-by-Week Timeline

### Week 1 — Environment + Data (Days 1–7)
**Exit criteria:** `corpus.parquet` exists; BM25 and FAISS both return results on a 5k sample.

| | Abubakar (Colab) | Raahim (RTX rig) | Mujtaba (M4) |
|---|---|---|---|
| Day 1–2 | Set up Colab, clone repo, `./setup.sh`, verify `bm25s` + `gensim` import cleanly | `./setup.sh` (uv, GPU extra), verify `torch.cuda.is_available()` + `faiss.get_num_gpus()` | Download arXiv Kaggle snapshot (`kaggle datasets download -d Cornell-University/arxiv`) |
| Day 3–4 | Write `bm25_index.py`, test on 5k paper sample from Mujtaba | Test SPECTER2 encoding on 5k sample, confirm dim=768, time the batch | Write `arxiv_loader.py`, filter cs.* categories, output parquet with all Contract A columns |
| Day 5–6 | Write `rrf.py`, unit test with dummy ranked lists (no actual index needed) | Write `faiss_index.py`, `IndexIVFFlat` build+search, confirm nlist=1024 training | Write `openalex_enrich.py`, enrich 1k papers via DOI, verify institutions+countries populate |
| Day 7 | **✓ BM25 search works on 5k sample; RRF unit tests pass** | **✓ FAISS search works on 5k sample; embeddings.npy on GitHub Releases** | **✓ corpus.parquet committed: 50k papers minimum, all columns present** |

**Hard dependency:** Mujtaba's parquet must be available by Day 3 at minimum as a 5k sample, even if the full corpus isn't ready until Day 7.

---

### Week 2 — Full Retrieval Pipeline (Days 8–14)
**Exit criteria:** Full hybrid pipeline runs end-to-end on full corpus in <15 seconds per query.

| | Abubakar | Raahim | Mujtaba |
|---|---|---|---|
| Day 8–9 | Build BM25 on full corpus (~100k papers), commit index files | Build FAISS on full corpus with SPECTER2 — **start this as an overnight job** | Begin `builder.py` — paper + author + year nodes |
| Day 10–11 | Write `query_expansion.py`: WordNet synonyms first, PRF second | Write `reranker.py` with cross-encoder; test latency on RTX (target: <3s for 50 pairs) | Add institution + country + topic placeholder nodes to KG builder |
| Day 12–13 | Create `evaluation/test_queries.json` — 20 queries from the list in Section 10; write ablation harness skeleton | Write `pipeline.py` integrating BM25+FAISS+RRF+reranker using Contract D signature | Write `collab_network.py` — co-authorship edges weighted by shared paper count |
| Day 14 | **✓ 5 sanity queries through full pipeline, results look plausible** | **✓ `pipeline.py` importable and returns 20 results for any string query** | **✓ KG builds without errors for a 50-paper set** |

---

### Week 3 — Topic Modeling + Graph Analytics (Days 15–21)
**Exit criteria:** Topics attached to KG; PageRank scores computed; coherence metrics logged.

| | Abubakar | Raahim | Mujtaba |
|---|---|---|---|
| Day 15–16 | Run NMF k-sweep (k=5→20), pick best k by c_v, log coherence scores | Run BERTopic on 50-paper retrieved set using precomputed embeddings from `embeddings.npy` | Write `analytics.py`: PageRank on citation subgraph + entity impact (Eq. 20 from ISLE) |
| Day 17–18 | Write `corpus_stats.py`: Heaps' Law vocabulary growth plot + Zipf term frequency plot on full corpus | Implement `topics_over_time()` (BERTopic built-in) → export Plotly HTML | Wire topic assignments into KG builder via Contract E; add `belongs_to` edges |
| Day 19–20 | Start labeling `test_queries.json` — binary relevance for top-5 per query (this takes time, do it steadily) | Implement "More Like This" in `faiss_index.py` via `search_by_embedding` | Write `visualization.py`: PyVis HTML export + streamlit-agraph inline render |
| Day 21 | **✓ NMF coherence score logged; k selected; `corpus_stats.py` produces two plots** | **✓ BERTopic coherence > NMF (verify); "More Like This" returns in <1s** | **✓ Full KG with all 6 node types; PageRank scores on all paper nodes** |

**Shared file transfer (Day 21):** Raahim publishes `embeddings.npy` to a GitHub Release for Mujtaba to use in KG topic probability embedding.

---

### Week 4 — Streamlit App + Full Integration (Days 22–28)
**Exit criteria:** Demo runs on Raahim's machine. All tabs functional. Ablation numbers exist.

| | Abubakar | Raahim | Mujtaba |
|---|---|---|---|
| Day 22–23 | Write `results_tab.py` (result cards with RRF score, year, categories, "More Like This" button) and `topics_tab.py` | Full integration test: import all modules in one script, fix any import/path errors | Write `main.py` layout: sidebar, `@st.cache_resource` for indexes, `@st.cache_data` for corpus |
| Day 24–25 | Finish all 20 test queries labeled in `test_queries.json` | Add session state management to Streamlit — results persist across tab switches without re-running | Write `graph_tab.py` (PyVis embed) + `analytics_tab.py` (year histogram, topic pie, PageRank vs citation table) |
| Day 26–27 | Run `ablation.py`: 4 system variants × 20 queries × 3 metrics → write `ablation_table.csv` | Run complete demo walkthrough, time each step; add loading spinners where >2 seconds | Write `README.md` with exact run instructions (`pip install`, `streamlit run src/app/main.py`) |
| Day 28 | **✓ `ablation_table.csv` committed with real numbers** | **✓ Full demo runs without crashes; all 5 tabs display correctly** | **✓ Streamlit app committed; README complete** |

---

### Week 5 — Report + Presentation + Buffer (Days 29–35)
**Exit criteria:** Report submitted, demo rehearsed, slides polished.

| | Abubakar | Raahim | Mujtaba |
|---|---|---|---|
| Day 29–30 | Draft Sections 1 (Intro), 2 (Related Work), 11 (Conclusion) | Draft Section 5 (Retrieval Architecture) with system diagram + all equations | Draft Sections 3 (Data), 4 (System Architecture), 10 (UI + screenshots) |
| Day 31–32 | Draft Section 9 (Evaluation) — ablation table + coherence scores + Heaps/Zipf plots | Draft Section 8 (Above and Beyond) — SPECTER2 vs MiniLM comparison, cross-encoder, More Like This | Draft Section 7 (KG) — graph definition, PageRank, collab network; create all presentation diagrams |
| Day 33 | Merge all sections, fix formatting, check IEEE citation format | Review full report, fill technical gaps in Section 5/6 | Draft Section 6 (Topic Modeling) from Abubakar's coherence notebook; finalize presentation slides |
| Day 34 | Proofread, finalize PDF | **Demo rehearsal** — full walkthrough with narration, time it | Slides finalized; rehearse architecture section |
| Day 35 | **📤 Submit report** | **🎯 Demo ready** | **📊 Slides final** |

---

## 8. Report Structure

**Format:** IEEE double-column · Target: 10–14 pages

| Section | Owner | Pages | Course weeks it demonstrates |
|---|---|---|---|
| Abstract | Abubakar | 0.25 | — |
| 1. Introduction | Abubakar | 1.0 | Wk 1 (IR problem framing) |
| 2. Related Work | Abubakar | 1.5 | Wk 1, 9, 12, 13 |
| 3. Data Acquisition | Mujtaba | 0.75 | Wk 4 (indexing), Wk 10 (crawling analogy) |
| 4. System Architecture | Mujtaba | 0.5 | All |
| 5. Hybrid Retrieval | Raahim | 2.0 | Wk 2, 4, 5, 8, 9, 13, 14 |
| 6. Topic Modeling | Abubakar + Raahim | 1.5 | Wk 5, 12, 13 |
| 7. Knowledge Graph | Mujtaba | 1.5 | Wk 10 (link analysis / PageRank) |
| 8. Above and Beyond | Raahim | 1.0 | Wk 13, 14, 16 |
| 9. Evaluation | Abubakar | 2.0 | Wk 7 (MAP, MRR, P@k, coherence) |
| 10. System Demo / UI | Mujtaba | 0.5 | — |
| 11. Conclusion | Abubakar | 0.5 | — |
| References | All | 1.0 | — |

**Tip for Abubakar writing Section 9:** Structure the evaluation subsections as: (a) Corpus Statistics — Heaps' and Zipf's law plots. (b) Retrieval Ablation — table with 4 system variants. (c) Topic Coherence — NMF vs BERTopic c_v scores. (d) Qualitative Case Study — one query, walk through all outputs. This mirrors ISLE Section 4 exactly and gives you a strong story to tell.

**Papers to cite (every one of these should appear):**
- ISLE: arXiv:2512.12760
- BM25S: arXiv:2407.03618
- Cormack et al. RRF, SIGIR 2009
- BERTopic: arXiv:2203.05794
- SPECTER/SPECTER2: arXiv:2004.07180
- MiniLM: arXiv:2002.10957
- OpenAlex: arXiv:2205.01833
- FAISS: arXiv:1702.08734
- Manning, Raghavan & Schütze "Introduction to IR" (textbook — your course book, cite by chapter)
- LDA: Blei et al., JMLR 2003 (to contrast with BERTopic)
- NMF: Lee & Seung, Nature 1999
- PageRank: Page et al., 1999
- UMAP: McInnes et al., 2018
- LitSearch: Ajith et al., EMNLP 2024 (retrieval benchmark for scientific literature)

---

## 9. Presentation Structure

**Target:** 15–20 minutes + 5 min Q&A

| Slide | Speaker | Content |
|---|---|---|
| 1 | Mujtaba | Title, team, one-line problem statement |
| 2 | Mujtaba | The problem with traditional search (Week 1/2 course framing) |
| 3 | Mujtaba | System architecture diagram — full pipeline |
| 4 | Mujtaba | Data: arXiv + OpenAlex, corpus stats (Heaps/Zipf plots) |
| 5–6 | Raahim | Retrieval: BM25 equation → FAISS/SPECTER2 → RRF equation → reranker |
| 7 | Abubakar | Topic modeling: NMF TF-IDF approach vs BERTopic neural pipeline; coherence comparison table |
| 8 | Mujtaba | Knowledge Graph: node types, edge types, PageRank vs raw in-degree (show they differ) |
| 9 | Abubakar | Ablation table — 4 rows, 3 columns, clear improvement trend |
| 10–11 | Raahim | **Live demo** (RTX rig, app pre-loaded) — see demo script below |
| 12 | All | Conclusion + course concepts covered |

**Demo script for Raahim (keep to 4 minutes):**

1. Enter: `"graph neural network node classification"` → hit Search
2. *Results tab* → scroll top-3 results, note RRF scores, click "More Like This" on #1
3. *Topics tab* → show BERTopic pie chart, hover a topic word cloud bar
4. *Knowledge Graph tab* → hover a paper node (shows title, year, citations), hover a topic node
5. *Analytics tab* → point out PageRank vs citation-count top papers side by side: *"PageRank surfaces foundational papers that are heavily cited by other high-impact papers, not just raw citation counts"*
6. *Collab Network tab* → highlight top betweenness-centrality node: *"This author bridges two research communities"*

---

## 10. Above-and-Beyond Features

### Cross-Encoder Reranking (Raahim)
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Where in pipeline:** after RRF top-50, before returning to UI
- **Latency target:** <3 seconds on RTX 3060 for 50 pairs
- **How to show value:** Abubakar's ablation compares P@5 with and without it — row 3 vs row 4 in ablation table

### SPECTER2 vs MiniLM Comparison (Raahim)
- Build a small (10k paper) FAISS index with each model on your rig
- Run same 10 queries on both, compute intra-result cosine similarity
- Expected: SPECTER2 wins on scientific queries, MiniLM is faster
- Goes in Section 8 of report — positions as deliberate design decision, not arbitrary choice

### PageRank on Citation Graph (Mujtaba)
- `nx.pagerank(citation_subgraph, alpha=0.85)` — one line
- The story: *show two ranked lists side by side in the UI*. They differ. Explain why in the demo.
- Cite Page et al. 1999 in the report — this directly parallels Week 10 link analysis

### "More Like This" (Raahim)
- Button on every result card in the UI
- Calls `FAISSIndex.search_by_embedding()` with the paper's precomputed embedding
- Sub-second response since embeddings.npy is preloaded in memory
- No model inference needed at query time

### Pseudo-Relevance Feedback Query Expansion (Abubakar)
- **Phase 1:** WordNet synonyms — 3 synonyms per query token, appended to query string
- **Phase 2:** PRF — take top-3 BM25 results, extract top-10 TF-IDF terms, append to query, re-run search
- **Report framing:** "Query expansion implements Pseudo-Relevance Feedback (Manning et al. Ch. 9), a core technique from Week 8 of the course"
- **Ablation:** Row 5 in ablation table: RRF + reranker + query expansion

### Author Collaboration Network (Mujtaba)
- Separate graph from main KG — its own Streamlit tab
- Nodes = authors, edges = co-authored paper (weight = number of shared papers)
- Node size in PyVis = total citation count of all their papers
- Betweenness centrality highlights bridge researchers between communities

---

## 11. Ablation Study (Abubakar owns this entirely)

**20 test queries** — assessed by all three team members independently; take majority vote for relevance. Binary judgment: relevant (1) / not relevant (0) for top-5 and top-10 results per query.

**System variants to evaluate:**

| Variant | Components |
|---|---|
| 1 — BM25 only | BM25 search, no expansion, no reranking |
| 2 — Dense only | SPECTER2 + FAISS, no BM25, no reranking |
| 3 — Hybrid (RRF) | BM25 + FAISS fused via RRF |
| 4 — Hybrid + Reranker | Variant 3 + cross-encoder reranking |
| 5 — Full system | Variant 4 + query expansion |

**Metrics:**

```
P@5  = (relevant results in top 5) / 5
P@10 = (relevant results in top 10) / 10
MRR  = 1 / rank of first relevant result
MAP  = mean of precision at each relevant result position
```

**Expected table shape (fill with real numbers):**

| System | P@5 | P@10 | MRR | MAP |
|---|---|---|---|---|
| BM25 only | | | | |
| Dense only | | | | |
| Hybrid (RRF) | | | | |
| Hybrid + Reranker | | | | |
| Full system | | | | |

Expected trend: each row should improve on the previous. If it doesn't for some metric, explain why in the report — that's a legitimate finding.

---

## 12. Test Queries for Ablation

```json
[
  {"id": 1,  "query": "transformer self-attention mechanism NLP",          "type": "specific_method"},
  {"id": 2,  "query": "graph neural network node classification",          "type": "combined_topic"},
  {"id": 3,  "query": "large language model instruction fine-tuning",      "type": "broad_topic"},
  {"id": 4,  "query": "BERT pretraining masked language model",            "type": "named_model"},
  {"id": 5,  "query": "reinforcement learning policy gradient optimization","type": "specific_method"},
  {"id": 6,  "query": "federated learning privacy data",                   "type": "combined_topic"},
  {"id": 7,  "query": "neural machine translation low resource languages", "type": "combined_topic"},
  {"id": 8,  "query": "image segmentation semantic pixel labeling",        "type": "combined_topic"},
  {"id": 9,  "query": "knowledge graph embedding link prediction",         "type": "combined_topic"},
  {"id": 10, "query": "contrastive learning self-supervised visual",       "type": "specific_method"},
  {"id": 11, "query": "adversarial attack robustness defense",             "type": "combined_topic"},
  {"id": 12, "query": "object detection YOLO real-time speed",             "type": "named_model"},
  {"id": 13, "query": "question answering reading comprehension extraction","type": "task"},
  {"id": 14, "query": "generative adversarial network image synthesis",    "type": "specific_method"},
  {"id": 15, "query": "text classification sentiment analysis BERT",       "type": "task"},
  {"id": 16, "query": "speech recognition end-to-end sequence model",      "type": "combined_topic"},
  {"id": 17, "query": "recommendation system collaborative filtering",     "type": "combined_topic"},
  {"id": 18, "query": "differential privacy noise mechanism deep learning","type": "combined_topic"},
  {"id": 19, "query": "meta-learning few-shot generalization",             "type": "specific_method"},
  {"id": 20, "query": "vision transformer image recognition ViT",          "type": "named_model"}
]
```

---

## 13. Heaps' and Zipf's Law (Abubakar — corpus_stats.py)

This is a 30-minute task that directly demonstrates Week 4 course knowledge in your evaluation section. Add it to your notebook.

```python
# evaluation/corpus_stats.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter

df = pd.read_parquet("data/processed/corpus.parquet")
all_tokens = []
vocab_sizes = []
cumulative = []

for i, text in enumerate(df["text"]):
    tokens = text.lower().split()
    all_tokens.extend(tokens)
    if i % 5000 == 0:
        vocab_sizes.append(len(set(all_tokens)))
        cumulative.append(len(all_tokens))

# ── Heaps' Law plot: vocabulary growth ──────────────────
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=cumulative, y=vocab_sizes, mode="lines",
                           name="Observed", line=dict(color="#4fc3f7")))
fig1.update_layout(title="Heaps' Law — Vocabulary Growth",
                   xaxis_title="Tokens processed", yaxis_title="Vocabulary size")
fig1.write_html("evaluation/results/heaps_law.html")

# ── Zipf's Law plot: rank vs frequency ──────────────────
freq = Counter(all_tokens)
sorted_freq = sorted(freq.values(), reverse=True)[:5000]
ranks = list(range(1, len(sorted_freq) + 1))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=np.log(ranks), y=np.log(sorted_freq),
                           mode="lines", name="Observed",
                           line=dict(color="#ff8a65")))
fig2.update_layout(title="Zipf's Law — Term Frequency Distribution (log-log)",
                   xaxis_title="log(Rank)", yaxis_title="log(Frequency)")
fig2.write_html("evaluation/results/zipf_law.html")

print(f"Corpus: {len(df)} papers | {len(all_tokens):,} tokens | {len(freq):,} unique terms")
```

Use these plots in Section 9.a of the report under "Corpus Statistics."

---

## 14. Risk Register

| Risk | Likelihood | Owner to resolve | Mitigation |
|---|---|---|---|
| Colab disconnects mid-job | High | Abubakar | Keep all jobs under 90 min; save to Drive every 30 min; BM25 build takes <5 min so no risk |
| FAISS build OOM on RTX 3060 | Low | Raahim | 100k × 768-dim = 300 MB VRAM — well within 12 GB; only risk is if batch encoding eats VRAM simultaneously |
| OpenAlex API changes / rate limits | Medium | Mujtaba | Cache all API responses locally as JSON immediately; 0.05s sleep between requests; free API key required since Feb 2026 |
| BERTopic produces too many micro-topics | Medium | Raahim | Raise `min_cluster_size` to 20–30; use `nr_topics="auto"` which merges similar topics post-hoc |
| Relevance labeling bottleneck | Medium | Abubakar | Start on Day 12; do 4–5 queries per day, not all at once; Raahim and Mujtaba each check 5 queries independently |
| Integration breaks in Week 4 | Medium | Raahim | Contracts are strict for exactly this reason; each module must pass a simple smoke test before Week 4 |
| Report sections are written in isolation and don't flow | Medium | Abubakar (merges) | Share a running Google Doc from Week 3; don't wait until Week 5 to put sections together |

---

## 15. Communication Rules

1. **Async standup every evening:** Message in group chat — done / blocked / next. Two lines max.
2. **Blocking dependency → immediate message:** Don't sit on a blocker for 24 hours.
3. **Large files go to GitHub Releases**, not the git tree: `corpus.parquet`,
   `embeddings.npy`, FAISS/BM25 index files. Release assets allow up to 2 GB
   per file and stay private to collaborators on a private repo. Publish with
   `gh release create data-v1 corpus.parquet embeddings.npy --notes "..."` and
   fetch with `gh release download data-v1`. (Google Drive is an acceptable
   fallback.) The git tree itself never carries files >50 MB.
4. **Raahim's machine = integration environment:** Code must run on Linux/Windows with CUDA without modification. Mujtaba (macOS) should test imports carefully — some libraries behave differently on Apple Silicon.
5. **No hardcoded paths:** Everything goes through `config.py`. This is the single most common cause of "works on my machine" integration failures.

---

*This is the authoritative project reference. When anything changes — scope, timeline, interface contracts — update this document and notify the team. Last updated by Raahim.*
