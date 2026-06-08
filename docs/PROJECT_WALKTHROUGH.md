# ISLE-IR — Complete Project Walkthrough

> **Audience:** Someone with medium AI/ML knowledge seeing this project for the first time.
> Read this end-to-end before your evaluation. It covers purpose, architecture, data flow, every pipeline stage, every UI tab, and every sidebar toggle.

---

## 1. What Is This Project?

**ISLE-IR** is a **hybrid scientific-literature search system**. Given a natural-language query like *"transformer attention mechanism NLP"*, it:

1. Retrieves the most relevant research papers from a 100,000-paper arXiv corpus
2. Clusters those results into discoverable **topics** (e.g. "self-attention variants", "efficient transformers")
3. Builds an interactive **knowledge graph** connecting papers, authors, institutions, countries, topics, and years
4. Presents everything in a web UI with multiple exploration tabs

It is based on the **ISLE paper** (arXiv:2512.12760) — a system that fuses lexical and semantic retrieval, then enriches results with topic modeling and knowledge-graph exploration.

**The core idea:** Traditional academic search (like Google Scholar) gives you a flat list of papers. ISLE-IR goes further — it *understands* the retrieved set by finding topics, mapping relationships between papers/authors/institutions, and letting you explore interactively.

---

## 2. The Big Picture — Pipeline Architecture

```
User types a query
        │
        ▼
┌──────────────────────────────┐
│  1. QUERY EXPANSION          │   WordNet synonyms + Pseudo-Relevance Feedback
│     (optional)               │   Makes the query richer for better recall
└──────────────┬───────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌───────────┐    ┌──────────────┐
│  2. BM25  │    │  3. SPECTER2 │    Two independent retrievers
│  Lexical  │    │  Dense       │    one matches keywords, the other
│  Search   │    │  Search      │    matches meaning
└─────┬─────┘    └──────┬───────┘
      │                 │
      └────────┬────────┘
               ▼
┌──────────────────────────────┐
│  4. RECIPROCAL RANK FUSION   │   Merges the two ranked lists into one
│     (RRF)                    │   using position-based scoring
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  5. CROSS-ENCODER RERANKER   │   Re-scores top results with a neural
│     (optional)               │   model that reads query+abstract together
└──────────────┬───────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌───────────┐    ┌──────────────┐
│  6. TOPIC │    │  7. KNOWLEDGE│    Two parallel enrichment stages
│  MODELING │    │  GRAPH      │    that transform raw results into
│  BERTopic │    │  BUILDING   │    structured, explorable knowledge
└─────┬─────┘    └──────┬───────┘
      │                 │
      └────────┬────────┘
               ▼
┌──────────────────────────────┐
│  8. STREAMLIT WEB UI         │   6 tabs: Results, Topics, Knowledge Graph,
│                              │   Analytics, Collaboration Network, Semantic Map
└──────────────────────────────┘
```

---

## 3. Phase-by-Phase Breakdown

### Phase 0 — Environment Setup

**What:** Install dependencies, configure GPU vs CPU, decide on embedding model.

**Key decision — SPECTER2 vs MiniLM:**
- **SPECTER2** (`allenai/specter2_base`, 768-dim) — a model specifically trained on citation graphs for scientific papers. Chosen because this is a scientific-literature system.
- MiniLM (`all-MiniLM-L6-v2`, 384-dim) — a general-purpose model kept only as a comparison baseline.
- The choice lives in `src/app/config.py` — one constant, propagated everywhere.

**Package manager:** `uv` — fast, handles GPU/CPU variants via `--extra gpu` or `--extra cpu`. FAISS-GPU is Linux-only (NVIDIA CUDA), so Mac/Intel machines use FAISS-CPU.

---

### Phase 1 — Data Acquisition (100,000 papers)

**Goal:** Build `corpus.parquet` — the master dataset every downstream module reads.

**Sources:**
1. **arXiv Kaggle snapshot** — 2.4M papers in a JSON file. We filter to `cs.*` categories (cs.AI, cs.CL, cs.CV, cs.IR, cs.LG, etc.) and cap at 100,000 papers.
2. **OpenAlex API** — enriches each paper with citation counts, author institutions, countries, and reference lists. Free API key required since Feb 2026.

**The output — `corpus.parquet`:**

| Column | Type | Example | Source |
|--------|------|---------|--------|
| `arxiv_id` | str | `"2301.07041"` | arXiv |
| `title` | str | `"Attention Is All You Need"` | arXiv |
| `abstract` | str | `"The dominant sequence transduction models..."` | arXiv |
| `text` | str | title + " " + abstract (pre-joined for indexing) | arXiv |
| `authors` | list[str] | `["Ashish Vaswani", "Noam Shazeer"]` | arXiv |
| `year` | int | `2017` | arXiv |
| `categories` | list[str] | `["cs.CL", "cs.LG"]` | arXiv |
| `doi` | str | `"10.5555/3295222.3295349"` | arXiv |
| `cited_by_count` | int | `15234` | OpenAlex |
| `institutions` | list[str] | `["Google Brain", "University of Toronto"]` | OpenAlex |
| `countries` | list[str] | `["US", "CA"]` | OpenAlex |
| `references` | list[str] | `["W2964149567", ...]` | OpenAlex |

**Why Parquet?** It's a columnar format — much faster to read specific columns than CSV. A 100k-paper corpus is ~120MB in Parquet vs ~300MB in CSV. It also preserves data types (lists stay as lists, not strings).

**Files involved:**
- `data/raw/arxiv-metadata-oai-snapshot.json` → download from Kaggle
- `src/ingestion/arxiv_loader.py` → parses, filters, outputs parquet
- `src/ingestion/openalex_enrich.py` → adds citation counts, institutions, references
- `data/processed/corpus.parquet` → the final output (~120MB)

---

### Phase 2 — Hybrid Retrieval Engine (the core)

This is the heart of the system — how we go from a query to a ranked list of papers.

#### Stage 2a: BM25 Lexical Search

**What it does:** Classic keyword matching. Tokenizes the query, stems words (e.g. "running" → "run"), removes stopwords, then scores documents using the **BM25 formula** — a probabilistic ranking function that considers term frequency, document length, and inverse document frequency.

**Tech:** `bm25s` library with `PyStemmer` (Porter stemmer). ~500× faster than the older `rank_bm25`.

**BM25 formula:**
```
BM25(D, Q) = Σ IDF(qi) · (f(qi, D) · (k1 + 1)) / (f(qi, D) + k1 · (1 - b + b · |D|/avgdl))
```
- `k1 = 1.5` — controls how much term frequency saturates
- `b = 0.75` — controls document-length normalization
- These are from the ISLE paper (Section 3.2.1)

**Returns:** Top-100 papers as `[(arxiv_id, bm25_score), ...]`

**Strengths:** Exact keyword match. Great for named entities ("BERT", "YOLO"), specific methods ("policy gradient"), precise queries.

**Weaknesses:** Cannot handle synonyms or semantic similarity. "neural network" won't match "deep learning".

**Index files:** `data/indexes/bm25/retriever.joblib` + `paper_ids.joblib` (~60MB total)

#### Stage 2b: SPECTER2 Dense Semantic Search

**What it does:** Encodes every paper into a 768-dimensional vector using SPECTER2 — a transformer model trained on citation graphs to understand scientific text. At query time, encodes the query into the same 768D space, then finds nearest neighbors using cosine similarity.

**Why SPECTER2 (not a general model):** SPECTER2 was trained with a **citation-aware contrastive objective** — papers that cite each other are pushed closer in vector space. This makes it especially good at matching scientific queries to scientific papers.

**How it loads:** SPECTER2 uses the `adapters` library, not plain `SentenceTransformer`. It loads a base model + a "proximity" adapter that was fine-tuned for paper similarity.

**FAISS IVFFlat index:**
- Splits the 100k vectors into ~1024 clusters (Voronoi cells)
- At query time, probes the 64 nearest clusters → returns top-100
- Much faster than brute-force search, with ~99% recall
- Built on GPU (Raahim's RTX 3060), serves on CPU

**Returns:** Top-100 papers as `[(arxiv_id, cosine_similarity), ...]`

**Strengths:** Semantic matching — "how do transformers handle long sequences" matches papers about "efficient attention mechanisms for extended context".

**Weaknesses:** Might miss papers with exact terminology that differs from the query's phrasing.

**Index files:** `data/indexes/faiss/vectors.index` (~300MB) + `paper_ids.json` + `embeddings.npy` (~300MB)

**The embeddings file (`embeddings.npy`):** A NumPy array of shape (100000, 768) — the precomputed vector for every paper. This file is shared across modules:
- FAISS search uses it for the index
- BERTopic uses it to avoid re-encoding documents
- The Streamlit app uses it for "More Like This" and the Semantic Map

#### Stage 2c: Reciprocal Rank Fusion (RRF)

**What it does:** Merges the two top-100 lists (BM25 and SPECTER2) into a single ranked list.

**Why not just average the scores?** BM25 scores are unbounded (can range from 0 to 100+), while cosine similarity is between 0 and 1. Even with normalization, the distributions are mismatched — one method would dominate.

**RRF formula (Cormack et al., SIGIR 2009):**
```
RRF_score(d) = Σ  1 / (k + rank(d))    over all rankers
```
Where `k = 60` (the standard constant from the paper).

**Example:**
- Paper X is rank #2 in BM25 and rank #5 in SPECTER2
- RRF_score(X) = 1/(60+2) + 1/(60+5) = 1/62 + 1/65 = 0.0161 + 0.0154 = 0.0315

**Why it works:** It uses *position* not *score*. A paper that ranks high in both lists gets a strong combined score, regardless of the raw score magnitudes.

**Returns:** Top-50 papers as `[(arxiv_id, rrf_score), ...]`

#### Stage 2d: Cross-Encoder Reranker (optional)

**What it does:** Takes the top-50 RRF results and re-scores each (query, paper) pair using a cross-encoder — a transformer that reads both texts together (not separately like the bi-encoder used for FAISS).

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — trained on MS MARCO passage ranking.

**Why it improves results:** The bi-encoder (SPECTER2) encodes query and paper *independently*, which is fast but misses fine-grained interactions. The cross-encoder reads them *together* — it can see that the query asks about "low-resource languages" and the paper specifically addresses this, even if the phrasing differs.

**Returns:** Top-20 papers, re-ordered by cross-encoder relevance score.

**Latency:** ~0.4 seconds for 50 query-paper pairs on GPU. On CPU, ~2-3 seconds.

#### Stage 2e: Query Expansion (optional)

**What it does:** Broadens the query before search to improve recall.

**Two techniques combined:**
1. **WordNet synonyms** — for each word in the query, adds up to 3 synonyms from the WordNet lexical database. E.g. "classification" → adds "categorization", "sorting".
2. **Pseudo-Relevance Feedback (PRF)** — runs an initial BM25 search, takes the top-3 results, extracts the top-10 TF-IDF terms from those results, and appends them to the query. The idea: if the top results mention "attention mechanism" and "self-attention", those terms should be added to the query.

**Course tie-in:** PRF is a direct implementation of **Rocchio's relevance feedback** (Week 8 of the IR course).

#### The Full Pipeline Function

```python
def full_retrieval_pipeline(query, bm25_index, faiss_index, papers_lookup,
                            expand_query=True, rerank=True):
    # 1. Expand query (optional)
    # 2. BM25 search → top-100
    # 3. SPECTER2 search → top-100
    # 4. RRF fusion → top-50
    # 5. Cross-encoder rerank → top-20
    return [(arxiv_id, final_score), ...]
```

All parameters (k1, b, top_k, RRF k, etc.) come from `src/app/config.py`.

---

### Phase 3 — Topic Modeling

**What it does:** Takes the retrieved set of papers and discovers what *topics* or *themes* they cover.

#### BERTopic (neural pipeline, GPU-accelerated)

**The algorithm:**
1. **Encode** documents into embeddings (reuse precomputed SPECTER2 embeddings — no re-encoding!)
2. **UMAP** — reduces 768-dim embeddings to 5 dimensions for clustering. UMAP preserves both local and global structure better than PCA.
3. **HDBSCAN** — density-based clustering on the 5-dim space. Finds clusters of varying shapes and sizes without specifying k. Documents that don't fit any cluster are labeled -1 (outliers).
4. **c-TF-IDF** — for each cluster, finds the most representative words. Unlike regular TF-IDF which operates per document, c-TF-IDF operates per cluster — it finds words that are distinctive to that topic relative to all other topics.
5. **auto-merge** — `nr_topics="auto"` merges highly similar topics post-hoc (e.g. two clusters that both represent "transformer variants" get merged).

**Result:** 657 topics discovered across the full corpus, with C_V coherence score of 0.6623. On a query's result set (20-50 papers), it produces ~3-8 topics.

**What each topic has:**
```python
{
    "topic_id": 5,
    "keywords": ["attention", "transformer", "self-attention", "vision", "efficient"],
    "doc_count": 12,           # papers assigned to this topic
    "label": "attention_transformer_vision"   # auto-generated
}
```

**Each paper gets:**
```python
doc_topics[i] = 5      # assigned topic (or -1 for outlier)
doc_probs[i] = 0.87    # confidence of assignment (0-1)
```

#### NMF (CPU baseline)

**What it does:** Non-negative Matrix Factorization on a TF-IDF matrix. Pure linear algebra — no neural networks.

**How it works:**
1. Build a TF-IDF matrix from document texts (documents × terms)
2. Factorize it into two non-negative matrices: W (documents × topics) and H (topics × terms)
3. Each document's topic = the column in W with the highest value

**Parameter tuning:** Sweeps k from 5 to 20, picks the k with best C_V coherence score using gensim.

**Why both?** NMF is a course-concept baseline (Week 5: Vector Space Model / TF-IDF, Week 12: text clustering). BERTopic is the above-and-beyond neural approach. Comparing them demonstrates both classical and modern techniques.

---

### Phase 4 — Knowledge Graph Construction

**What it does:** Builds a heterogeneous directed graph connecting papers, authors, institutions, countries, topics, and years — capturing the *relationships* between entities in the retrieved set.

#### Graph Schema (6 node types, 6 edge types)

| Node Type | Example ID | Color | Size | Represents |
|-----------|-----------|-------|------|------------|
| `paper` | `"2301.07041"` | Blue | 10 | A research paper |
| `author` | `"author::ashish vaswani"` | Green | 8 | A person |
| `institution` | `"inst::google brain"` | Orange | 8 | A research org |
| `country` | `"country::US"` | Purple | 12 | A country code |
| `topic` | `"topic::5"` | Red/Orange | 15 | A discovered topic |
| `year` | `"year::2017"` | Gray | 6 | Publication year |

| Edge Type | Direction | Meaning |
|-----------|-----------|---------|
| `authored` | author → paper | "This author wrote this paper" |
| `affiliated` | paper → institution | "This paper's authors are from this institution" |
| `from_country` | paper → country | "This paper has authors from this country" |
| `published_in` | paper → year | "This paper was published in this year" |
| `belongs_to` | paper → topic | "This paper belongs to this topic" (weight = confidence) |
| `cites` | paper → paper | "This paper cites that paper" (within the retrieved set only) |

#### How it's built

For every paper in the retrieved set:
1. Add a **paper node** with metadata (title, abstract, year, citations, topic assignment)
2. For each author → add an **author node** + `authored` edge
3. For each institution → add an **institution node** + `affiliated` edge
4. For each country → add a **country node** + `from_country` edge
5. Add a **year node** + `published_in` edge
6. For the assigned topic → add a **topic node** + `belongs_to` edge (with probability weight)
7. For each cited paper that's also in the retrieved set → add a `cites` edge (intra-corpus citations only)

#### Graph Analytics

- **PageRank** on the citation subgraph — treats citations like web links. A paper cited by other highly-cited papers gets a higher PageRank than one cited by many low-impact papers. This surfaces *foundational* papers, not just popular ones.
- **Entity impact** — sums the citation counts of all papers associated with an author/institution/country. This is ISLE Eq. 20.
- **Top papers** — ranked by citation in-degree vs. PageRank (they differ — demonstrating why link analysis matters).

#### Collaboration Network

A **separate graph** from the main KG:
- Nodes = **authors**
- Edges = **co-authored papers** (edge weight = number of shared papers)
- Node size = total citations of all their papers in the set
- **Betweenness centrality** identifies "bridge researchers" who connect otherwise separate research communities

**Visualization:** PyVis generates interactive HTML — drag nodes, zoom, hover for details. Force-directed layout positions connected nodes close together.

---

### Phase 5 — Streamlit Frontend

**What it does:** A web application (runs at `http://localhost:8501`) that ties everything together.

**Tech:** Streamlit — a Python framework that turns data scripts into web apps. Every user interaction (button click, tab switch) re-runs the script from top to bottom, so we use:
- `@st.cache_resource` — loads models and indexes once, reuses across interactions
- `@st.cache_data` — caches DataFrames and arrays
- `st.session_state` — persists search results across tab switches without re-running the pipeline

#### The Sidebar Controls

```
┌─────────────────────────────────────┐
│  Search                             │
│  ┌─────────────────────────────┐    │
│  │ Query: [_________________]  │    │
│  └─────────────────────────────┘    │
│                                     │
│  [✓] Query expansion                │
│  [✓] Cross-encoder rerank           │
│  [ ] Freshness boost                │
│      Freshness λ [====●====] 0.30   │
│  [ ] MMR diversity                  │
│      MMR λ [====●====] 0.50        │
│                                     │
│  ──────────────                     │
│  Rocchio Feedback                   │
│  Mark results relevant / not        │
│  relevant, then refine.             │
│  [🔄 Refine Search]                │
│                                     │
│  [🔍 Search]  (primary button)     │
└─────────────────────────────────────┘
```

**Each toggle explained:**

| Toggle | What it does | Technical detail |
|--------|-------------|------------------|
| **Query expansion** | Enriches the query with synonyms and feedback terms before searching | WordNet synonyms + Pseudo-Relevance Feedback (top-3 BM25 results → top-10 TF-IDF terms appended) |
| **Cross-encoder rerank** | Re-scores top-50 RRF results with a neural model for better precision | `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, abstract) pair jointly |
| **Freshness boost** | Gives newer papers a score bonus, especially in fast-moving topics | Exponential decay: `score × (1 + λ · e^(-decay × years_old))`. Decay rate is per-topic (LLM topics decay fast, math topics decay slow). λ controls the strength. |
| **MMR diversity** | Balances between relevance and diversity in results | Maximal Marginal Relevance penalizes papers too similar to already-selected ones. λ=1 means pure relevance, λ=0 means pure diversity. |
| **Rocchio Feedback** | Re-runs the search with a modified query vector based on user feedback | User marks papers 👍 (relevant) or 👎 (not relevant). The system shifts the query embedding toward relevant papers and away from non-relevant ones using the Rocchio formula. |

#### The Tabs

When you hit Search, the pipeline runs and results appear across 6 tabs:

---

**📄 Results Tab**

Shows the top-20 papers as cards. Each card has:
- **Rank number + title** (title links to arxiv.org)
- **Metadata row:** authors, year, categories
- **Expandable details:** RRF score, abstract (truncated), citation count
- **Rocchio buttons:** 👍 / 👎 to mark relevant/non-relevant for feedback
- **Export section:** Download results as BibTeX (for citation managers) or CSV (for spreadsheets)

---

**🧩 Topics Tab**

Visualizes the topic structure of the retrieved papers:
- **Topic distribution pie chart** — shows how many papers belong to each topic (Plotly interactive chart — hover for details)
- **Per-topic keyword bar charts** — for each topic, shows the top words that define it
- **Documents-per-topic table** — which papers are in which topic

---

**🕸 Knowledge Graph Tab**

Interactive graph visualization:
- **PyVis embed** — a force-directed graph rendered as interactive HTML. Drag nodes, zoom, hover for tooltips showing title/year/citations.
- **Graph metrics** — node count, edge count
- **Citation Path Explorer** — pick two papers from dropdowns, click "Find connection", and the system finds the shortest citation path between them (max 4 hops). If a path exists, it renders a highlighted subgraph showing the citation chain in gold.

How path finding works: Builds a subgraph of only `cites` edges, then runs BFS (NetworkX `shortest_path`) from source to target.

---

**📊 Analytics Tab**

Statistical overview of the result set:
- **Graph summary metrics** — nodes, edges, density, per-type breakdown
- **Publication year histogram** — distribution of papers by year (are results mostly recent or foundational?)
- **Top cited papers table** — papers ranked by citation count within the retrieved set
- **PageRank vs citations comparison** — shows that PageRank surfaces different papers than raw citation count, demonstrating the value of link analysis

---

**👥 Collaboration Network Tab**

A separate co-authorship graph:
- **Top bridge researchers** — ranked by betweenness centrality (authors who bridge different research communities)
- **PyVis visualization** — nodes are authors (sized by total citations), edges are co-authorships (weighted by shared paper count). Force-directed layout clusters frequent collaborators together.

What betweenness centrality measures: An author has high betweenness if the shortest paths between many pairs of other authors pass through them. They are "bridges" between otherwise disconnected research groups.

---

**🗺 Semantic Map Tab**

A 2D UMAP projection of the retrieved papers:
- Each point is a paper
- Color represents its assigned topic
- Papers close together have similar SPECTER2 embeddings
- Lets you visually see how topics cluster in embedding space

---

### Phase 6 — Evaluation

#### Ablation Study

Runs 5 system variants over 20 labeled test queries:

| Variant | Components |
|---------|-----------|
| BM25 only | Just keyword search |
| Dense only | Just SPECTER2 + FAISS |
| Hybrid (RRF) | BM25 + FAISS fused |
| Hybrid + Reranker | Above + cross-encoder |
| Full system | Above + query expansion |

**Metrics:**
- **P@5** — how many of the top 5 are relevant
- **P@10** — how many of the top 10 are relevant
- **MRR** — reciprocal rank of the first relevant result
- **MAP** — mean of precision at each relevant result position

#### Corpus Statistics

- **Heaps' Law plot** — vocabulary growth as more documents are added. Shows that vocabulary grows sub-linearly (∝ N^β), confirming the corpus follows expected statistical properties.
- **Zipf's Law plot** — term frequency vs rank (log-log). The classic power-law distribution where a few terms appear very frequently and most terms appear rarely.

#### Topic Coherence

- C_V coherence score compares NMF vs BERTopic topic quality
- BERTopic achieves C_V = 0.6623 with 657 topics

---

## 4. Intermediate Files and Data Flow

```
data/raw/arxiv-metadata-oai-snapshot.json     (2.4M papers, downloaded from Kaggle)
        │
        ▼  arxiv_loader.py  (filter cs.*, parse metadata)
data/processed/corpus.parquet                   (100k papers, ~120MB)
        │
        ├──► bm25_index.py  → data/indexes/bm25/               (~60MB)
        │                     ├── retriever.joblib              (BM25 inverted index)
        │                     └── paper_ids.joblib              (ID ordering)
        │
        ├──► faiss_index.py → data/indexes/faiss/              (~600MB)
        │                     ├── vectors.index                 (FAISS IVFFlat, 768-dim)
        │                     ├── paper_ids.json                (ID ordering)
        │                     └── embeddings.npy                (100k × 768 float32)
        │
        └──► bertopic_model.py → data/indexes/bertopic/
                                  └── model                      (pre-fit BERTopic)
```

At query time:
```
query + corpus.parquet + bm25 index + faiss index
        │
        ▼  pipeline.py
top-20 papers [(id, score)]
        │
        ├──► bertopic_model.py (uses embeddings.npy)
        │   → topic_info, doc_topics, doc_probs
        │
        ├──► builder.py
        │   → knowledge graph (NetworkX DiGraph, in-memory)
        │
        └──► collab_network.py
            → co-authorship graph (NetworkX Graph, in-memory)
```

Nothing is written to disk at query time — everything is built in-memory and displayed in the Streamlit UI.

---

## 5. Key Technologies Summary

| Layer | Technology | What it does |
|-------|-----------|--------------|
| **Data** | `pandas` + `pyarrow` | Load/save corpus as Parquet — columnar, typed, compressed |
| **Data sources** | `arxiv` + `pyalex` | arXiv metadata + OpenAlex enrichment (citations, affiliations) |
| **Lexical search** | `bm25s` + `PyStemmer` | BM25 with stemming — keyword matching, 500× faster than rank_bm25 |
| **Dense search** | `SPECTER2` via `adapters` + `torch` | 768-dim scientific paper embeddings, citation-aware |
| **Vector index** | `faiss` (IVFFlat) | Fast approximate nearest-neighbor search over 100k vectors |
| **Rank fusion** | custom (`rrf.py`) | Reciprocal Rank Fusion — position-based merging of BM25 + FAISS |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Neural re-scoring of (query, abstract) pairs |
| **Query expansion** | `nltk` (WordNet) + TF-IDF | Synonym addition + pseudo-relevance feedback |
| **Neural topics** | `bertopic` + `umap-learn` + `hdbscan` | UMAP → HDBSCAN → c-TF-IDF pipeline |
| **Classical topics** | `scikit-learn` NMF + TF-IDF | Non-negative matrix factorization baseline |
| **Topic evaluation** | `gensim` CoherenceModel | C_V and U_MASS coherence scores |
| **Knowledge graph** | `networkx` | Heterogeneous directed graph with 6 node types |
| **Link analysis** | NetworkX `pagerank` | PageRank on citation subgraph |
| **Graph viz** | `pyvis` | Interactive HTML graph rendering |
| **Frontend** | `streamlit` | Web UI with tabs, caching, session state |
| **Charts** | `plotly` | Interactive histograms, pie charts, bar charts |
| **Linting** | `ruff` | Python linter (replaces flake8 + isort + black) |
| **Testing** | `pytest` | Unit tests for RRF, freshness, export, graph builder |
| **Package mgmt** | `uv` | Fast dependency resolver with GPU/CPU variant support |

---

## 6. How to Answer Evaluation Questions

### "What problem does this solve?"

Scientific literature search is overwhelmed by volume (arXiv alone gets 20,000+ papers/month). Keyword search misses semantically related work. Pure semantic search misses exact terminology. ISLE-IR combines both (hybrid retrieval), then helps you *understand* the results through topics and knowledge graphs — not just a flat list.

### "Why hybrid retrieval?"

BM25 and dense retrieval have complementary strengths:
- BM25 excels at exact term matching, named entities, specific methods
- Dense excels at semantic similarity, paraphrases, conceptual matching
- RRF combines them without score-normalization headaches

### "Why SPECTER2 over a general model?"

SPECTER2 was trained on citation graphs — papers that cite each other are pulled closer in embedding space. For a scientific-literature search system, this is strictly better than a general-purpose model like MiniLM that was trained on NLI/STS data (news, conversations, etc.).

### "What does the knowledge graph add?"

A flat result list tells you *what* papers exist. The knowledge graph tells you *how they relate*:
- Which authors collaborate across papers
- Which institutions are prominent in this research area
- Which papers cite each other (and which are foundational via PageRank)
- Which topics bridge different research communities

### "What course concepts does this cover?"

| Week | Course Topic | Where in ISLE-IR |
|------|-------------|-----------------|
| 2 | Document processing, stemming | BM25 tokenization with Porter stemmer |
| 4 | Index construction, Heaps'/Zipf's Law | BM25 inverted index + corpus statistics |
| 5 | Vector Space Model / TF-IDF | NMF pipeline uses TF-IDF document representation |
| 7 | Evaluation (MAP, MRR, P@k) | Full ablation study with 5 system variants |
| 8 | Relevance feedback | Pseudo-relevance feedback in query expansion + Rocchio feedback in UI |
| 9 | Probabilistic model / BM25 | BM25 retrieval with k1=1.5, b=0.75 |
| 10 | Link analysis | PageRank on citation graph |
| 12 | Text clustering | BERTopic + NMF topic modeling |
| 13 | Neural IR | SPECTER2 dense retrieval + cross-encoder reranker |
| 14 | Word representations | SPECTER2 embeddings = learned scientific paper representations |
