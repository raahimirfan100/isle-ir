# Hybrid IR System — Deep Implementation Plan
### Hybrid Information Retrieval with Topic Modeling & Knowledge Graph-based Exploration
**Team:** Abubakar Ahmed (23k-0801) · Raahim Irfan (23k-0837) · Mujtaba Kamran (23k-0715)
**Based on:** ISLE — arXiv:2512.12760v1

> **Canonical technical guide** — phase-by-phase implementation code, library
> choices and known pitfalls. Project management (roles, interface contracts,
> timeline, evaluation) lives in [`SSOT.md`](SSOT.md). The two are complementary.
>
> **Embedding model — decided: SPECTER2** (`allenai/specter2_base`, 768-dim),
> the citation-trained scientific-document model that matches `SSOT.md`
> Contract C. The code samples below use `all-MiniLM-L6-v2` as a simpler
> illustration — treat the model as config-driven (`src/app/config.py` is the
> single knob); MiniLM stays as the baseline for the SPECTER2-vs-MiniLM study
> (SSOT §10). Module filenames here are illustrative; the repo follows
> `SSOT.md` §4 (e.g. `arxiv_loader.py`, not `arxiv_fetch.py`).

---

## How to Read This Document

Every section below is grounded in a specific library, GitHub repo, paper, or verified implementation pattern — not generic advice. Code blocks are production-ready starting points, not pseudocode. Pitfalls are drawn from actual GitHub issues and community discussions.

---

## Hardware & Scope

**Compute — Raahim's rig (the primary build / integration node).**

| Component | Spec | Relevance |
|---|---|---|
| GPU | RTX 3060 · 12 GB GDDR6 · CUDA 8.6 | FAISS-GPU indexes, batch encoding (~2000 docs/s with MiniLM), GPU UMAP+HDBSCAN via RAPIDS cuML |
| CPU | Ryzen 7 7700 · 8C/16T · Zen 4 | parallel workers for ingestion, tokenisation, NMF |
| RAM | 64 GB DDR5 | full embedding matrix stays resident (~1.5 GB for 1M×384 float32) |
| Storage | Samsung 990 Pro · PCIe 4.0 | sub-second cold load of the embedding index |

**Key VRAM insight:** MiniLM-L6-v2 uses ~500 MB on the GPU, leaving ~11.5 GB
for FAISS. A 500k-paper / 384-dim index needs only ~750 MB VRAM — corpus size
is not VRAM-bound on this hardware; it is bound by encoding time and the
project timeline.

**Corpus scope.** arXiv `cs.*` categories, roughly 2018–2024. Target
**~100–150k papers** for the committed timeline (`SSOT.md` §7 sets a 50k
floor). The hardware comfortably supports **400–600k** as a stretch if time
allows — treat the final size as a Phase-1 team decision, set via `max_records`
in the arXiv loader. Sources: arXiv (Kaggle snapshot / API) + OpenAlex (free
API key required since Feb 2026). Store: `corpus.parquet` plus the FAISS /
bm25s indexes — no database server is needed.

---

## Phase 0 — Project Skeleton & Environment

### 0.1 Recommended Project Structure

```
isle-ir/
├── data/
│   ├── raw/              # downloaded JSON/JSONL from arXiv + OpenAlex
│   ├── processed/        # cleaned DataFrame as parquet
│   └── indexes/          # bm25s index + FAISS .index file
├── src/
│   ├── ingestion/
│   │   ├── arxiv_fetch.py
│   │   └── openalex_enrich.py
│   ├── retrieval/
│   │   ├── bm25_index.py
│   │   ├── faiss_index.py
│   │   └── rrf.py
│   ├── topics/
│   │   ├── nmf_model.py
│   │   └── bertopic_model.py
│   ├── graph/
│   │   ├── builder.py
│   │   └── analytics.py
│   └── app/
│       └── main.py       # Streamlit entry point
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_eval.ipynb
│   └── 03_topic_eval.ipynb
├── pyproject.toml       # dependencies + uv config — single source of truth
├── uv.lock              # pinned, reproducible install
├── setup.sh             # one-command environment setup
└── README.md
```

### 0.2 Dependencies — `pyproject.toml` + uv

Dependencies live in **`pyproject.toml`** (the single source of truth), locked
to exact versions in **`uv.lock`**. Hardware-specific packages — PyTorch and
FAISS — are split into two mutually-exclusive extras so one repo works across
the whole team:

| Extra | Installs | For |
|---|---|---|
| `gpu` | `torch` (CUDA 12.6) · `faiss-gpu-cu12` | Raahim's RTX 3060 rig · Google Colab |
| `cpu` | `torch` (CPU / Apple MPS) · `faiss-cpu` | Mujtaba's M4 · Abubakar's Intel laptop |

`torch` is pulled from the matching PyTorch index (`download.pytorch.org/whl/
cu126` or `/cpu`) via `[tool.uv.sources]`; `[tool.uv] conflicts` stops the two
extras being installed together. Everything else (pandas, scikit-learn,
sentence-transformers, bertopic, networkx, streamlit, gensim, …) is
platform-agnostic and lives in the base `dependencies`; `ruff` and `pytest`
are in the `dev` dependency-group.

> CUDA 12.6 (`cu126`) is the chosen PyTorch build — stable for the RTX 3060
> (Ampere) and the Colab T4 (Turing), and current as of May 2026. FAISS-GPU
> comes from the maintained `faiss-gpu-cu12` wheels (Linux x86-64 only — a
> platform marker keeps the universal lockfile resolvable on macOS/Windows).

`requirements-gpu.txt` / `requirements-cpu.txt` are `uv export` snapshots kept
as a fallback for environments without uv (e.g. a bare `pip` cell on Colab).

### 0.3 Environment Setup

[`uv`](https://docs.astral.sh/uv/) manages the Python toolchain, the virtual
environment and the lockfile. There is no conda step and no manual
`pip install -e .` — `uv sync` installs the project editable.

```bash
# Install uv once per machine
curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS / Linux / WSL
#   Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex

# Create .venv (Python 3.11) + install — setup.sh auto-detects GPU vs CPU
./setup.sh                       # or force it:  ./setup.sh gpu | ./setup.sh cpu

# Equivalent manual form
uv sync --extra gpu              # RTX rig / Colab
uv sync --extra cpu              # M4 Mac / Intel laptop
```

**Verify GPU** (RTX rig / Colab):
```bash
uv run python -c "import torch, faiss; \
  print(torch.cuda.get_device_name(0)); \
  print('FAISS GPUs:', faiss.get_num_gpus())"
# NVIDIA GeForce RTX 3060
# FAISS GPUs: 1
```

---

## Phase 1 — Data Acquisition

The ISLE paper uses arXiv + OpenAlex. You have three options in increasing order of effort:

| Option | Size | Time | Best for |
|--------|------|------|---------|
| **A — Kaggle arXiv snapshot** | 2.4M papers | ~30 min download | Fast start |
| **B — arXiv REST API** | any size | ~2–4 hrs for 100k | Controlled, recent |
| **C — OAI-PMH harvest** | full corpus | days | Production-scale |

**Recommendation: Option A for corpus, then enrich with OpenAlex API.**

### 1.1 Option A — Kaggle arXiv Dataset (Fastest)

The official Kaggle arXiv metadata dataset is the easiest bulk source.

```bash
# Install Kaggle CLI
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
kaggle datasets download -d Cornell-University/arxiv
unzip arxiv.zip -d data/raw/
```

Dataset file: `arxiv-metadata-oai-snapshot.json` — one JSON object per line.

```python
# src/ingestion/arxiv_fetch.py
import json, pandas as pd
from pathlib import Path

def load_arxiv_snapshot(path: str, categories: list[str], max_records: int = 100_000) -> pd.DataFrame:
    """
    Load arXiv Kaggle snapshot, filter to given categories.
    categories: e.g. ['cs.IR', 'cs.LG', 'cs.CL', 'cs.AI']
    """
    records = []
    target_cats = set(categories)

    with open(path) as f:
        for line in f:
            if len(records) >= max_records:
                break
            paper = json.loads(line)
            paper_cats = set(paper.get("categories", "").split())
            if paper_cats & target_cats:        # intersection: paper has at least one target cat
                records.append({
                    "arxiv_id":   paper["id"],
                    "title":      paper["title"].replace("\n", " ").strip(),
                    "abstract":   paper["abstract"].replace("\n", " ").strip(),
                    "authors":    paper["authors"],
                    "categories": paper["categories"],
                    "year":       int(paper.get("update_date", "2020")[:4]),
                    "doi":        paper.get("doi", ""),
                })

    df = pd.DataFrame(records)
    df["text"] = df["title"] + " " + df["abstract"]   # unified field for indexing
    df.to_parquet("data/processed/corpus.parquet", index=False)
    print(f"Saved {len(df)} papers to parquet")
    return df

# Usage
df = load_arxiv_snapshot(
    "data/raw/arxiv-metadata-oai-snapshot.json",
    categories=["cs.IR", "cs.LG", "cs.CL", "cs.AI", "cs.CV"],
    max_records=150_000
)
```

**GitHub reference:** https://github.com/mattbierbaum/arxiv-public-datasets

### 1.2 Option B — arXiv REST API (For Recent Papers)

```python
# src/ingestion/arxiv_fetch.py  (API variant)
import arxiv, time

def fetch_arxiv_api(query: str = "cat:cs.IR OR cat:cs.LG",
                    max_results: int = 10_000) -> list[dict]:
    client = arxiv.Client(page_size=500, delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    records = []
    for result in client.results(search):
        records.append({
            "arxiv_id":   result.entry_id.split("/abs/")[-1],
            "title":      result.title,
            "abstract":   result.summary,
            "authors":    [a.name for a in result.authors],
            "year":       result.published.year,
            "doi":        result.doi or "",
            "categories": [c for c in result.categories],
        })
    return records
```

**PyPI:** https://pypi.org/project/arxiv/ — `arxiv.Client` is the modern interface; the old `arxiv.Search` alone is deprecated.

### 1.3 OpenAlex Enrichment (Citations, Affiliations, Countries)

**Critical note as of 2026:** OpenAlex now requires a **free API key** (as of Feb 13 2026). Register at https://openalex.org/api-key.

```python
# src/ingestion/openalex_enrich.py
import pyalex, time
from pyalex import Works, config

# Setup — do this once at startup
config.email = "your@email.com"
config.api_key = "YOUR_FREE_OPENALEX_KEY"   # required from Feb 2026
config.max_retries = 3
config.retry_backoff_factor = 0.5
config.retry_http_codes = [429, 500, 503]

def enrich_batch(doi_list: list[str]) -> dict:
    """Return dict keyed by DOI with citation counts, institutions, countries."""
    results = {}
    for doi in doi_list:
        if not doi:
            continue
        try:
            work = Works()[f"doi:{doi}"]
            # PyAlex reconstructs plain-text abstract from inverted index on the fly
            institutions = []
            countries = []
            for authorship in work.get("authorships", []):
                for inst in authorship.get("institutions", []):
                    institutions.append(inst.get("display_name", ""))
                    countries.append(inst.get("country_code", ""))

            results[doi] = {
                "cited_by_count": work.get("cited_by_count", 0),
                "references":     [ref.get("id","") for ref in work.get("referenced_works", [])],
                "institutions":   list(set(institutions)),
                "countries":      list(set(countries)),
                "oa_id":          work.get("id", ""),
            }
            time.sleep(0.05)    # polite pool: ~20 req/sec
        except Exception as e:
            print(f"Failed DOI {doi}: {e}")
    return results
```

**Key pyalex note:** `Works()["doi:..."]` uses the `__getitem__` shortcut. For batch processing, use cursor pagination:

```python
# Fetch all cs.IR papers from OpenAlex directly (alternative to arXiv)
from itertools import chain

query = Works().filter(
    primary_topic={"field": {"id": "https://openalex.org/fields/17"}},  # Computer Science
    publication_year=">2018",
    type="article"
).sort("cited_by_count:desc")

all_works = []
for page in query.paginate(per_page=200, n_max=50_000):
    all_works.extend(page)
```

**GitHub tutorials:** https://github.com/ourresearch/openalex-api-tutorials (official Jupyter notebooks)

---

## Phase 2 — Hybrid Retrieval Engine (Core of the Project)

### 2.1 BM25 Lexical Retrieval with `bm25s`

**Why `bm25s` over `rank_bm25`:** `bm25s` uses eager sparse scoring with scipy sparse matrices. Benchmarks show it's **500× faster** than `rank_bm25` on BEIR datasets and faster or equivalent to Elasticsearch on single-node workloads — with zero Java dependency.

**Paper:** arXiv:2407.03618 | **GitHub:** https://github.com/xhluca/bm25s

```python
# src/retrieval/bm25_index.py
import bm25s, Stemmer, json
import numpy as np
from pathlib import Path

class BM25Index:
    def __init__(self, index_dir: str = "data/indexes/bm25"):
        self.index_dir = index_dir
        self.stemmer = Stemmer.Stemmer("english")   # pip install PyStemmer
        self.retriever = None
        self.paper_ids = []

    def build(self, papers: list[dict]) -> None:
        """Build and persist BM25 index from paper dicts."""
        corpus_texts = [f"{p['title']} {p['abstract']}" for p in papers]
        self.paper_ids = [p["arxiv_id"] for p in papers]

        # Tokenise with stopwords + stemming
        tokenized = bm25s.tokenize(
            corpus_texts,
            stopwords="en",
            stemmer=self.stemmer
        )

        self.retriever = bm25s.BM25(
            corpus=corpus_texts,
            k1=1.5,     # ISLE paper default BM25 params
            b=0.75
        )
        self.retriever.index(tokenized)

        # Persist
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        self.retriever.save(self.index_dir, corpus=corpus_texts)
        with open(f"{self.index_dir}/paper_ids.json", "w") as f:
            json.dump(self.paper_ids, f)
        print(f"BM25 index built: {len(corpus_texts)} docs")

    def load(self) -> None:
        """Load persisted index."""
        self.retriever = bm25s.BM25.load(self.index_dir, load_corpus=True)
        with open(f"{self.index_dir}/paper_ids.json") as f:
            self.paper_ids = json.load(f)

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return [(arxiv_id, score), ...] sorted by descending BM25 score."""
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, stopwords="en")
        results, scores = self.retriever.retrieve(query_tokens, k=top_k)
        return [(self.paper_ids[idx], float(score))
                for idx, score in zip(results[0], scores[0])]
```

**Known pitfall:** `bm25s.tokenize` returns a `bm25s.tokenization.Tokenized` object, not a plain list. Always pass it directly to `retrieve()` — don't try to access `.ids` manually unless doing custom scoring.

### 2.2 Dense Semantic Retrieval with FAISS

**Index choice for your corpus size:**

| Corpus Size | Recommended Index | Why |
|---|---|---|
| < 100k | `IndexFlatIP` | Exact search, fast enough |
| 100k–1M | `IndexIVFFlat` (nlist=1024) | ~10× faster, ~1% recall loss |
| 1M+ | `IndexIVFPQ` | Compressed, fits in VRAM |

For a 100k–200k paper corpus, `IndexIVFFlat` is the sweet spot.

```python
# src/retrieval/faiss_index.py
import faiss, numpy as np, torch, json
from sentence_transformers import SentenceTransformer
from pathlib import Path

class FAISSIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_dir: str = "data/indexes/faiss",
                 use_gpu: bool = True):
        self.model_name = model_name
        self.index_dir = index_dir
        self.dim = 384                  # all-MiniLM-L6-v2 output dimension
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.model = None
        self.index = None
        self.paper_ids = []

    def _load_model(self):
        if self.model is None:
            device = "cuda" if self.use_gpu else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)

    def build(self, papers: list[dict], batch_size: int = 256) -> None:
        """Encode all papers and build FAISS IVF index."""
        self._load_model()
        self.paper_ids = [p["arxiv_id"] for p in papers]
        texts = [f"{p['title']} {p['abstract']}" for p in papers]

        # Encode in batches — normalize for cosine similarity via dot product
        print("Encoding corpus...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2-normalize → dot product = cosine
            convert_to_numpy=True
        ).astype(np.float32)

        # Build IVFFlat index
        n = len(embeddings)
        nlist = min(1024, int(np.sqrt(n)))   # rule: nlist ≈ sqrt(N)

        quantizer = faiss.IndexFlatIP(self.dim)
        cpu_index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)

        if self.use_gpu:
            # Train and add on GPU
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            gpu_index.train(embeddings)
            gpu_index.add(embeddings)
            cpu_index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            cpu_index.train(embeddings)
            cpu_index.add(embeddings)

        # Persist
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        faiss.write_index(cpu_index, f"{self.index_dir}/vectors.index")
        with open(f"{self.index_dir}/paper_ids.json", "w") as f:
            json.dump(self.paper_ids, f)
        np.save(f"{self.index_dir}/embeddings.npy", embeddings)  # keep for BERTopic reuse
        self.index = cpu_index
        print(f"FAISS index built: {n} vectors, nlist={nlist}")

    def load(self) -> None:
        self.index = faiss.read_index(f"{self.index_dir}/vectors.index")
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        with open(f"{self.index_dir}/paper_ids.json") as f:
            self.paper_ids = json.load(f)

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        self._load_model()
        self.index.nprobe = 64      # probe 64 of nlist=1024 clusters; tunable recall/speed tradeoff
        query_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        distances, indices = self.index.search(query_vec, top_k)
        return [(self.paper_ids[idx], float(dist))
                for idx, dist in zip(indices[0], distances[0])
                if idx != -1]       # FAISS returns -1 for empty slots
```

**Critical pitfall — IVFFlat requires training:** Unlike `IndexFlatIP`, `IndexIVFFlat` must be trained on a sample of your data before adding vectors. The code above handles this but it's a common source of `faiss.swigfaiss.FaissException` errors for newcomers.

**FAISS index guide:** https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
**GPU usage:** https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU

### 2.3 Reciprocal Rank Fusion

This is the mathematically clean piece. From Cormack et al. (SIGIR 2009):

```python
# src/retrieval/rrf.py
from collections import defaultdict

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_n: int = 50
) -> list[tuple[str, float]]:
    """
    Standard RRF from Cormack et al. (SIGIR 2009).

    ranked_lists: each is [(doc_id, original_score), ...] sorted desc by relevance.
    k=60 is the standard constant from the original paper.

    RRF_score(d) = Σ 1 / (k + rank(d)) over all rankers.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank_pos, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank_pos)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def hybrid_search(
    query: str,
    bm25_index,
    faiss_index,
    top_k_each: int = 100,
    rrf_k: int = 60,
    top_n: int = 50
) -> list[tuple[str, float]]:
    """Full hybrid search pipeline: BM25 + Dense → RRF fusion."""
    bm25_results   = bm25_index.search(query, top_k=top_k_each)
    faiss_results  = faiss_index.search(query, top_k=top_k_each)
    fused          = reciprocal_rank_fusion([bm25_results, faiss_results],
                                            k=rrf_k, top_n=top_n)
    return fused
```

**Why NOT weighted score fusion:** BM25 scores are unbounded; cosine is [0, 1]. Even with min-max normalization, the distribution mismatch causes one method to dominate for certain queries. RRF sidesteps this entirely by using rank positions. Source: https://avchauzov.github.io/blog/2025/hybrid-retrieval-rrf-rank-fusion/

**Reference implementation with multiple retrievers:** https://github.com/nabeelshan78/Optimizing-RAG-with-Hybrid-Search

---

## Phase 3 — Topic Modeling

**Crucial implementation detail:** Pre-compute and cache your sentence embeddings (from Phase 2). BERTopic accepts pre-computed embeddings — you save 15–30 minutes of encoding time on each run.

### 3.1 NMF — CPU Baseline (Always Available)

Use `gensim`'s NMF for coherence sweeping to find optimal `k`, then `sklearn`'s NMF for final training (faster, more stable).

```python
# src/topics/nmf_model.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

def find_optimal_k(docs: list[str], k_range: range = range(5, 21)) -> int:
    """Sweep k values, return k with best c_v coherence."""
    tokenized = [d.lower().split() for d in docs]
    dictionary = Dictionary(tokenized)

    tfidf = TfidfVectorizer(max_features=15_000, stop_words="english",
                            min_df=2, max_df=0.90)
    X = tfidf.fit_transform(docs)
    vocab = tfidf.get_feature_names_out()

    best_k, best_score = 5, -1.0
    scores = []
    for k in k_range:
        model = NMF(n_components=k, random_state=42, max_iter=500, init="nndsvda")
        model.fit(X)
        # Top-10 words per topic
        topics_words = [[vocab[i] for i in row.argsort()[-10:][::-1]]
                        for row in model.components_]
        cm = CoherenceModel(topics=topics_words, texts=tokenized,
                            dictionary=dictionary, coherence="c_v")
        score = cm.get_coherence()
        scores.append((k, score))
        if score > best_score:
            best_k, best_score = k, score
        print(f"k={k}: coherence={score:.4f}")

    print(f"\nBest k={best_k} (c_v={best_score:.4f})")
    return best_k

def run_nmf(docs: list[str], n_topics: int = 10) -> tuple[list[dict], list[int], list[float]]:
    """
    Returns:
        topic_info: [{topic_id, keywords, doc_count}, ...]
        doc_topics: list of topic assignments per document
        doc_probs:  list of max topic probability per document
    """
    tfidf = TfidfVectorizer(max_features=15_000, stop_words="english",
                            min_df=2, max_df=0.90)
    X = tfidf.fit_transform(docs)
    vocab = tfidf.get_feature_names_out()

    model = NMF(n_components=n_topics, random_state=42, max_iter=1000, init="nndsvda")
    W = model.fit_transform(X)   # doc × topic
    H = model.components_        # topic × term

    doc_topics = W.argmax(axis=1).tolist()
    doc_probs  = W.max(axis=1).tolist()

    topic_info = []
    for i, row in enumerate(H):
        top_idx = row.argsort()[-10:][::-1]
        topic_info.append({
            "topic_id":  i,
            "keywords":  [vocab[j] for j in top_idx],
            "weights":   [float(row[j]) for j in top_idx],
            "doc_count": int((np.array(doc_topics) == i).sum()),
        })
    return topic_info, doc_topics, doc_probs
```

**NMF tutorial with coherence selection:** https://github.com/derekgreene/topic-model-tutorial
*(Notebook 3 covers exactly this k-selection approach.)*

### 3.2 BERTopic — Neural Pipeline (GPU-accelerated)

**Key insight from BERTopic best practices page:** Always separate the embedding step from the modeling step. This way you can tune UMAP/HDBSCAN parameters without re-encoding 100k documents each time.

```python
# src/topics/bertopic_model.py
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import torch

def build_bertopic_model(min_cluster_size: int = 15,
                         n_components: int = 5,
                         n_neighbors: int = 15) -> BERTopic:
    """
    Construct a tunable BERTopic model.
    min_cluster_size controls number of topics indirectly:
        higher → fewer topics, lower → more topics.
    """
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,           # keep tight clusters; 0.0 is the BERTopic default
        metric="cosine",
        random_state=42         # reproducibility
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True    # required for approximate_predict on new docs
    )

    # Scientific text: bigrams help ("neural network", "machine translation")
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90
    )

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=10,
        nr_topics="auto",       # merges highly similar topics post-hoc
        calculate_probabilities=True,
        verbose=True
    )


def run_bertopic(
    docs: list[str],
    precomputed_embeddings: np.ndarray | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> tuple[BERTopic, list[int], np.ndarray]:
    """
    Fit BERTopic. Reuses precomputed_embeddings from FAISS build step.
    Returns: (fitted_model, topic_assignments, probabilities)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if precomputed_embeddings is None:
        encoder = SentenceTransformer(model_name, device=device)
        print("Encoding documents (this takes a while without precomputed embeddings)...")
        precomputed_embeddings = encoder.encode(
            docs, batch_size=256, show_progress_bar=True,
            normalize_embeddings=True
        )

    topic_model = build_bertopic_model()
    topics, probs = topic_model.fit_transform(docs, embeddings=precomputed_embeddings)
    return topic_model, topics, probs


def get_topic_info(topic_model: BERTopic) -> list[dict]:
    """Structured topic info compatible with KG builder."""
    info = topic_model.get_topic_info()
    return [
        {
            "topic_id":  row["Topic"],
            "keywords":  row["Representation"] if "Representation" in row else [],
            "doc_count": row["Count"],
            "label":     row.get("Name", f"Topic_{row['Topic']}"),
        }
        for _, row in info.iterrows()
        if row["Topic"] != -1   # -1 = outlier cluster in HDBSCAN
    ]
```

**Critical BERTopic pitfalls from GitHub issues:**
1. **Reproducibility:** Always set `random_state=42` in UMAP. Without it, results change every run.
2. **Outlier cluster -1:** HDBSCAN assigns noisy documents to topic -1. Filter these out in downstream steps.
3. **`min_dist=0.0`:** Any positive value tends to scatter clusters; keep it at 0 for topic modeling.
4. **`prediction_data=True` in HDBSCAN:** Required if you want to call `topic_model.transform()` on new documents later.

**Source:** https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html
**Parameter tuning guide:** https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html
**Scientific abstracts walkthrough:** https://kb-labb.github.io/posts/2023-03-17-scientific-discourse-with-bertopic/

### 3.3 Topics-Over-Time (Temporal Analysis, mirrors ISLE Figure 6)

```python
# Built into BERTopic — generates Figure 6 Right equivalent
timestamps = [str(papers[i]["year"]) for i in range(len(docs))]
topics_over_time = topic_model.topics_over_time(
    docs, timestamps, nr_bins=10, global_tuning=True
)
fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
fig.write_html("outputs/topics_over_time.html")   # Interactive Plotly chart
```

---

## Phase 4 — Knowledge Graph Construction

### 4.1 Graph Schema (Matches ISLE Section 3.4 Exactly)

```python
# src/graph/builder.py
import networkx as nx
from typing import Optional

NODE_TYPES = {
    "paper":       {"color": "#4fc3f7", "size": 10},
    "author":      {"color": "#81c784", "size": 8},
    "institution": {"color": "#ffb74d", "size": 8},
    "country":     {"color": "#ce93d8", "size": 12},
    "topic":       {"color": "#ff8a65", "size": 15},
    "year":        {"color": "#90a4ae", "size": 6},
}

def build_knowledge_graph(
    papers: list[dict],
    topic_assignments: list[int],
    topic_probs: list[float],
    topic_info: list[dict],
) -> nx.DiGraph:
    """
    Builds typed multigraph matching ISLE Eq. 14–18.
    Node types: paper (VP), author (VA), institution (VI),
                country (VC), topic (VT), year (VY)
    Edge types: authored, affiliated, from_country,
                published_in, belongs_to, cites
    """
    G = nx.DiGraph()

    # ── Paper nodes ─────────────────────────────────────────────────────────
    for paper, t_id, t_prob in zip(papers, topic_assignments, topic_probs):
        pid = paper["arxiv_id"]
        G.add_node(pid,
                   node_type="paper",
                   title=paper["title"],
                   abstract=paper.get("abstract", "")[:500],
                   year=paper.get("year", 0),
                   cited_by_count=paper.get("cited_by_count", 0),
                   topic_id=int(t_id),
                   topic_prob=round(float(t_prob), 4))

    # ── Author nodes ─────────────────────────────────────────────────────────
    for paper in papers:
        pid = paper["arxiv_id"]
        for author_name in paper.get("authors", []):
            # Normalise name as node key
            a_id = f"author::{author_name.strip().lower()}"
            if not G.has_node(a_id):
                G.add_node(a_id, node_type="author", name=author_name)
            G.add_edge(a_id, pid, rel="authored")

    # ── Institution + Country nodes ─────────────────────────────────────────
    for paper in papers:
        pid = paper["arxiv_id"]
        for inst in paper.get("institutions", []):
            i_id = f"inst::{inst.lower()}"
            if not G.has_node(i_id):
                G.add_node(i_id, node_type="institution", name=inst)
            G.add_edge(pid, i_id, rel="affiliated")

        for country in paper.get("countries", []):
            c_id = f"country::{country}"
            if not G.has_node(c_id):
                G.add_node(c_id, node_type="country", code=country)
            G.add_edge(pid, c_id, rel="from_country")

    # ── Year nodes ───────────────────────────────────────────────────────────
    for paper in papers:
        pid = paper["arxiv_id"]
        yr = paper.get("year", 0)
        y_id = f"year::{yr}"
        if yr and not G.has_node(y_id):
            G.add_node(y_id, node_type="year", year=yr)
        if yr:
            G.add_edge(pid, y_id, rel="published_in")

    # ── Topic nodes ──────────────────────────────────────────────────────────
    paper_set = {p["arxiv_id"] for p in papers}
    for t in topic_info:
        t_id = f"topic::{t['topic_id']}"
        G.add_node(t_id,
                   node_type="topic",
                   keywords=t["keywords"][:5],
                   doc_count=t["doc_count"],
                   label=", ".join(t["keywords"][:3]))

    for paper, t_int, t_prob in zip(papers, topic_assignments, topic_probs):
        if t_int == -1:     # skip outlier cluster
            continue
        pid = paper["arxiv_id"]
        t_id = f"topic::{t_int}"
        G.add_edge(pid, t_id, rel="belongs_to", weight=round(float(t_prob), 4))

    # ── Citation edges ───────────────────────────────────────────────────────
    # Only keep intra-corpus citations (within retrieved set)
    for paper in papers:
        pid = paper["arxiv_id"]
        for ref_oa_id in paper.get("references", []):
            # Match OpenAlex IDs back to arxiv IDs
            ref_arxiv = paper.get("ref_arxiv_map", {}).get(ref_oa_id)
            if ref_arxiv and ref_arxiv in paper_set:
                G.add_edge(pid, ref_arxiv, rel="cites")

    return G
```

### 4.2 Graph Analytics (ISLE Eq. 19–20)

```python
# src/graph/analytics.py
import networkx as nx

def compute_paper_impact(G: nx.DiGraph) -> dict[str, int]:
    """Citation indegree = impact proxy (ISLE Eq. 19)."""
    return {n: G.in_degree(n)
            for n, d in G.nodes(data=True)
            if d.get("node_type") == "paper"}

def compute_entity_impact(G: nx.DiGraph, entity_type: str) -> dict[str, int]:
    """
    Cumulative citation indegree of associated papers (ISLE Eq. 20).
    entity_type: 'author', 'institution', 'country'
    """
    paper_impact = compute_paper_impact(G)
    entity_impact = {}
    for node, data in G.nodes(data=True):
        if data.get("node_type") != entity_type:
            continue
        total = sum(
            paper_impact.get(nbr, 0)
            for nbr in G.successors(node)        # outgoing edges from entity → paper
            if G.nodes[nbr].get("node_type") == "paper"
        )
        entity_impact[node] = total
    return entity_impact

def top_papers_by_citation(G: nx.DiGraph, n: int = 10) -> list[tuple[str, dict, int]]:
    paper_impact = compute_paper_impact(G)
    sorted_papers = sorted(paper_impact.items(), key=lambda x: x[1], reverse=True)[:n]
    return [(pid, G.nodes[pid], score) for pid, score in sorted_papers]

def graph_summary(G: nx.DiGraph) -> dict:
    type_counts = {}
    for _, d in G.nodes(data=True):
        t = d.get("node_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types":  type_counts,
        "density":     nx.density(G),
    }
```

### 4.3 Visualization

**Option A — PyVis (renders standalone HTML):**
```python
from pyvis.network import Network

def export_pyvis(G: nx.DiGraph, output: str = "graph.html",
                 max_nodes: int = 500) -> None:
    """Export subgraph to interactive HTML via PyVis."""
    # Limit to top nodes by degree for readability
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
    sub = G.subgraph([n for n, _ in top_nodes])

    net = Network(height="750px", width="100%", bgcolor="#111827",
                  font_color="#e5e7eb", directed=True)
    net.set_options("""{ "physics": { "barnesHut": { "gravitationalConstant": -8000 } } }""")

    color_map = {t: v["color"] for t, v in NODE_TYPES.items()}
    for node, attrs in sub.nodes(data=True):
        ntype = attrs.get("node_type", "paper")
        label = attrs.get("title", attrs.get("name", str(node)))[:40]
        net.add_node(str(node), label=label,
                     color=color_map.get(ntype, "#aaa"),
                     title=str({k: v for k, v in attrs.items() if k != "abstract"}),
                     size=NODE_TYPES.get(ntype, {}).get("size", 8))

    for u, v, attrs in sub.edges(data=True):
        net.add_edge(str(u), str(v), title=attrs.get("rel", ""))

    net.save_graph(output)
    print(f"Graph saved to {output}")
```

**Option B — `streamlit-agraph` (inline in Streamlit, no file needed):**
```python
from streamlit_agraph import agraph, Node, Edge, Config
# pip install streamlit-agraph

def render_agraph(G: nx.DiGraph, max_nodes: int = 200):
    nodes, edges = [], []
    color_map = {t: v["color"] for t, v in NODE_TYPES.items()}
    for node, attrs in list(G.nodes(data=True))[:max_nodes]:
        ntype = attrs.get("node_type", "paper")
        nodes.append(Node(id=str(node),
                          label=str(node)[:25],
                          color=color_map.get(ntype, "#aaa"),
                          size=NODE_TYPES.get(ntype, {}).get("size", 8)))
    for u, v, attrs in G.edges(data=True):
        if str(u) in {n.id for n in nodes} and str(v) in {n.id for n in nodes}:
            edges.append(Edge(source=str(u), target=str(v),
                              label=attrs.get("rel", "")))

    config = Config(width=900, height=600, directed=True,
                    physics=True, hierarchical=False)
    return agraph(nodes=nodes, edges=edges, config=config)
```

---

## Phase 5 — Streamlit Application

**Streamlit's rendering model:** Every user interaction re-runs the entire script from top to bottom. Use `@st.cache_resource` (for models/indexes, loaded once) and `@st.cache_data` (for data frames, cached per arguments). Without this, your FAISS index reloads on every click.

```python
# src/app/main.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
sys.path.insert(0, "src")

from retrieval.bm25_index import BM25Index
from retrieval.faiss_index import FAISSIndex
from retrieval.rrf import hybrid_search
from topics.nmf_model import run_nmf, find_optimal_k
from topics.bertopic_model import run_bertopic, get_topic_info
from graph.builder import build_knowledge_graph
from graph.analytics import graph_summary, top_papers_by_citation
from graph.visualization import export_pyvis

# ── Cached resource loading ──────────────────────────────────────────────────
@st.cache_resource
def load_indexes():
    bm25  = BM25Index(); bm25.load()
    faiss = FAISSIndex(); faiss.load()
    return bm25, faiss

@st.cache_data
def load_corpus() -> pd.DataFrame:
    return pd.read_parquet("data/processed/corpus.parquet")

@st.cache_data
def load_embeddings() -> np.ndarray:
    return np.load("data/indexes/faiss/embeddings.npy")

# ── Layout ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IR Explorer", layout="wide", page_icon="🔬")
st.title("🔬 Hybrid IR Explorer")
st.caption("BM25 + Dense Retrieval → RRF → BERTopic → Knowledge Graph")

df = load_corpus()
bm25_idx, faiss_idx = load_indexes()
all_embeddings = load_embeddings()    # shape: (N, 384)
paper_ids_ordered = df["arxiv_id"].tolist()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search Parameters")
    query    = st.text_input("Query", placeholder="e.g. transformer attention mechanism")
    top_n    = st.slider("Max results", 10, 200, 50)
    use_bert = st.toggle("BERTopic (GPU)", value=True)
    year_min, year_max = st.slider("Year range", 2015, 2025, (2019, 2025))
    min_cite = st.number_input("Min citations", 0, 1000, 0)
    run_btn  = st.button("🔍 Search", type="primary")

# ── Search & Display ──────────────────────────────────────────────────────────
if run_btn and query:
    with st.spinner("Running hybrid retrieval..."):
        fused = hybrid_search(query, bm25_idx, faiss_idx, top_k_each=200, top_n=top_n)

    result_ids = [pid for pid, _ in fused]
    result_df = df[df["arxiv_id"].isin(result_ids)].copy()
    result_df["rrf_score"] = result_df["arxiv_id"].map(dict(fused))

    # Apply filters
    result_df = result_df[
        (result_df["year"].between(year_min, year_max)) &
        (result_df.get("cited_by_count", 0) >= min_cite)
    ].sort_values("rrf_score", ascending=False)

    docs = result_df["abstract"].tolist()
    papers_list = result_df.to_dict("records")

    # Get precomputed embeddings for retrieved subset (avoid re-encoding)
    pid_to_idx  = {pid: i for i, pid in enumerate(paper_ids_ordered)}
    subset_idxs = [pid_to_idx[pid] for pid in result_df["arxiv_id"] if pid in pid_to_idx]
    subset_embs = all_embeddings[subset_idxs]

    # ── Topic Modeling ───────────────────────────────────────────────────────
    with st.spinner("Running topic modeling..."):
        if use_bert:
            topic_model, topic_assignments, probs = run_bertopic(docs, subset_embs)
            topic_info = get_topic_info(topic_model)
            doc_probs = probs.max(axis=1).tolist() if hasattr(probs, "max") else list(probs)
        else:
            topic_info, topic_assignments, doc_probs = run_nmf(docs, n_topics=10)

    # ── Knowledge Graph ──────────────────────────────────────────────────────
    with st.spinner("Building knowledge graph..."):
        G = build_knowledge_graph(papers_list, topic_assignments, doc_probs, topic_info)
        summary = graph_summary(G)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Results", "📊 Analytics", "🧩 Topics", "🕸 Knowledge Graph"])

    with tab1:
        st.metric("Papers Found", len(result_df))
        for _, row in result_df.head(20).iterrows():
            with st.expander(f"**{row['title']}** — {row['year']} | RRF: {row['rrf_score']:.4f}"):
                st.write(row["abstract"])
                cols = st.columns(3)
                cols[0].metric("Citations", row.get("cited_by_count", "N/A"))
                cols[1].write(f"**ArXiv ID:** {row['arxiv_id']}")
                cols[2].write(f"**Categories:** {row.get('categories','')}")

    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Papers", summary["total_nodes"])
        col2.metric("Graph Edges", summary["total_edges"])
        col3.metric("Topics Found", len(topic_info))
        col4.metric("Authors", summary["node_types"].get("author", 0))

        fig_year = px.histogram(result_df, x="year", nbins=20,
                                title="Publication Year Distribution",
                                color_discrete_sequence=["#4fc3f7"])
        st.plotly_chart(fig_year, use_container_width=True)

        top_papers = top_papers_by_citation(G, n=10)
        st.subheader("Top Cited Papers (in retrieved set)")
        for pid, attrs, score in top_papers:
            st.write(f"**{attrs.get('title','?')}** ({attrs.get('year','?')}) — {score} citations")

    with tab3:
        fig_topics = px.pie(
            values=[t["doc_count"] for t in topic_info],
            names=[", ".join(t["keywords"][:3]) for t in topic_info],
            title="Topic Distribution"
        )
        st.plotly_chart(fig_topics, use_container_width=True)

        if use_bert:
            with st.spinner("Generating word clouds..."):
                try:
                    fig_wc = topic_model.visualize_barchart(top_n_topics=8, n_words=8)
                    st.plotly_chart(fig_wc, use_container_width=True)
                except Exception as e:
                    st.warning(f"Word cloud unavailable: {e}")

    with tab4:
        st.write(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
        export_pyvis(G, output="/tmp/graph.html", max_nodes=300)
        with open("/tmp/graph.html") as f:
            st.components.v1.html(f.read(), height=700, scrolling=True)
```

---

## Phase 6 — Evaluation

### 6.1 Retrieval Evaluation (No Ground-Truth Labels Needed)

Since you have no human relevance judgments, use these proxy metrics:

```python
# notebooks/02_retrieval_eval.ipynb

# ── Metric 1: Jaccard Overlap (how much do BM25 and Semantic agree?) ─────────
def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)

bm25_top20   = [pid for pid, _ in bm25_results[:20]]
faiss_top20  = [pid for pid, _ in faiss_results[:20]]
hybrid_top20 = [pid for pid, _ in fused[:20]]

print(f"BM25 ∩ Semantic Jaccard@20: {jaccard(bm25_top20, faiss_top20):.3f}")
print(f"Hybrid ∩ BM25   Jaccard@20: {jaccard(hybrid_top20, bm25_top20):.3f}")
print(f"Hybrid ∩ Semantic Jaccard@20: {jaccard(hybrid_top20, faiss_top20):.3f}")
# Expect: Hybrid partially overlaps both — that's the "best of both" argument

# ── Metric 2: Result Set Semantic Coherence ────────────────────────────────
from sklearn.metrics.pairwise import cosine_similarity
result_embs = subset_embs[:len(hybrid_top20)]
sim_mat = cosine_similarity(result_embs)
np.fill_diagonal(sim_mat, 0)
coherence = sim_mat.sum() / (len(result_embs) * (len(result_embs) - 1))
print(f"Result set intra-coherence: {coherence:.4f}")
# Compare across queries: higher = more focused results

# ── Metric 3: MRR on a small hand-annotated set ────────────────────────────
# Create 10-20 test queries manually, note which papers are relevant.
# This is doable in an afternoon and makes your evaluation section solid.
def mrr(ranked_list: list[str], relevant_ids: set[str]) -> float:
    for rank, pid in enumerate(ranked_list, start=1):
        if pid in relevant_ids:
            return 1.0 / rank
    return 0.0
```

### 6.2 Topic Coherence Evaluation

```python
# notebooks/03_topic_eval.ipynb
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

tokenized = [d.lower().split() for d in docs]
dictionary = Dictionary(tokenized)

def eval_coherence(keywords_per_topic: list[list[str]], measure: str = "c_v") -> float:
    cm = CoherenceModel(topics=keywords_per_topic, texts=tokenized,
                        dictionary=dictionary, coherence=measure)
    return cm.get_coherence()

nmf_keywords     = [t["keywords"] for t in nmf_topic_info]
bertopic_keywords = [t["keywords"] for t in bert_topic_info]

print("NMF c_v:", eval_coherence(nmf_keywords, "c_v"))
print("NMF u_mass:", eval_coherence(nmf_keywords, "u_mass"))
print("BERTopic c_v:", eval_coherence(bertopic_keywords, "c_v"))
print("BERTopic u_mass:", eval_coherence(bertopic_keywords, "u_mass"))

# Report both — ISLE paper (Eq. 12) uses NPMI which is approximated by c_npmi:
print("BERTopic c_npmi:", eval_coherence(bertopic_keywords, "c_npmi"))
```

**Note from Gensim docs:** `c_v` is the most cited metric but has criticisms. Report `u_mass` alongside it — it's computed on corpus co-occurrences and is more reliable for scientific text evaluation.

### 6.3 Knowledge Graph Statistics

One figure for the KG section of the report: node/edge counts and the degree
distribution, which exposes the graph's power-law structure.

```python
import plotly.express as px

print(f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

degrees = [d for _, d in G.degree()]
fig = px.histogram(x=degrees, log_y=True,
                   title="Node Degree Distribution (log scale)",
                   labels={"x": "Degree"})
fig.write_html("evaluation/results/kg_degree_distribution.html")
```

---

## Complete Library & Resource Reference

### Libraries

| Library | Role | Install | Key Docs / Source |
|---|---|---|---|
| `arxiv` | arXiv REST API client | `pip install arxiv` | https://pypi.org/project/arxiv/ |
| `sickle` | arXiv OAI-PMH bulk harvest | `pip install sickle` | https://sickle.readthedocs.io/ |
| `pyalex` | OpenAlex API wrapper | `pip install pyalex` | https://github.com/J535D165/pyalex |
| `bm25s` | Ultra-fast pure-Python BM25 | `pip install bm25s[full]` | https://github.com/xhluca/bm25s |
| `sentence-transformers` | MiniLM-L6-v2 encoding | `pip install sentence-transformers` | https://www.sbert.net/ |
| `faiss-gpu` | ANN vector search | `pip install faiss-gpu` | https://github.com/facebookresearch/faiss |
| `bertopic` | Neural topic modeling | `pip install bertopic` | https://maartengr.github.io/BERTopic/ |
| `umap-learn` | Dimensionality reduction for BERTopic | `pip install umap-learn` | https://umap-learn.readthedocs.io/ |
| `hdbscan` | Density clustering for BERTopic | `pip install hdbscan` | https://hdbscan.readthedocs.io/ |
| `scikit-learn` | NMF + TF-IDF | `pip install scikit-learn` | https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html |
| `gensim` | Coherence scoring only | `pip install gensim` | https://radimrehurek.com/gensim/models/coherencemodel.html |
| `networkx` | Knowledge graph (in-memory) | `pip install networkx` | https://networkx.org/ |
| `pyvis` | Interactive graph HTML export | `pip install pyvis` | https://pyvis.readthedocs.io/ |
| `streamlit-agraph` | Inline graph in Streamlit | `pip install streamlit-agraph` | https://github.com/ChrisDelClea/streamlit-agraph |
| `streamlit` | Frontend app | `pip install streamlit` | https://docs.streamlit.io/ |
| `plotly` | Charts + BERTopic visualizations | `pip install plotly` | https://plotly.com/python/ |

### Papers

| Paper | What it covers | Where |
|---|---|---|
| ISLE (Jani et al. 2025) | Your reference system | arXiv:2512.12760 |
| BM25S (Lù 2024) | Why `bm25s` is the right library | arXiv:2407.03618 |
| RRF (Cormack et al. 2009) | Mathematical foundation of fusion | SIGIR 2009 |
| BERTopic (Grootendorst 2022) | Neural topic model algorithm | arXiv:2203.05794 |
| MiniLM (Song et al. 2020) | Embedding model you're using | NeurIPS 2020 / arXiv:2002.10957 |
| OpenAlex (Priem et al. 2022) | Scholarly metadata source | arXiv:2205.01833 |
| FAISS (Johnson et al. 2017) | GPU vector search | arXiv:1702.08734 |
| LitSearch (Ajith et al. 2024) | IR benchmark for science queries | EMNLP 2024 |

### GitHub Repositories to Study

| Repo | What to learn from it |
|---|---|
| https://github.com/xhluca/bm25s | BM25 indexing API + persistence pattern |
| https://github.com/MaartenGr/BERTopic | BERTopic full codebase + examples |
| https://github.com/nabeelshan78/Optimizing-RAG-with-Hybrid-Search | Complete hybrid BM25 + dense + RRF pipeline |
| https://github.com/derekgreene/topic-model-tutorial | NMF coherence sweep (k-selection) |
| https://github.com/mattbierbaum/arxiv-public-datasets | arXiv bulk data tools |
| https://github.com/ourresearch/openalex-api-tutorials | OpenAlex API notebooks |
| https://github.com/carloodq/rrf | Minimal RRF implementation |
| https://github.com/J535D165/pyalex | pyalex complete usage examples |
| https://github.com/facebookresearch/faiss | FAISS index types + GPU guide |

### Official Docs to Bookmark

| Resource | URL |
|---|---|
| BERTopic algorithm | https://maartengr.github.io/BERTopic/algorithm/algorithm.html |
| BERTopic best practices | https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html |
| BERTopic parameter tuning | https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html |
| FAISS index guide | https://github.com/facebookresearch/faiss/wiki/Faiss-indexes |
| FAISS GPU | https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU |
| SentenceTransformers MiniLM | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| OpenAlex API reference | https://docs.openalex.org/ |
| arXiv bulk data | https://info.arxiv.org/help/bulk_data.html |
| Gensim coherence | https://radimrehurek.com/gensim/models/coherencemodel.html |
| Streamlit cache | https://docs.streamlit.io/develop/concepts/architecture/caching |
| NetworkX tutorial | https://networkx.org/documentation/stable/tutorial.html |

---

## Timeline

```
Week 1:  Phase 0 (env) + Phase 1 (arXiv Kaggle download + OpenAlex enrichment)
Week 2:  Phase 2 (BM25 index + FAISS index + RRF) — run sanity queries
Week 3:  Phase 3 (NMF k-sweep + BERTopic on retrieved subsets)
Week 4:  Phase 4 (graph builder + analytics + PyVis)
Week 5:  Phase 5 (Streamlit UI) + Phase 6 (evaluation notebook) + report
```

**Development order for Phase 2 (most important):** Build BM25 first → verify results manually → build FAISS → verify → add RRF → compare all three modes side by side.
