"""Shared project configuration — the single source of truth for paths,
model names and hyperparameters.

Owner:  All (SSOT §5)
STATUS: core

Every module imports constants from here. Do NOT hardcode a path, model name
or hyperparameter anywhere else — that is the most common cause of
"works on my machine" integration failures (SSOT §15).
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
# ROOT resolves to the repo root: src/app/config.py → src/app → src → root.
ROOT = Path(__file__).resolve().parent.parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CORPUS_PATH = DATA_PROCESSED / "corpus.parquet"
BM25_DIR = ROOT / "data" / "indexes" / "bm25"
FAISS_DIR = ROOT / "data" / "indexes" / "faiss"
BERTOPIC_DIR = ROOT / "data" / "indexes" / "bertopic"
EVAL_RESULTS_DIR = ROOT / "evaluation" / "results"

# ── Embedding model — DECIDED: SPECTER2 ──────────────────────────────────
# allenai/specter2_base — a scientific-document embedding model trained on
# citation graphs, purpose-built for scientific-paper retrieval. Chosen over
# general-purpose MiniLM because this is a scientific-literature search
# system; 768-dim is comfortable on the RTX 3060. MiniLM/384 is kept only as
# the baseline for the SPECTER2-vs-MiniLM study (SSOT §10). Loading SPECTER2
# needs the `adapters` library (base model + proximity adapter).
# This file is the ONLY knob — change here and it propagates everywhere.
EMBEDDING_MODEL = "allenai/specter2_base"
EMBEDDING_DIM = 768

# Cross-encoder reranker (enhancement — SSOT §10).
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── BM25 (ISLE paper §3.2.1) ─────────────────────────────────────────────
BM25_K1 = 1.5
BM25_B = 0.75

# ── Retrieval ────────────────────────────────────────────────────────────
TOP_K_EACH = 100   # results BM25 and FAISS each return before fusion
RRF_K = 60         # Reciprocal Rank Fusion constant (Cormack et al. 2009)
RRF_TOP_N = 50     # results kept after fusion
RERANK_TOP_K = 20  # cross-encoder final output count

# ── FAISS IVFFlat ────────────────────────────────────────────────────────
FAISS_NLIST = 1024
FAISS_NPROBE = 64

# ── BERTopic ─────────────────────────────────────────────────────────────
BERTOPIC_MIN_CLUSTER = 15
BERTOPIC_N_COMPONENTS = 5   # UMAP output dimensions
BERTOPIC_N_NEIGHBORS = 15
BERTOPIC_UMAP_MIN_DIST = 0.0      # 0.0 keeps tight clusters — BERTopic default
BERTOPIC_UMAP_METRIC = "cosine"
BERTOPIC_RANDOM_STATE = 42
BERTOPIC_HDBSCAN_MIN_SAMPLES = 5
BERTOPIC_HDBSCAN_METRIC = "euclidean"
BERTOPIC_HDBSCAN_SELECTION = "eom"
BERTOPIC_TOP_N_WORDS = 10
BERTOPIC_MIN_DF = 3
BERTOPIC_MAX_DF = 0.90
BERTOPIC_NR_TOPICS: str | int = "auto"
BERTOPIC_HDBSCAN_N_JOBS = -1              # use all CPU cores for core-dist computation
BERTOPIC_HDBSCAN_APPROX_MST = True        # approximate MST — speedup, minor quality tradeoff
BERTOPIC_HDBSCAN_ALGORITHM = "boruvka_kdtree"  # fastest for euclidean + low-dim (5-dim UMAP output)

# ── NMF ──────────────────────────────────────────────────────────────────
NMF_K_RANGE = range(5, 21)
NMF_TFIDF_MAX_FEATURES = 15_000
NMF_TFIDF_MIN_DF = 2
NMF_TFIDF_MAX_DF = 0.90
NMF_RANDOM_STATE = 42
NMF_MAX_ITER_SWEEP = 500
NMF_MAX_ITER_FINAL = 1000
NMF_INIT = "nndsvda"
NMF_TOP_N_WORDS = 10

# ── Freshness (bonus B-M1) ───────────────────────────────────────────────
FRESHNESS_LAMBDA = 0.3

# ── OpenAlex — free API key required since Feb 2026 ──────────────────────
# Keep real credentials OUT of git. Set them in the environment (or a local
# .env / .claude/settings.local.json, both gitignored). Register a free key
# at https://openalex.org/api-key.
# TODO(team, phase-0): set OPENALEX_EMAIL / OPENALEX_API_KEY in your env.
OA_EMAIL = os.environ.get("OPENALEX_EMAIL", "your-team-email@nu.edu.pk")
OA_API_KEY = os.environ.get("OPENALEX_API_KEY", "")
