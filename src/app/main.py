"""Streamlit application entry point.

Owner:    Mujtaba Kamran (layout) · Raahim Irfan (integration review) (SSOT §2)
Contract: none — orchestrates Contracts B-F
Phase:    5 — Streamlit UI (IMPLEMENTATION.md Phase 5)
STATUS:   core

Run with:  uv run streamlit run src/app/main.py

Loads the indexes and corpus (cached), runs the retrieval pipeline, models
topics, builds the knowledge graph, and renders the tabbed UI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import streamlit as st

# Ensure src/ is on sys.path so ``from app.config`` works when running
# ``streamlit run src/app/main.py`` from the repo root.
SRC = str(Path(__file__).resolve().parent.parent)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from app.config import (  # noqa: E402
    BERTOPIC_DIR,
    CORPUS_PATH,
    FAISS_DIR,
    FRESHNESS_LAMBDA,
)
from retrieval.bm25_index import BM25Index  # noqa: E402
from retrieval.faiss_index import FAISSIndex  # noqa: E402
from retrieval.pipeline import full_retrieval_pipeline  # noqa: E402

# ── Cached resource loading ────────────────────────────────────────────────


@st.cache_resource
def load_indexes() -> tuple[BM25Index, FAISSIndex]:
    bm25 = BM25Index()
    bm25.load()
    faiss = FAISSIndex()
    faiss.load()
    return bm25, faiss


@st.cache_data
def load_corpus() -> pd.DataFrame:
    return pd.read_parquet(str(CORPUS_PATH))


@st.cache_data
def load_embeddings() -> np.ndarray:
    return np.load(str(FAISS_DIR / "embeddings.npy"))


@st.cache_data
def load_bertopic_model():
    """Load a pre-fit BERTopic model for inference-only use."""
    model_dir = str(BERTOPIC_DIR / "model")
    if not Path(model_dir).exists():
        return None
    from bertopic import BERTopic
    return BERTopic.load(model_dir)


@st.cache_data
def _get_cached_topic_assignments(df: pd.DataFrame) -> list[int]:
    """Return full-corpus topic assignments (placeholder)."""
    # Real implementation would load from the pre-fit BERTopic model.
    # For now, return -1 for all — estimate_decay_rates handles this.
    return [-1] * len(df)


def _render_collab_tab(collab_G) -> None:
    """Render the collaboration network tab."""
    import os
    import tempfile

    from graph.collab_network import export_collab_pyvis, top_betweenness_authors

    st.subheader("Author Collaboration Network")

    if collab_G.number_of_nodes() == 0:
        st.info("No co-authorship links found in this result set.")
        return

    top_bc = top_betweenness_authors(collab_G, n=10)
    st.write("**Top bridge researchers (betweenness centrality):**")
    for author, score in top_bc:
        st.write(f"  - {author}: {score:.4f}")

    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w")
    tmp.close()
    try:
        export_collab_pyvis(collab_G, output=tmp.name, max_nodes=200)
        with open(tmp.name) as f:
            st.components.v1.html(f.read(), height=700, scrolling=True)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def _render_semantic_map_tab() -> None:
    """Stub for Raahim's B-R2 UMAP 2D embedding visualization."""
    st.subheader("Semantic Map")
    st.info(
        "This tab will display a 2D UMAP projection of "
        "retrieved papers, coloured by topic cluster. Coming soon."
    )


# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(page_title="ISLE-IR", layout="wide", page_icon="🔬")
st.title("🔬 ISLE-IR — Hybrid Scientific Literature Explorer")
st.caption(
    "BM25 + SPECTER2 Dense Retrieval → RRF → BERTopic → Knowledge Graph"
)

# ── Load resources ────────────────────────────────────────────────────────

with st.spinner("Loading indexes and corpus…"):
    bm25_index, faiss_index = load_indexes()
    corpus_df = load_corpus()
    all_embeddings = load_embeddings()

papers_lookup = cast(
    dict[str, dict], corpus_df.set_index("arxiv_id").to_dict("index")
)
paper_ids_ordered: list[str] = corpus_df["arxiv_id"].tolist()

# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Search")

    query = st.text_input(
        "Query", placeholder="e.g. transformer attention mechanism NLP"
    )

    use_expand = st.toggle(
        "Query expansion", value=True,
        help="WordNet synonyms + pseudo-relevance feedback",
    )
    use_rerank = st.toggle("Cross-encoder rerank", value=True)
    use_freshness = st.toggle(
        "Freshness boost", value=False,
        help="Boost recent papers in fast-moving topics",
    )
    freshness_lam = (
        st.slider("Freshness λ", 0.0, 1.0, FRESHNESS_LAMBDA, step=0.05)
        if use_freshness
        else FRESHNESS_LAMBDA
    )
    use_mmr = st.toggle(
        "MMR diversity", value=False,
        help="Balance relevance vs. diversity",
    )
    mmr_lam = (
        st.slider("MMR λ (1=relevance, 0=diversity)", 0.0, 1.0, 0.5, step=0.05)
        if use_mmr
        else 0.5
    )

    st.divider()
    st.header("Rocchio Feedback")
    st.caption("Mark results relevant / not relevant, then refine.")
    if st.button("🔄 Refine Search", disabled="results" not in st.session_state):
        st.info("Run a search first, then thumbs-up/down results.")

    run_btn = st.button("🔍 Search", type="primary", use_container_width=True)

# ── Search execution (cached in session state) ─────────────────────────────

if run_btn and query.strip():
    with st.spinner("Running hybrid retrieval…"):
        from retrieval.freshness import apply_freshness, estimate_decay_rates

        results = full_retrieval_pipeline(
            query,
            bm25_index,
            faiss_index,
            papers_lookup,
            expand_query=use_expand,
            rerank=use_rerank,
        )

        # Build pid → corpus-index lookup for downstream O(1) access
        pid_to_idx = {pid: i for i, pid in enumerate(paper_ids_ordered)}

        # Freshness boost (B-M1)
        if use_freshness and results:
            topic_assignments_full = _get_cached_topic_assignments(corpus_df)
            decay_rates = estimate_decay_rates(
                corpus_df, topic_assignments_full,
            )
            topic_assign_map = {
                pid: topic_assignments_full[pid_to_idx[pid]]
                for pid, _ in results
                if pid in pid_to_idx
            }
            results = apply_freshness(
                results, papers_lookup, topic_assign_map,
                decay_rates, lam=freshness_lam,
            )

        # MMR diversity re-ranking (B-R1)
        if use_mmr and results:
            from retrieval.mmr import mmr as apply_mmr

            embs = load_embeddings()
            query_vec = faiss_index._encode_query(query)
            candidates_with_embs = [
                (pid, score, embs[pid_to_idx[pid]])
                for pid, score in results
                if pid in pid_to_idx
            ]
            if candidates_with_embs:
                results = apply_mmr(query_vec[0], candidates_with_embs, lam=mmr_lam)

        # Store query embedding for Rocchio refinement
        st.session_state["query_emb"] = faiss_index._encode_query(query)[0]

        # Build papers list for downstream tabs
        result_ids = [pid for pid, _ in results]
        result_papers = [
            papers_lookup[pid] for pid in result_ids if pid in papers_lookup
        ]
        result_df = corpus_df[corpus_df["arxiv_id"].isin(result_ids)].copy()
        result_df["_score"] = result_df["arxiv_id"].map(dict(results))

        # Topic modeling on retrieved set
        topic_info = []
        doc_topics = []
        doc_probs = []
        bertopic_model = load_bertopic_model()
        if bertopic_model and len(result_papers) >= 5:
            with st.spinner("Running BERTopic on results…"):
                docs = [
                    p.get("text", f"{p['title']} {p['abstract']}")
                    for p in result_papers
                ]
                subset_idxs = [
                    pid_to_idx[pid]
                    for pid in result_ids
                    if pid in pid_to_idx
                ]
                subset_embs = all_embeddings[subset_idxs]
                predictions, _ = bertopic_model.transform(
                    docs, embeddings=subset_embs,
                )
                doc_topics = list(predictions)
                ti = bertopic_model.get_topic_info()
                for _, row in ti.iterrows():
                    tid = int(row["Topic"])
                    if tid == -1:
                        continue
                    kw = bertopic_model.get_topic(tid)
                    keywords = (
                        [w for w, _ in kw]
                        if isinstance(kw, (list, tuple))
                        else []
                    )
                    topic_info.append(
                        {
                            "topic_id": tid,
                            "keywords": keywords,
                            "doc_count": int(row["Count"]),
                            "label": str(row.get("Name", f"Topic_{tid}")),
                        }
                    )
                try:
                    topic_distr, _ = bertopic_model.approximate_distribution(
                        docs,
                        window=4,
                        stride=1,
                        batch_size=200,
                        use_embedding_model=False,
                    )
                    doc_probs = [
                        (
                            float(topic_distr[i, doc_topics[i]])
                            if doc_topics[i] != -1
                            else 0.0
                        )
                        for i in range(len(doc_topics))
                    ]
                except Exception:
                    doc_probs = [
                        1.0 if t != -1 else 0.0 for t in doc_topics
                    ]

        # Knowledge graph
        G = None
        if result_papers and topic_info:
            with st.spinner("Building knowledge graph…"):
                from graph.builder import build_knowledge_graph

                topics_for_kg = doc_topics[: len(result_papers)]
                probs_for_kg = doc_probs[: len(result_papers)]
                while len(topics_for_kg) < len(result_papers):
                    topics_for_kg.append(-1)
                    probs_for_kg.append(0.0)
                G = build_knowledge_graph(
                    result_papers, topics_for_kg, probs_for_kg, topic_info,
                )

        # Collaboration network
        collab_G = None
        if result_papers:
            with st.spinner("Building collaboration network…"):
                from graph.collab_network import build_collab_network

                collab_G = build_collab_network(result_papers)

        # Store in session state
        st.session_state["results"] = results
        st.session_state["result_papers"] = result_papers
        st.session_state["result_df"] = result_df
        st.session_state["topic_info"] = topic_info
        st.session_state["doc_topics"] = doc_topics
        st.session_state["doc_probs"] = doc_probs
        st.session_state["G"] = G
        st.session_state["collab_G"] = collab_G
        st.session_state["query"] = query

# ── Tab rendering ───────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.info(
        "Enter a query in the sidebar and click **Search** to get started."
    )
else:
    from app.components.analytics_tab import render as render_analytics
    from app.components.graph_tab import render as render_graph
    from app.components.results_tab import render as render_results
    from app.components.topics_tab import render as render_topics

    tabs = st.tabs(
        [
            "📄 Results",
            "🧩 Topics",
            "🕸 Knowledge Graph",
            "📊 Analytics",
            "👥 Collaboration Network",
            "🗺 Semantic Map",
        ]
    )
    tab_results, tab_topics, tab_graph, tab_analytics, tab_collab, tab_semantic = tabs

    with tab_results:
        render_results(
            st.session_state["results"],
            st.session_state["result_papers"],
            st.session_state["result_df"],
            faiss_index,
        )

    with tab_topics:
        render_topics(
            st.session_state["topic_info"],
            st.session_state["doc_topics"],
            st.session_state["result_papers"],
        )

    with tab_graph:
        if st.session_state["G"] is not None:
            render_graph(
                st.session_state["G"],
                st.session_state["result_papers"],
            )
        else:
            st.warning("Knowledge graph is empty — try a broader query.")

    with tab_analytics:
        if st.session_state["G"] is not None:
            render_analytics(
                st.session_state["G"],
                st.session_state["result_df"],
            )
        else:
            st.warning("No analytics data available.")

    with tab_collab:
        if st.session_state.get("collab_G") is not None:
            _render_collab_tab(st.session_state["collab_G"])
        else:
            st.warning("No collaboration data available.")

    with tab_semantic:
        _render_semantic_map_tab()


def main() -> None:
    """Render the Streamlit app."""
    pass


if __name__ == "__main__":
    main()
