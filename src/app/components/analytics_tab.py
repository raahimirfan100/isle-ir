"""Analytics tab — charts over the retrieved set and graph.

Owner:   Mujtaba Kamran (SSOT §2)
Phase:   5 — Streamlit UI
STATUS:  core
"""

from __future__ import annotations

import networkx as nx
import plotly.express as px
import streamlit as st

from graph.analytics import graph_summary, top_papers_by_citation


def render(G: nx.DiGraph, result_df=None) -> None:
    """Render publication-year histogram, graph summary and top-cited table."""
    if G.number_of_nodes() == 0:
        st.info("No graph data available. Run a query to build the knowledge graph.")
        return

    summary = graph_summary(G)

    # ── Key metrics row ────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Total nodes", summary["total_nodes"])
    c2.metric("Total edges", summary["total_edges"])
    c3.metric("Density", f"{summary['density']:.4f}")

    # ── Per-type breakdowns ────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Nodes by type")
        node_types = summary["node_types"]
        if node_types:
            st.dataframe(
                {"Type": list(node_types.keys()), "Count": list(node_types.values())},
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No node-type data found.")

    with right:
        st.subheader("Edges by type")
        edge_types = summary["edge_types"]
        if edge_types:
            st.dataframe(
                {"Type": list(edge_types.keys()), "Count": list(edge_types.values())},
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No edge-type data found.")

    st.divider()

    # ── Publication year histogram ─────────────────────────────────────
    if result_df is not None and not result_df.empty:
        year_col = "year"
        if year_col in result_df.columns:
            st.subheader("Publication year distribution")
            fig = px.histogram(
                result_df,
                x=year_col,
                nbins=20,
                labels={year_col: "Year", "count": "Papers"},
                title="Number of papers per publication year",
            )
            fig.update_layout(bargap=0.2)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                'Result DataFrame has no "year" column; year histogram is unavailable.'
            )

    # ── Top cited papers table ────────────────────────────────────────
    st.subheader("Top cited papers")
    top_papers = top_papers_by_citation(G, n=10)
    if top_papers:
        rows = []
        for pid, attrs, citations in top_papers:
            rows.append(
                {
                    "Title": attrs.get("title", pid),
                    "Year": attrs.get("year", "—"),
                    "Citations": citations,
                }
            )
        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True,
            column_order=["Title", "Year", "Citations"],
        )
    else:
        st.caption("No citation data available.")
