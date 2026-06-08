"""Analytics tab — charts over the retrieved set and graph.

Owner:   Mujtaba Kamran (SSOT §2)
Phase:   5 — Streamlit UI
STATUS:  core
"""

from __future__ import annotations

import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from graph.analytics import graph_summary, top_papers_by_citation


def render(G: nx.DiGraph, result_df=None) -> None:
    """Render publication-year histogram, graph summary, top-cited table,
    and citation flow Sankey with bridge papers (B-A3)."""
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

    # ── Citation flow Sankey (B-A3) ───────────────────────────────────────
    st.divider()
    st.subheader("Citation Flow Over Time")

    paper_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "paper"]

    try:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "evaluation"))
        from citation_flow import build_citation_flow, find_bridge_papers

        flow = build_citation_flow(G, paper_nodes)

        if not flow["labels"]:
            st.info("Not enough intra-set citation links for flow analysis.")
        else:
            fig_sankey = go.Figure(
                go.Sankey(
                    node=dict(label=flow["labels"], pad=15, thickness=20),
                    link=dict(
                        source=flow["source"],
                        target=flow["target"],
                        value=flow["value"],
                    ),
                )
            )
            fig_sankey.update_layout(
                title_text="Citation flow between publication years",
                font_size=12,
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_sankey, use_container_width=True)

            bridge = find_bridge_papers(G, paper_nodes, top_n=5)
            if bridge:
                st.subheader("Bridge Papers")
                st.caption("Papers with highest betweenness centrality in the citation subgraph.")
                bridge_rows = []
                for pid, score in bridge:
                    attrs = G.nodes.get(pid, {})
                    bridge_rows.append(
                        {
                            "Title": attrs.get("title", pid),
                            "Year": attrs.get("year", "—"),
                            "Centrality": round(score, 4),
                        }
                    )
                st.dataframe(bridge_rows, hide_index=True, use_container_width=True)

    except Exception as exc:
        st.caption(f"Citation flow analysis unavailable: {exc}")
