"""Knowledge Graph tab — interactive graph view.

Owner:   Mujtaba Kamran (SSOT §2)
Phase:   5 — Streamlit UI
STATUS:  core
"""

from __future__ import annotations

import html
import json
import os
import tempfile

import networkx as nx
import streamlit as st
from pyvis.network import Network as PyVisNetwork

from graph.builder import NODE_TYPES
from graph.paths import find_citation_path
from graph.visualization import export_pyvis


def render(G: nx.DiGraph, result_papers: list[dict]) -> None:
    """Embed the interactive knowledge graph and citation-path explorer.

    Parameters
    ----------
    G:
        The knowledge graph produced by ``graph.builder.build_knowledge_graph``.
    result_papers:
        Paper dicts from the retrieval pipeline, each containing ``arxiv_id``
        and ``title`` at minimum.
    """
    st.subheader("Knowledge Graph")

    # -- Summary metrics ---------------------------------------------------
    col_nodes, col_edges = st.columns(2)
    col_nodes.metric("Nodes", G.number_of_nodes())
    col_edges.metric("Edges", G.number_of_edges())

    # -- PyVis interactive graph --------------------------------------------
    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", prefix="graph_"
    )
    tmp.close()
    try:
        export_pyvis(G, output=tmp.name, max_nodes=300)
        with open(tmp.name, encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=700, scrolling=True)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    # -- Citation Path Explorer ---------------------------------------------
    _render_path_explorer(G, result_papers)


# ---------------------------------------------------------------------------
# Citation Path Explorer
# ---------------------------------------------------------------------------


def _render_path_explorer(G: nx.DiGraph, result_papers: list[dict]) -> None:
    """Render the Citation Path Explorer sub-section."""
    st.divider()
    st.subheader("Citation Path Explorer")

    # Check that the graph has enough citation edges to be useful.
    cites_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("rel") == "cites"
    ]
    if len(cites_edges) < 2:
        st.info("Too few intra-set citation links for path exploration.")
        return

    # Build display-title -> arxiv_id mapping (truncated titles for selectbox).
    title_to_id: dict[str, str] = {}
    display_titles: list[str] = []
    for p in result_papers:
        title = p.get("title", "Untitled")
        short = title if len(title) <= 60 else title[:60] + "..."
        title_to_id[short] = p["arxiv_id"]
        display_titles.append(short)

    col_a, col_b = st.columns(2)
    sel_a = col_a.selectbox("From paper", display_titles, key="path_from")
    sel_b = col_b.selectbox("To paper", display_titles, key="path_to")

    if st.button("Find connection", key="find_path"):
        id_a = title_to_id.get(sel_a)
        id_b = title_to_id.get(sel_b)
        if not id_a or not id_b:
            st.warning("Selected papers could not be resolved to IDs.")
            return

        path = find_citation_path(G, id_a, id_b, max_hops=4)
        if path:
            st.success(
                f"Path found ({len(path) - 1} hops):  {'  →  '.join(path)}"
            )
            _render_highlighted_path(G, path)
        else:
            st.warning("No citation path found within 4 hops.")


# ---------------------------------------------------------------------------
# Highlighted-path PyVis graph
# ---------------------------------------------------------------------------


def _render_highlighted_path(G: nx.DiGraph, path: list[str]) -> None:
    """Render a focused PyVis graph with the citation path highlighted."""
    HIGHLIGHT_COLOR = "#FFD700"

    # Collect all nodes reachable from path nodes (1-hop) for context.
    expanded: set[str] = set(path)
    for node in path:
        for neighbor in G.predecessors(node):
            expanded.add(neighbor)
        for neighbor in G.successors(node):
            expanded.add(neighbor)

    sub = G.subgraph(expanded).copy()

    path_set = set(path)
    path_edge_pairs: set[tuple[str, str]] = set()
    for i in range(len(path) - 1):
        path_edge_pairs.add((path[i], path[i + 1]))

    net = PyVisNetwork(
        height="500px",
        width="100%",
        bgcolor="#111827",
        font_color="#e5e7eb",
        directed=True,
    )
    net.set_options(
        '{ "physics": { "barnesHut": { "gravitationalConstant": -8000 } } }'
    )

    color_map = {t: v["color"] for t, v in NODE_TYPES.items()}

    for node, attrs in sub.nodes(data=True):
        ntype = attrs.get("node_type", "paper")
        raw_label = attrs.get("title", attrs.get("name", str(node)))[:40]
        label = html.escape(raw_label)
        tooltip = {k: v for k, v in attrs.items() if k != "abstract"}
        color = HIGHLIGHT_COLOR if node in path_set else color_map.get(ntype, "#aaa")
        size = (
            15 if node in path_set else NODE_TYPES.get(ntype, {}).get("size", 8)
        )
        net.add_node(
            str(node),
            label=label,
            color=color,
            title=html.escape(json.dumps(tooltip, ensure_ascii=False)),
            size=size,
        )

    for u, v, attrs in sub.edges(data=True):
        edge_color = (
            HIGHLIGHT_COLOR
            if (u, v) in path_edge_pairs
            else "#555555"
        )
        edge_width = 2.5 if (u, v) in path_edge_pairs else 1
        net.add_edge(
            str(u),
            str(v),
            title=html.escape(attrs.get("rel", "")),
            color=edge_color,
            width=edge_width,
        )

    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", prefix="path_"
    )
    tmp.close()
    try:
        net.save_graph(tmp.name)
        with open(tmp.name, encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=500, scrolling=True)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
