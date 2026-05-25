"""Knowledge-graph visualization.

Owner:    Mujtaba Kamran (SSOT §2)
Contract: none
Phase:    4 — Knowledge Graph (IMPLEMENTATION.md §4.3)
STATUS:   core

Renders the graph as interactive HTML (PyVis) or inline (streamlit-agraph).
"""

from __future__ import annotations

import networkx as nx
from pyvis.network import Network

from graph.builder import NODE_TYPES


def export_pyvis(
    G: nx.DiGraph, output_path: str = "graph.html", max_nodes: int = 500
) -> None:
    """Export a degree-capped subgraph to an interactive HTML file.

    Limits the graph to *max_nodes* highest-degree nodes for readability.
    Nodes are coloured and sized by ``node_type`` using the ``NODE_TYPES``
    map from ``builder.py``.
    """
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
    sub = G.subgraph([n for n, _ in top_nodes]).copy()

    net = Network(
        height="750px",
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
        label = attrs.get("title", attrs.get("name", str(node)))[:40]
        tooltip = {k: v for k, v in attrs.items() if k != "abstract"}
        net.add_node(
            str(node),
            label=label,
            color=color_map.get(ntype, "#aaa"),
            title=str(tooltip),
            size=NODE_TYPES.get(ntype, {}).get("size", 8),
        )

    for u, v, attrs in sub.edges(data=True):
        net.add_edge(str(u), str(v), title=attrs.get("rel", ""))

    net.save_graph(output_path)
