"""Knowledge-graph visualization.

Owner:    Mujtaba Kamran (SSOT §2)
Contract: none
Phase:    4 — Knowledge Graph (IMPLEMENTATION.md §4.3)
STATUS:   core

Renders the graph as interactive HTML (PyVis) or inline (streamlit-agraph).
"""

from __future__ import annotations

import networkx as nx


def export_pyvis(
    G: nx.DiGraph, output: str = "graph.html", max_nodes: int = 500
) -> None:
    """Export a degree-capped subgraph to an interactive HTML file."""
    # TODO(mujtaba, phase-4): PyVis export, colour by node_type —
    #   see IMPLEMENTATION.md §4.3.
    raise NotImplementedError
