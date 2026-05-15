"""Knowledge Graph tab — interactive graph view.

Owner:   Mujtaba Kamran (SSOT §2)
Phase:   5 — Streamlit UI
STATUS:  core
"""

from __future__ import annotations

import networkx as nx


def render(G: nx.DiGraph) -> None:
    """Embed the interactive knowledge graph (PyVis HTML / streamlit-agraph)."""
    # TODO(mujtaba, phase-5): embed the graph; hover shows node metadata.
    raise NotImplementedError
