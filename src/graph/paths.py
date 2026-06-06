"""Citation path finder for the knowledge graph.

Owner:    Mujtaba Kamran (SSOT §2)
Phase:    Bonus (B-M2)
STATUS:   enhancement

Finds shortest directed citation paths between papers in the KG.
"""

from __future__ import annotations

import networkx as nx


def find_citation_path(
    G: nx.DiGraph,
    source_id: str,
    target_id: str,
    max_hops: int = 4,
) -> list[str] | None:
    """Find the shortest directed citation path from source_id to target_id.

    Only traverses 'cites' edges. Returns a list of arxiv_ids forming the
    path (inclusive of source and target), or None if no path exists within
    max_hops.
    """
    cites_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("rel") == "cites"]
    citation_sub = nx.DiGraph(cites_edges)

    try:
        path = nx.shortest_path(citation_sub, source=source_id, target=target_id)
        if len(path) - 1 <= max_hops:
            return path
        return None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
