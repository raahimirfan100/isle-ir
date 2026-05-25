"""Author collaboration network.

Owner:    Mujtaba Kamran (SSOT §2)
Contract: none
Phase:    4 — Knowledge Graph (SSOT §10 Above-and-Beyond)
STATUS:   enhancement

Co-authorship graph: edges weighted by shared papers; betweenness centrality
surfaces "bridge" researchers between communities.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
from pyvis.network import Network


def build_collab_network(papers: list[dict]) -> nx.Graph:
    """Build the weighted co-authorship graph.

    Nodes are authors, edges represent co-authorship weighted by the number
    of shared papers. Node attributes include total citation count across
    all their papers.
    """
    G = nx.Graph()

    author_citations: dict[str, int] = Counter()
    coauthor_pairs: Counter[tuple[str, str]] = Counter()

    for paper in papers:
        authors = paper.get("authors", [])
        cited = paper.get("cited_by_count", 0)
        normalised = tuple(a.strip().lower() for a in authors if a.strip())

        for author in normalised:
            author_citations[author] += cited

        for i in range(len(normalised)):
            for j in range(i + 1, len(normalised)):
                pair = tuple(sorted((normalised[i], normalised[j])))
                coauthor_pairs[pair] += 1

    for author, citations in author_citations.items():
        G.add_node(author, citation_count=citations)

    for (a, b), weight in coauthor_pairs.items():
        G.add_edge(a, b, weight=weight)

    return G


def top_betweenness_authors(
    G: nx.Graph, n: int = 10
) -> list[tuple[str, float]]:
    """Return top-*n* authors by betweenness centrality."""
    centrality = nx.betweenness_centrality(G, weight="weight")
    return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n]


def export_collab_pyvis(
    G: nx.Graph, output: str = "collab_network.html", max_nodes: int = 300
) -> None:
    """Export the collaboration network to an interactive PyVis HTML file."""
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
    sub = G.subgraph([n for n, _ in top_nodes]).copy()

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#111827",
        font_color="#e5e7eb",
    )
    net.set_options(
        '{ "physics": { "barnesHut": { "gravitationalConstant": -5000 } } }'
    )

    max_cit = max((d.get("citation_count", 0) for _, d in sub.nodes(data=True)), default=1) or 1

    for node, attrs in sub.nodes(data=True):
        cite = attrs.get("citation_count", 0)
        size = 5 + 20 * (cite / max_cit)
        net.add_node(
            str(node),
            label=str(node)[:30],
            size=size,
            title=f"{node} — citations: {cite}",
        )

    for u, v, attrs in sub.edges(data=True):
        net.add_edge(
            str(u), str(v), value=attrs.get("weight", 1),
            title=f"shared papers: {attrs.get('weight', 1)}",
        )

    net.save_graph(output)
