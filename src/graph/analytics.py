"""Knowledge-graph analytics.

Owner:    Mujtaba Kamran (SSOT §2)
Contract: none (consumes the Contract F graph)
Phase:    4 — Knowledge Graph (IMPLEMENTATION.md §4.2)
STATUS:   core

Citation-impact metrics over the graph (ISLE Eq. 19-20) plus summary stats.
PageRank link analysis is an enhancement — see the TODO below.
"""

from __future__ import annotations

import networkx as nx


def compute_paper_impact(G: nx.DiGraph) -> dict[str, int]:
    """Citation count per paper node (ISLE Eq. 19).

    Uses the ``cited_by_count`` node attribute (from OpenAlex) when available.
    Falls back to intra-result ``cites`` in-degree when the attribute is absent
    or zero.
    """
    impact: dict[str, int] = {}
    for node in G.nodes:
        attrs = G.nodes[node]
        if attrs.get("node_type") != "paper":
            continue
        corpus_count = attrs.get("cited_by_count", 0) or 0
        if corpus_count > 0:
            impact[node] = int(corpus_count)
        else:
            impact[node] = sum(
                1 for _, _, d in G.in_edges(node, data=True)
                if d.get("rel") == "cites"
            )
    return impact


def compute_entity_impact(G: nx.DiGraph, entity_type: str) -> dict[str, int]:
    """Cumulative citation in-degree of associated papers (ISLE Eq. 20).

    *entity_type* is one of ``"author"``, ``"institution"``, ``"country"``.
    Author nodes have outgoing edges (author → paper), while institution
    and country nodes have incoming edges (paper → entity). This function
    traverses the correct direction for each type.
    """
    paper_impact = compute_paper_impact(G)
    entity_impact: dict[str, int] = {}
    for node, data in G.nodes(data=True):
        if data.get("node_type") != entity_type:
            continue
        # Author → paper (outgoing); paper → institution/country (incoming)
        if entity_type == "author":
            neighbours = G.successors(node)
        else:
            neighbours = G.predecessors(node)
        total = sum(
            paper_impact.get(nbr, 0)
            for nbr in neighbours
            if G.nodes[nbr].get("node_type") == "paper"
        )
        entity_impact[node] = total
    return entity_impact


def top_papers_by_citation(
    G: nx.DiGraph, n: int = 10
) -> list[tuple[str, dict, int]]:
    """Return the top-*n* papers by citation in-degree."""
    paper_impact = compute_paper_impact(G)
    sorted_papers = sorted(paper_impact.items(), key=lambda x: x[1], reverse=True)[:n]
    return [(pid, G.nodes[pid], score) for pid, score in sorted_papers]


def graph_summary(G: nx.DiGraph) -> dict:
    """Node/edge counts, per-type breakdown and density for the Analytics tab."""
    node_types: dict[str, int] = {}
    edge_rels: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        t = data.get("node_type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    for _, _, data in G.edges(data=True):
        r = data.get("rel", "unknown")
        edge_rels[r] = edge_rels.get(r, 0) + 1
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types": node_types,
        "edge_types": edge_rels,
        "density": nx.density(G),
    }


# TODO(mujtaba, enhancement): add pagerank() over the citation subgraph
#   (nx.pagerank, alpha=0.85) and contrast it with raw in-degree —
#   SSOT §10 above-and-beyond, Week 10 link-analysis tie-in.
