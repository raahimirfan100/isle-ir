"""Citation flow and temporal influence analysis.

Owner:   Abubakar Ahmed (SSOT §2)
Phase:   Bonus (B-A3)
STATUS:  enhancement

Uses citation edges and year metadata already in the KG to show how influence
propagates through the result set over time. Produces data for a Sankey diagram
and identifies 'bridge' papers by betweenness centrality.
Course tie-in: bibliometric analysis + link analysis (Week 10).
"""

from __future__ import annotations

import networkx as nx


def build_citation_flow(
    G: nx.DiGraph,
    result_ids: list[str],
) -> dict:
    """Build year-binned citation flow data for a Plotly Sankey diagram.

    Extracts the citation subgraph of *result_ids* from the full KG and
    counts directed citations between year bins.

    Returns
    -------
    {
      "labels": list[str],   # ["2019", "2020", "2021", ...]
      "source": list[int],   # index into labels (citing paper's year)
      "target": list[int],   # index into labels (cited paper's year)
      "value":  list[int],   # citation count between those year bins
    }
    Returns empty lists when there are fewer than 2 citation edges in the set.
    """
    result_set = set(result_ids)

    cites_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("rel") == "cites" and u in result_set and v in result_set
    ]

    if len(cites_edges) < 2:
        return {"labels": [], "source": [], "target": [], "value": []}

    year_of: dict[str, str] = {}
    for pid in result_ids:
        if pid in G.nodes:
            attrs = G.nodes[pid]
            year = attrs.get("year", "")
            if year:
                year_of[pid] = str(year)

    flow: dict[tuple[str, str], int] = {}
    for src, dst in cites_edges:
        y_src = year_of.get(src)
        y_dst = year_of.get(dst)
        if y_src and y_dst and y_src != y_dst:
            key = (y_src, y_dst)
            flow[key] = flow.get(key, 0) + 1

    if not flow:
        return {"labels": [], "source": [], "target": [], "value": []}

    all_years = sorted({y for pair in flow for y in pair})
    year_idx = {y: i for i, y in enumerate(all_years)}

    sources, targets, values = [], [], []
    for (y_src, y_dst), count in sorted(flow.items()):
        sources.append(year_idx[y_src])
        targets.append(year_idx[y_dst])
        values.append(count)

    return {"labels": all_years, "source": sources, "target": targets, "value": values}


def find_bridge_papers(
    G: nx.DiGraph,
    result_ids: list[str],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Return the top_n papers by betweenness centrality in the citation subgraph.

    Bridge papers sit on many shortest paths between other papers — they connect
    disparate sub-communities in the result set.

    Returns [(arxiv_id, betweenness_score), ...] sorted descending.
    Returns an empty list when the citation subgraph has fewer than 3 nodes.
    """
    result_set = set(result_ids)

    cites_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("rel") == "cites" and u in result_set and v in result_set
    ]

    if len(cites_edges) < 2:
        return []

    citation_sub = nx.DiGraph(cites_edges)
    if citation_sub.number_of_nodes() < 3:
        return []

    centrality = nx.betweenness_centrality(citation_sub, normalized=True)
    sorted_papers = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return sorted_papers[:top_n]
