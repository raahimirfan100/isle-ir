"""Knowledge-graph builder.

Owner:    Mujtaba Kamran (SSOT §2)
Contract: F — build_knowledge_graph (SSOT §6F)
Phase:    4 — Knowledge Graph (IMPLEMENTATION.md §4.1)
STATUS:   core

Builds the heterogeneous DiGraph connecting papers, authors, institutions,
countries, topics and years. Course tie-in: Week 10 link analysis.
"""

from __future__ import annotations

import networkx as nx

NODE_TYPES = {
    "paper": {"color": "#4fc3f7", "size": 10},
    "author": {"color": "#81c784", "size": 8},
    "institution": {"color": "#ffb74d", "size": 8},
    "country": {"color": "#ce93d8", "size": 12},
    "topic": {"color": "#ff8a65", "size": 15},
    "year": {"color": "#90a4ae", "size": 6},
}


def build_knowledge_graph(
    papers: list[dict],
    topic_assignments: list[int],
    topic_probs: list[float],
    topic_info: list[dict],
) -> nx.DiGraph:
    """Build the typed knowledge graph — Contract F (SSOT §6F).

    Builds typed multigraph matching ISLE Eq. 14–18.

    Node types: paper, author, institution, country, topic, year.
    Edge types: authored, affiliated, from_country, published_in,
                belongs_to, cites. Every node carries a ``node_type`` attr.

    Citation edges are kept intra-corpus only (skip references not in the
    paper set). Outlier topic ``-1`` is skipped for ``belongs_to`` edges.
    """
    G = nx.DiGraph()

    # ── Paper nodes ─────────────────────────────────────────────────────
    for paper, t_id, t_prob in zip(
        papers, topic_assignments, topic_probs, strict=True
    ):
        pid = paper["arxiv_id"]
        G.add_node(
            pid,
            node_type="paper",
            title=paper["title"],
            abstract=paper.get("abstract", "")[:500],
            year=paper.get("year", 0),
            cited_by_count=paper.get("cited_by_count", 0),
            topic_id=int(t_id),
            topic_prob=round(float(t_prob), 4),
        )

    # ── Author nodes + authored edges ───────────────────────────────────
    for paper in papers:
        pid = paper["arxiv_id"]
        for author_name in paper.get("authors", []):
            a_id = f"author::{author_name.strip().lower()}"
            if not G.has_node(a_id):
                G.add_node(a_id, node_type="author", name=author_name)
            G.add_edge(a_id, pid, rel="authored")

    # ── Institution nodes + affiliated edges ─────────────────────────────
    for paper in papers:
        pid = paper["arxiv_id"]
        for inst in paper.get("institutions", []):
            i_id = f"inst::{inst.lower()}"
            if not G.has_node(i_id):
                G.add_node(i_id, node_type="institution", name=inst)
            G.add_edge(pid, i_id, rel="affiliated")

    # ── Country nodes + from_country edges ───────────────────────────────
    for paper in papers:
        pid = paper["arxiv_id"]
        for country in paper.get("countries", []):
            c_id = f"country::{country}"
            if not G.has_node(c_id):
                G.add_node(c_id, node_type="country", code=country)
            G.add_edge(pid, c_id, rel="from_country")

    # ── Year nodes + published_in edges ──────────────────────────────────
    for paper in papers:
        pid = paper["arxiv_id"]
        yr = paper.get("year", 0)
        if not yr:
            continue
        y_id = f"year::{yr}"
        if not G.has_node(y_id):
            G.add_node(y_id, node_type="year", year=yr)
        G.add_edge(pid, y_id, rel="published_in")

    # ── Topic nodes + belongs_to edges ───────────────────────────────────
    for t in topic_info:
        t_id = f"topic::{t['topic_id']}"
        G.add_node(
            t_id,
            node_type="topic",
            keywords=t["keywords"][:5],
            doc_count=t["doc_count"],
            label=", ".join(t["keywords"][:3]),
        )

    for paper, t_int, t_prob in zip(
        papers, topic_assignments, topic_probs, strict=True
    ):
        if t_int == -1:
            continue
        pid = paper["arxiv_id"]
        t_id = f"topic::{t_int}"
        G.add_edge(pid, t_id, rel="belongs_to", weight=round(float(t_prob), 4))

    # ── Citation edges (intra-corpus only) ───────────────────────────────
    paper_set = {p["arxiv_id"] for p in papers}
    for paper in papers:
        pid = paper["arxiv_id"]
        for ref_oa_id in paper.get("references", []):
            ref_arxiv = paper.get("ref_arxiv_map", {}).get(ref_oa_id)
            if ref_arxiv and ref_arxiv in paper_set:
                G.add_edge(pid, ref_arxiv, rel="cites")

    # TODO(mujtaba, enhancement): add pagerank() over the citation subgraph
    #   (nx.pagerank, alpha=0.85) and contrast it with raw in-degree —
    #   SSOT §10 above-and-beyond, Week 10 link-analysis tie-in.

    return G
