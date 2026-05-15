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


def build_knowledge_graph(
    papers: list[dict],
    topic_assignments: list[int],
    topic_probs: list[float],
    topic_info: list[dict],
) -> nx.DiGraph:
    """Build the typed knowledge graph — Contract F (SSOT §6F).

    Node types: paper, author, institution, country, topic, year.
    Edge types: authored, affiliated, from_country, published_in,
                belongs_to, cites. Every node carries a ``node_type`` attr.
    """
    # TODO(mujtaba, phase-4): add nodes/edges per IMPLEMENTATION.md §4.1;
    #   keep citation edges intra-corpus; skip outlier topic -1.
    raise NotImplementedError
