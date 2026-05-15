"""Author collaboration network.

Owner:    Mujtaba Kamran (SSOT §2)
Contract: none
Phase:    3-4 (SSOT §10 Above-and-Beyond)
STATUS:   enhancement — schedule after the core graph works

Co-authorship graph: edges weighted by shared papers; betweenness centrality
surfaces "bridge" researchers between communities.
"""

from __future__ import annotations

import networkx as nx


def build_collab_network(papers: list[dict]) -> nx.Graph:
    """Build the weighted co-authorship graph."""
    # TODO(mujtaba, enhancement): co-authorship edges weighted by shared paper
    #   count; node size = total citation count; compute betweenness centrality.
    raise NotImplementedError
