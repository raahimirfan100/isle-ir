from __future__ import annotations

import os
import tempfile

import pytest

from graph.analytics import (
    compute_entity_impact,
    compute_paper_impact,
    graph_summary,
    top_papers_by_citation,
)
from graph.builder import build_knowledge_graph
from graph.collab_network import (
    build_collab_network,
    export_collab_pyvis,
    top_betweenness_authors,
)
from graph.visualization import export_pyvis

PAPERS = [
    {
        "arxiv_id": "2301.00001",
        "title": "Attention Is All You Need",
        "abstract": "We propose a new simple network architecture.",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2023,
        "categories": ["cs.CL"],
        "doi": "",
        "cited_by_count": 50,
        "institutions": ["Google Research", "DeepMind"],
        "countries": ["US", "GB"],
        "references": ["W2964149567"],
        "ref_arxiv_map": {"W2964149567": "2301.00002"},
    },
    {
        "arxiv_id": "2301.00002",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": "We introduce a new language representation model.",
        "authors": ["Jacob Devlin", "Noam Shazeer"],
        "year": 2023,
        "categories": ["cs.CL"],
        "doi": "",
        "cited_by_count": 30,
        "institutions": ["Google Research"],
        "countries": ["US"],
        "references": [],
    },
    {
        "arxiv_id": "2301.00003",
        "title": "Graph Neural Networks: A Review",
        "abstract": "Comprehensive review of GNNs.",
        "authors": ["Jie Zhou", "Cuiyun Gao"],
        "year": 2022,
        "categories": ["cs.LG"],
        "doi": "",
        "cited_by_count": 10,
        "institutions": ["Tsinghua University"],
        "countries": ["CN"],
        "references": ["W2964149567"],
        "ref_arxiv_map": {"W2964149567": "2301.00002"},
    },
]

TOPIC_ASSIGNMENTS = [0, 0, 1]
TOPIC_PROBS = [0.85, 0.72, 0.91]
TOPIC_INFO = [
    {
        "topic_id": 0,
        "keywords": ["transformer", "attention", "language model"],
        "doc_count": 2,
        "label": "transformer attention",
    },
    {
        "topic_id": 1,
        "keywords": ["graph", "neural network", "node"],
        "doc_count": 1,
        "label": "graph neural network",
    },
]


@pytest.fixture
def G():
    return build_knowledge_graph(
        PAPERS, TOPIC_ASSIGNMENTS, TOPIC_PROBS, TOPIC_INFO
    )


# ── builder ─────────────────────────────────────────────────────────────


class TestBuildKnowledgeGraph:
    def test_all_six_node_types_present(self, G):
        types = {d["node_type"] for _, d in G.nodes(data=True)}
        assert types == {
            "paper", "author", "institution", "country", "topic", "year",
        }

    def test_all_six_edge_rel_types_present(self, G):
        rels = {d["rel"] for _, _, d in G.edges(data=True)}
        assert rels == {
            "authored", "affiliated", "from_country",
            "published_in", "belongs_to", "cites",
        }

    def test_belongs_to_connects_papers_to_topics(self, G):
        for u, v, d in G.edges(data=True):
            if d["rel"] != "belongs_to":
                continue
            assert u in {p["arxiv_id"] for p in PAPERS}
            assert v in {"topic::0", "topic::1"}

    def test_cites_edges_intra_corpus_only(self, G):
        pids = {p["arxiv_id"] for p in PAPERS}
        for u, v, d in G.edges(data=True):
            if d["rel"] != "cites":
                continue
            assert u in pids and v in pids

    def test_outlier_topic_skipped(self, G):
        for _, _, d in G.edges(data=True):
            if d["rel"] == "belongs_to":
                assert "topic::-1" not in _


# ── analytics ───────────────────────────────────────────────────────────


class TestAnalytics:
    def test_compute_paper_impact(self, G):
        impact = compute_paper_impact(G)
        assert set(impact.keys()) == {p["arxiv_id"] for p in PAPERS}
        assert impact["2301.00002"] == 2  # cited by both 00001 and 00003

    def test_compute_entity_impact_author(self, G):
        impact = compute_entity_impact(G, "author")
        assert len(impact) > 0
        # Shazeer co-authored with authors of paper 00002 (cited twice)
        shazeer_impact = impact.get("author::noam shazeer", 0)
        assert shazeer_impact > 0

    def test_compute_entity_impact_institution(self, G):
        impact = compute_entity_impact(G, "institution")
        assert len(impact) > 0
        # Google Research is affiliated with papers cited twice total
        google_impact = impact.get("inst::google research", 0)
        assert google_impact > 0

    def test_compute_entity_impact_country(self, G):
        impact = compute_entity_impact(G, "country")
        assert len(impact) > 0
        us_impact = impact.get("country::US", 0)
        assert us_impact > 0

    def test_graph_summary(self, G):
        s = graph_summary(G)
        assert s["total_nodes"] > 0
        assert s["total_edges"] > 0
        assert "paper" in s["node_types"]
        assert "cites" in s["edge_types"]
        assert 0 < s["density"] < 1

    def test_top_papers_by_citation(self, G):
        top = top_papers_by_citation(G, n=3)
        assert len(top) == 3
        assert top[0][0] == "2301.00002"


# ── collab network ─────────────────────────────────────────────────────


class TestCollabNetwork:
    def test_builds_graph(self):
        collab = build_collab_network(PAPERS)
        assert collab.number_of_nodes() > 0
        assert collab.number_of_edges() > 0

    def test_coauthor_edges_weighted(self):
        collab = build_collab_network(PAPERS)
        u, v, w = list(collab.edges(data=True))[0]
        assert w["weight"] >= 1

    def test_top_betweenness(self):
        collab = build_collab_network(PAPERS)
        top = top_betweenness_authors(collab, n=3)
        assert len(top) == 3
        assert top[0][1] >= 0


# ── visualization ──────────────────────────────────────────────────────


class TestVisualization:
    def test_export_pyvis_produces_html(self, G):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            export_pyvis(G, output_path=path)
            assert os.path.getsize(path) > 0
            content = open(path).read()
            assert "<!DOCTYPE html>" in content or "<html" in content
        finally:
            os.unlink(path)

    def test_export_collab_pyvis_produces_html(self):
        collab = build_collab_network(PAPERS)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            export_collab_pyvis(collab, output=path)
            assert os.path.getsize(path) > 0
            content = open(path).read()
            assert "<!DOCTYPE html>" in content or "<html" in content
        finally:
            os.unlink(path)
