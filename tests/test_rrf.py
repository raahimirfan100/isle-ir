from __future__ import annotations

import pytest

from retrieval.rrf import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_merges_and_scores_all_documents() -> None:
    bm25_results = [("a", 12.0), ("b", 11.0), ("c", 10.0)]
    faiss_results = [("b", 0.9), ("d", 0.8)]

    fused = reciprocal_rank_fusion(bm25_results, faiss_results)

    assert fused == [
        ("b", pytest.approx((1 / 62) + (1 / 61))),
        ("a", pytest.approx(1 / 61)),
        ("d", pytest.approx(1 / 62)),
        ("c", pytest.approx(1 / 63)),
    ]