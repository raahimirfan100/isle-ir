"""Unit tests for MMR diversity re-ranking."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retrieval.mmr import mmr


def _normed(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_mmr_returns_k_results():
    rng = np.random.default_rng(0)
    q = _normed(rng.random(8))
    cands = [(_normed(rng.random(8))) for _ in range(5)]
    candidates = [(f"p{i}", float(5 - i), e) for i, e in enumerate(cands)]
    result = mmr(q, candidates, lam=0.5, k=3)
    assert len(result) == 3
    assert all(isinstance(pid, str) and isinstance(s, float) for pid, s in result)


def test_mmr_returns_fewer_than_k_when_not_enough():
    q = _normed(np.array([1, 0, 0, 0], dtype=np.float32))
    candidates = [("p0", 1.0, q.copy())]
    result = mmr(q, candidates, lam=0.5, k=5)
    assert len(result) == 1


def test_mmr_empty_candidates():
    q = _normed(np.array([1, 0, 0, 0], dtype=np.float32))
    assert mmr(q, [], lam=0.5, k=3) == []


def test_mmr_drops_near_duplicate():
    """Near-duplicate paper should be excluded in favour of a diverse one."""
    q = np.array([1, 0, 0, 0], dtype=np.float32)

    # dup1 and dup2 are almost the same direction
    dup1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    dup2 = _normed(np.array([0.99, 0.14, 0.0, 0.0], dtype=np.float32))
    # div is orthogonal to both q and dup1
    div = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    candidates = [
        ("dup1", 3.0, dup1),
        ("dup2", 2.9, dup2),
        ("div", 2.0, div),
    ]
    # With low lam (diversity-heavy), dup2 should be penalised after dup1 is chosen
    result_ids = [pid for pid, _ in mmr(q, candidates, lam=0.1, k=2)]
    assert "dup1" in result_ids  # highest relevance selected first
    assert "div" in result_ids   # diverse paper preferred over near-duplicate
    assert "dup2" not in result_ids


def test_mmr_pure_relevance_first_is_most_similar_to_query():
    """lam=1.0 ignores diversity — first pick must be the paper most similar to q."""
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    # p_close is nearest to q; p_far is furthest
    p_close = np.array([0.99, 0.14, 0.0, 0.0], dtype=np.float32)
    p_close /= np.linalg.norm(p_close)
    p_far = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    candidates = [("p_close", 1.0, p_close), ("p_far", 5.0, p_far)]
    result = mmr(q, candidates, lam=1.0, k=1)
    assert result[0][0] == "p_close"  # cosine sim wins over relevance score
