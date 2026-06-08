"""Unit tests for Rocchio relevance feedback."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retrieval.rocchio import rocchio


def test_output_is_unit_normalised():
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    rel = [np.array([0, 1, 0, 0], dtype=np.float32)]
    result = rocchio(q, rel, [])
    assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-5


def test_moves_closer_to_relevant():
    """Updated query should have a higher y-component after a relevant paper in y."""
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    rel = [np.array([0, 1, 0, 0], dtype=np.float32)]
    result = rocchio(q, rel, [])
    assert result[1] > 0.0, "Query should shift toward the relevant direction"


def test_moves_away_from_non_relevant():
    """Updated query should have a lower y-component after a non-relevant paper in y."""
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    nrel = [np.array([0, 1, 0, 0], dtype=np.float32)]
    result = rocchio(q, [], nrel)
    assert result[1] < 0.0, "Query should shift away from the non-relevant direction"


def test_empty_feedback_returns_normalised_query():
    """With no feedback, output should be the normalised original query."""
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    result = rocchio(q, [], [])
    np.testing.assert_array_almost_equal(result, q)
    assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-5


def test_multiple_relevant_papers():
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    rel = [
        np.array([0, 1, 0, 0], dtype=np.float32),
        np.array([0, 0, 1, 0], dtype=np.float32),
    ]
    result = rocchio(q, rel, [])
    assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-5
    # Should have moved in both y and z direction
    assert result[1] > 0 and result[2] > 0
