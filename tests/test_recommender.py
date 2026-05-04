"""Unit tests for the hybrid recommender."""

import numpy as np
import pytest

from src.models.recommender import HybridRecommender


@pytest.fixture
def recommender() -> HybridRecommender:
    """Build a tiny in-memory recommender for testing."""
    rng = np.random.RandomState(0)
    return HybridRecommender(
        user_factors=rng.rand(3, 8),
        item_factors=rng.rand(5, 8),
        item_embeddings=rng.rand(5, 8),
        item_metadata={
            f"p_{i}": {"title": f"Product {i}", "category": "books"}
            for i in range(5)
        },
        user_index={"u_a": 0, "u_b": 1, "u_c": 2},
        item_index={f"p_{i}": i for i in range(5)},
    )


def test_returns_correct_count(recommender: HybridRecommender) -> None:
    recs = recommender.predict("u_a", k=3)
    assert len(recs) == 3


def test_returns_sorted_by_score(recommender: HybridRecommender) -> None:
    recs = recommender.predict("u_b", k=5)
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_cold_start_user(recommender: HybridRecommender) -> None:
    """Unknown user should still return recommendations via content fallback."""
    recs = recommender.predict("unknown_user", k=3, context={"recently_viewed": ["p_0"]})
    assert len(recs) == 3
    assert all("product_id" in r for r in recs)


def test_each_recommendation_has_required_fields(recommender: HybridRecommender) -> None:
    recs = recommender.predict("u_a", k=2)
    for rec in recs:
        assert "product_id" in rec
        assert "title" in rec
        assert "score" in rec
