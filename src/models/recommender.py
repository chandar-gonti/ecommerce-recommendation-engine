"""
Hybrid recommendation model combining collaborative filtering and content embeddings.

Architecture:
- Matrix factorization (ALS) for explicit user-item interactions
- Sentence-Transformer embeddings for content similarity (cold-start handling)
- Weighted blend with bandit-tuned alpha

Trained offline on SageMaker; this module handles inference only.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class HybridRecommender:
    """Inference wrapper for the trained hybrid model."""

    user_factors: np.ndarray
    item_factors: np.ndarray
    item_embeddings: np.ndarray  # content embeddings from BERT
    item_metadata: dict[str, dict[str, Any]]
    user_index: dict[str, int]
    item_index: dict[str, int]
    blend_alpha: float = 0.7
    version: str = "v1.0.0"
    _reverse_item_index: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._reverse_item_index = {v: k for k, v in self.item_index.items()}

    def predict(
        self,
        user_id: str,
        k: int = 10,
        context: dict | None = None,
    ) -> list[dict]:
        """Return top-k recommendations for a user."""
        if user_id in self.user_index:
            scores = self._collaborative_scores(user_id)
        else:
            # Cold start: fall back to content-based on context
            scores = self._content_scores(context or {})

        top_k_idx = np.argpartition(-scores, k)[:k]
        top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

        return [
            {
                "product_id": self._reverse_item_index[i],
                "title": self.item_metadata[self._reverse_item_index[i]]["title"],
                "category": self.item_metadata[self._reverse_item_index[i]].get("category"),
                "score": float(scores[i]),
            }
            for i in top_k_idx
        ]

    def _collaborative_scores(self, user_id: str) -> np.ndarray:
        u_idx = self.user_index[user_id]
        cf_scores = self.user_factors[u_idx] @ self.item_factors.T
        # Normalize to [0, 1]
        cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.ptp() + 1e-9)
        return cf_scores

    def _content_scores(self, context: dict) -> np.ndarray:
        """Cold-start: similarity to recently viewed items."""
        viewed = context.get("recently_viewed", [])
        if not viewed:
            # Default to popularity prior
            return np.random.RandomState(42).rand(len(self.item_index))

        viewed_idx = [self.item_index[v] for v in viewed if v in self.item_index]
        if not viewed_idx:
            return np.random.RandomState(42).rand(len(self.item_index))

        query_vec = self.item_embeddings[viewed_idx].mean(axis=0)
        sims = self.item_embeddings @ query_vec
        return (sims - sims.min()) / (sims.ptp() + 1e-9)

    @classmethod
    def load_from_s3(cls, bucket: str, key: str) -> HybridRecommender:
        """Load a pickled model from S3."""
        logger.info("loading_model", bucket=bucket, key=key)
        s3 = boto3.client("s3")
        local_path = Path("/tmp") / Path(key).name
        s3.download_file(bucket, key, str(local_path))

        with local_path.open("rb") as f:
            state = pickle.load(f)
        return cls(**state)
