"""
Semantic search over the product catalog using Sentence-Transformer embeddings
indexed in AWS OpenSearch with k-NN search.
"""

from __future__ import annotations

from typing import Any

import structlog
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


class SemanticSearchEngine:
    """k-NN semantic search backed by OpenSearch."""

    def __init__(
        self,
        opensearch_endpoint: str,
        index_name: str = "products",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.index_name = index_name
        self.encoder = SentenceTransformer(model_name)
        self.client = OpenSearch(
            hosts=[opensearch_endpoint],
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

    def query(self, text: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Run a semantic k-NN query."""
        embedding = self.encoder.encode(text, normalize_embeddings=True).tolist()

        body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": top_k,
                    }
                }
            },
        }

        response = self.client.search(index=self.index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        return [
            {
                "product_id": h["_source"]["product_id"],
                "title": h["_source"]["title"],
                "score": h["_score"],
            }
            for h in hits
        ]

    def index_product(self, product: dict[str, Any]) -> None:
        """Index or update a single product document."""
        embedding = self.encoder.encode(
            f"{product['title']} {product.get('description', '')}",
            normalize_embeddings=True,
        ).tolist()

        doc = {**product, "embedding": embedding}
        self.client.index(index=self.index_name, id=product["product_id"], body=doc)
