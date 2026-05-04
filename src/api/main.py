"""
Recommendation API service.

Serves real-time recommendations and semantic search over the product catalog.
Deployed as containerized service on ECS Fargate behind an Application Load Balancer.
"""

from contextlib import asynccontextmanager
from typing import List

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.recommender import HybridRecommender
from src.models.semantic_search import SemanticSearchEngine
from src.utils.config import settings

logger = structlog.get_logger()

# Global model instances loaded at startup
recommender: HybridRecommender | None = None
search_engine: SemanticSearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    global recommender, search_engine
    logger.info("loading_models", region=settings.aws_region)

    recommender = HybridRecommender.load_from_s3(
        bucket=settings.model_bucket,
        key="recommender/latest.pkl",
    )
    search_engine = SemanticSearchEngine(
        opensearch_endpoint=settings.opensearch_endpoint,
        index_name="products",
    )

    logger.info("models_loaded")
    yield
    logger.info("shutting_down")


app = FastAPI(
    title="E-Commerce Recommendation API",
    version="1.0.0",
    description="Hybrid recommendation + semantic search service",
    lifespan=lifespan,
)


class RecommendRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    num_recommendations: int = Field(10, ge=1, le=50)
    context: dict | None = None


class Product(BaseModel):
    product_id: str
    title: str
    score: float
    category: str | None = None


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[Product]
    model_version: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(20, ge=1, le=100)


@app.get("/health")
async def health() -> dict:
    """Liveness probe used by ECS and ALB."""
    return {"status": "ok", "version": app.version}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    """Return personalized product recommendations for a user."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        items = recommender.predict(
            user_id=req.user_id,
            k=req.num_recommendations,
            context=req.context,
        )
    except Exception as exc:
        logger.exception("recommendation_failed", user_id=req.user_id)
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    return RecommendResponse(
        user_id=req.user_id,
        recommendations=[Product(**i) for i in items],
        model_version=recommender.version,
    )


@app.post("/search")
async def search(req: SearchRequest) -> dict:
    """Semantic search over the product catalog."""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    results = search_engine.query(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}
