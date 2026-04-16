"""RAG service: retrieves 판례/법령 from Qdrant cases collection."""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from app.config import settings
from app.services.embeddings import EMBEDDING_DIM, embed

log = structlog.get_logger()

CASES_COLLECTION_NAME = "cases"

_client: AsyncQdrantClient | None = None


def _get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url)
    return _client


async def ensure_cases_collection() -> None:
    """Create the cases collection if it does not already exist."""
    client = _get_client()
    collections = await client.get_collections()
    existing = {c.name for c in collections.collections}
    if CASES_COLLECTION_NAME not in existing:
        await client.create_collection(
            collection_name=CASES_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        log.info("qdrant collection created", collection=CASES_COLLECTION_NAME)
    else:
        log.info("qdrant collection exists", collection=CASES_COLLECTION_NAME)


_MIN_SCORE = 0.42  # 코사인 유사도 최소값 — 이 미만은 무관한 문서로 간주하고 제외


async def search_legal_references(
    query_text: str,
    top_k: int = 3,
    ref_type: str | None = None,
) -> list[dict[str, Any]]:
    """Search Qdrant cases collection for relevant 판례/법령.

    Args:
        query_text: focused search query — use legal terminology, not clause text.
        top_k: number of results to return (before score filtering).
        ref_type: optional filter — "prec" for 판례, "law" for 법령, None for both.

    Returns only results with cosine similarity >= _MIN_SCORE to avoid returning
    irrelevant documents when no good match exists.
    """
    client = _get_client()

    collections = await client.get_collections()
    existing = {c.name for c in collections.collections}
    if CASES_COLLECTION_NAME not in existing:
        return []

    vectors = await embed([query_text])
    query_vector = vectors[0]

    must: list[Any] = []
    if ref_type:
        must.append(FieldCondition(key="type", match=MatchValue(value=ref_type)))
    query_filter = Filter(must=must) if must else None

    # score_threshold를 서버에 전달하되, 클라이언트에서도 재검증한다.
    # Qdrant 서버 버전에 따라 query_points()의 score_threshold 파라미터가
    # 무시되는 경우가 있어 클라이언트 측 필터링을 안전망으로 추가한다.
    response = await client.query_points(
        collection_name=CASES_COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        score_threshold=_MIN_SCORE,
    )

    results = [
        {
            "type": r.payload.get("type", "prec"),
            "source_id": r.payload.get("source_id", ""),
            "title": r.payload.get("title", ""),
            "content": r.payload.get("content", ""),
            "date": r.payload.get("date", ""),
            "court": r.payload.get("court", ""),
            "score": r.score,
        }
        for r in response.points
        if r.score >= _MIN_SCORE  # client-side safety net
    ]

    log.info(
        "rag search complete",
        query_preview=query_text[:80],
        returned=len(results),
        top_score=results[0]["score"] if results else None,
    )
    return results
