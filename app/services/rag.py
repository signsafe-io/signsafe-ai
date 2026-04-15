"""RAG service: stores and retrieves clause embeddings from Qdrant."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import settings
from app.services.embeddings import EMBEDDING_DIM, embed

log = structlog.get_logger()

COLLECTION_NAME = "clauses"
CASES_COLLECTION_NAME = "cases"

_client: AsyncQdrantClient | None = None


def _get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url)
    return _client


async def _ensure_collection(name: str) -> None:
    client = _get_client()
    collections = await client.get_collections()
    existing = {c.name for c in collections.collections}
    if name not in existing:
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        log.info("qdrant collection created", collection=name)
    else:
        log.info("qdrant collection exists", collection=name)


async def ensure_collection() -> None:
    """Create the clauses collection if it does not already exist."""
    await _ensure_collection(COLLECTION_NAME)


async def ensure_cases_collection() -> None:
    """Create the cases collection if it does not already exist."""
    await _ensure_collection(CASES_COLLECTION_NAME)


async def upsert_clauses(
    points: list[dict[str, Any]],
) -> None:
    """Upsert clause vectors into Qdrant.

    Each point must have:
        id (str UUID or int),
        vector (list[float]),
        payload: {
            contract_id, clause_id, label, org_id, created_at (ISO str)
        }
    """
    client = _get_client()
    qdrant_points = [
        PointStruct(
            id=p["id"],
            vector=p["vector"],
            payload=p["payload"],
        )
        for p in points
    ]

    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=qdrant_points,
    )
    log.info("upserted clause vectors", count=len(qdrant_points))


async def search_similar_clauses(
    query_text: str,
    top_k: int = 3,
    org_id: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[dict[str, Any]]:
    """Search Qdrant for clauses similar to query_text.

    Returns a list of dicts with keys: clause_id, contract_id, label,
    score, payload.
    """
    client = _get_client()
    vectors = await embed([query_text])
    query_vector = vectors[0]

    # Build optional filters.
    must: list[Any] = []

    if org_id:
        must.append(
            FieldCondition(
                key="org_id",
                match=MatchValue(value=org_id),
            )
        )

    if date_from or date_to:
        range_kwargs: dict[str, Any] = {}
        if date_from:
            range_kwargs["gte"] = date_from.timestamp()
        if date_to:
            range_kwargs["lte"] = date_to.timestamp()
        must.append(
            FieldCondition(
                key="created_at_ts",
                range=Range(**range_kwargs),
            )
        )

    query_filter = Filter(must=must) if must else None

    response = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {
            "clause_id": r.payload.get("clause_id"),
            "contract_id": r.payload.get("contract_id"),
            "label": r.payload.get("label"),
            "score": r.score,
            "payload": r.payload,
        }
        for r in response.points
    ]


async def search_legal_references(
    query_text: str,
    top_k: int = 3,
    ref_type: str | None = None,
) -> list[dict[str, Any]]:
    """Search Qdrant cases collection for relevant 판례/법령.

    Args:
        query_text: clause text to search against.
        top_k: number of results to return.
        ref_type: optional filter — "prec" for 판례, "law" for 법령, None for both.
    """
    client = _get_client()

    # Silently skip if the collection doesn't exist yet.
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

    response = await client.query_points(
        collection_name=CASES_COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
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
    ]
