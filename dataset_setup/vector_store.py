import uuid

from qdrant_client.models import Distance, PointStruct, VectorParams

from config import COLLECTION, EMBEDDING_DIM, qdrant
from llm import get_embedding


def init_collection() -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION in existing:
        return
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def search_similar(query: str, limit: int = 3) -> list[dict]:
    emb = get_embedding(query)
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=emb,
        limit=limit,
    )
    return [hit.payload for hit in hits]


def _to_uuid(source_id: str, ref_type: str) -> str:
    """Deterministic UUID from source_id — prevents duplicate upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{ref_type}:{source_id}"))


def save_case(case: dict, structured: str) -> None:
    ref_type = case.get("type", "prec")
    source_id = case.get("source_id") or case.get("prec_seq") or str(uuid.uuid4())
    emb = get_embedding(structured)
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=_to_uuid(source_id, ref_type),
                vector=emb,
                payload={
                    "type": ref_type,
                    "source_id": source_id,
                    "title": case.get("title", ""),
                    "content": case.get("content", "")[:1000],
                    "structured": structured,
                    "date": case.get("date", ""),
                    "court": case.get("court", ""),
                },
            )
        ],
    )
