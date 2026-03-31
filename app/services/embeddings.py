"""Embedding service: generates vector embeddings using OpenAI text-embedding-3-small."""

from __future__ import annotations

from typing import Sequence

import structlog
from openai import AsyncOpenAI

from app.config import settings

log = structlog.get_logger()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
_BATCH_SIZE = 100

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def embed(texts: Sequence[str]) -> list[list[float]]:
    """Return embedding vectors for the given texts.

    Processes texts in batches of up to 100 to respect API limits.
    """
    if not texts:
        return []

    client = _get_client()
    all_vectors: list[list[float]] = []

    text_list = list(texts)
    batches = [
        text_list[i : i + _BATCH_SIZE] for i in range(0, len(text_list), _BATCH_SIZE)
    ]

    for batch_idx, batch in enumerate(batches):
        log.info(
            "embedding batch",
            batch=batch_idx + 1,
            total_batches=len(batches),
            size=len(batch),
        )
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        vectors = [
            item.embedding for item in sorted(response.data, key=lambda x: x.index)
        ]
        all_vectors.extend(vectors)

    return all_vectors
