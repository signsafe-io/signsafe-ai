"""Embedding service: generates vector embeddings for text chunks."""
from typing import Sequence


async def embed(texts: Sequence[str]) -> list[list[float]]:
    """Return embedding vectors for the given texts."""
    # TODO: call OpenAI embeddings API
    raise NotImplementedError
