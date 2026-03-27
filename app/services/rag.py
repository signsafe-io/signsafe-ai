"""RAG service: retrieves relevant document chunks from Qdrant."""


async def retrieve(query: str, collection: str, top_k: int = 5) -> list[str]:
    """Return the top-k most relevant text chunks for a query."""
    # TODO: embed query and search Qdrant
    raise NotImplementedError
