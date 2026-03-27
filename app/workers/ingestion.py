"""Document ingestion worker: receives upload events, parses documents, and stores embeddings."""
import structlog

log = structlog.get_logger()


async def handle_ingestion(message: dict) -> None:
    """Process a document ingestion message from the queue."""
    # TODO: implement ingestion pipeline
    #   1. download document from S3
    #   2. parse with parser service
    #   3. chunk text
    #   4. generate embeddings
    #   5. store in Qdrant
    log.info("ingestion worker: received message", message=message)
