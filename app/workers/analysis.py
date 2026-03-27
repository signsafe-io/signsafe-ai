"""Contract analysis worker: runs RAG + LLM pipeline and writes results to DB."""
import structlog

log = structlog.get_logger()


async def handle_analysis(message: dict) -> None:
    """Process a contract analysis message from the queue."""
    # TODO: implement analysis pipeline
    #   1. retrieve relevant chunks from Qdrant
    #   2. build prompt with retrieved context
    #   3. call LLM service
    #   4. parse and store results in DB
    log.info("analysis worker: received message", message=message)
