import asyncio

import structlog

from app.config import settings
from app.db import connect_db
from app.queue import ANALYSIS_QUEUE, INGESTION_QUEUE, connect_queue, consume

log = structlog.get_logger()


async def main() -> None:
    log.info("signsafe-ai worker starting", env=settings.env)

    db = await connect_db()
    queue_conn = await connect_queue()

    # Import workers here to avoid circular imports at module level.
    from app.workers.analysis import make_handler as make_analysis_handler
    from app.workers.ingestion import make_handler as make_ingestion_handler

    ingestion_handler = make_ingestion_handler(db)
    analysis_handler = make_analysis_handler(db)

    log.info("connections established, starting workers")

    try:
        await asyncio.gather(
            consume(queue_conn, INGESTION_QUEUE, ingestion_handler),
            consume(queue_conn, ANALYSIS_QUEUE, analysis_handler),
        )
    finally:
        await queue_conn.close()
        await db.close()
        log.info("worker shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())
