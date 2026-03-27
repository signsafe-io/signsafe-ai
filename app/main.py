import asyncio
import structlog

from app.config import settings
from app.queue import connect_queue
from app.db import connect_db

log = structlog.get_logger()


async def main() -> None:
    log.info("signsafe-ai worker starting", env=settings.env)

    db = await connect_db()
    queue = await connect_queue()

    log.info("connections established, waiting for messages")

    try:
        # TODO: register workers and start consuming
        await asyncio.Event().wait()
    finally:
        await queue.close()
        await db.close()
        log.info("worker shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())
