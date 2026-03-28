import asyncio
import signal
import sys

import structlog

from app.config import settings
from app.db import connect_db
from app.queue import (
    ANALYSIS_DLQ,
    ANALYSIS_QUEUE,
    INGESTION_DLQ,
    INGESTION_QUEUE,
    connect_queue,
    consume,
    consume_dlq,
)

log = structlog.get_logger()


def _validate_config() -> None:
    """Fail fast if required API keys are missing."""
    missing: list[str] = []
    if not settings.anthropic_api_key:
        missing.append("ANTHROPIC_API_KEY")
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if missing:
        log.error(
            "required API keys are not set — cannot start worker",
            missing=missing,
        )
        sys.exit(1)


async def main() -> None:
    _validate_config()
    log.info("signsafe-ai worker starting", env=settings.env)

    db = await connect_db()
    queue_conn = await connect_queue()

    # Import workers here to avoid circular imports at module level.
    from app.workers.analysis import make_handler as make_analysis_handler
    from app.workers.ingestion import make_handler as make_ingestion_handler

    ingestion_handler = make_ingestion_handler(db)
    analysis_handler = make_analysis_handler(db)

    log.info("connections established, starting workers")

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _request_shutdown(sig: int) -> None:
        log.info("shutdown signal received, stopping workers", signal=sig)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _request_shutdown, sig)

    # Run main consumers and DLQ monitors until a shutdown signal is received.
    # asyncio.gather is cancelled when shutdown_event fires so that
    # in-flight aio-pika message handlers finish naturally (aio-pika
    # drains pending acks before closing the channel).
    gather_task = asyncio.ensure_future(
        asyncio.gather(
            consume(queue_conn, INGESTION_QUEUE, ingestion_handler),
            consume(queue_conn, ANALYSIS_QUEUE, analysis_handler),
            consume_dlq(queue_conn, INGESTION_DLQ),
            consume_dlq(queue_conn, ANALYSIS_DLQ),
        )
    )

    try:
        # Block until either a signal fires or the consumers exit on their own.
        done, _ = await asyncio.wait(
            {gather_task, asyncio.ensure_future(shutdown_event.wait())},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if shutdown_event.is_set() and not gather_task.done():
            gather_task.cancel()
            try:
                await gather_task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        await queue_conn.close()
        await db.close()
        log.info("worker shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())
