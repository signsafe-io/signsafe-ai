import asyncio
import signal
import sys
from datetime import datetime, timedelta, timezone

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

_KST = timezone(timedelta(hours=9))


def _seconds_until_next_monday_4am() -> float:
    """Return seconds until next Monday 04:00 KST."""
    now = datetime.now(_KST)
    days_ahead = -now.weekday()  # Monday = 0
    if days_ahead < 0 or (days_ahead == 0 and now.hour >= 4):
        days_ahead += 7
    next_run = (now + timedelta(days=days_ahead)).replace(
        hour=4, minute=0, second=0, microsecond=0
    )
    return (next_run - now).total_seconds()


async def _run_weekly_legal_update() -> None:
    """Background task: update legal vector DB on startup and every Monday 04:00 KST."""
    from app.services.legal_updater import run_update

    # 시작 시 즉시 1회 실행 — cases 컬렉션 초기 데이터 확보
    log.info("running initial legal data update")
    try:
        await run_update()
    except Exception:
        log.exception("initial legal update failed — will retry next Monday")

    while True:
        wait = _seconds_until_next_monday_4am()
        next_run = datetime.now(_KST) + timedelta(seconds=wait)
        log.info(
            "legal update scheduled",
            next_run=next_run.strftime("%Y-%m-%d %H:%M KST"),
            wait_hours=round(wait / 3600, 1),
        )
        await asyncio.sleep(wait)
        try:
            await run_update()
        except Exception:
            log.exception("weekly legal update failed — will retry next Monday")


def _validate_config() -> None:
    """Fail fast if required API keys are missing."""
    missing: list[str] = []
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

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _request_shutdown, sig)
    except NotImplementedError:
        # Windows does not support add_signal_handler; fall back to signal.signal.
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda s, _: _request_shutdown(s))

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
            _run_weekly_legal_update(),
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
