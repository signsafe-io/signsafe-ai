"""Message queue connection and helpers using aio-pika (RabbitMQ)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import aio_pika
import structlog

from app.config import settings
from app.errors import PermanentError, RetryableError

log = structlog.get_logger()

# Queue names — must match signsafe-api declarations
INGESTION_QUEUE = "ingestion.jobs"
ANALYSIS_QUEUE = "analysis.jobs"

INGESTION_DLQ = f"{INGESTION_QUEUE}.dlq"
ANALYSIS_DLQ = f"{ANALYSIS_QUEUE}.dlq"

# Retry configuration for RetryableError
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds
_RETRY_MAX_DELAY = 30.0  # seconds


async def connect_queue() -> aio_pika.abc.AbstractRobustConnection:
    """Create and return a robust RabbitMQ connection."""
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    return connection


async def _declare_queue_with_dlq(
    channel: aio_pika.abc.AbstractChannel,
    name: str,
) -> aio_pika.abc.AbstractQueue:
    """Declare a durable queue with a dead-letter queue, mirroring signsafe-api setup."""
    dlq_name = f"{name}.dlq"

    # Declare DLQ first (plain durable queue).
    await channel.declare_queue(dlq_name, durable=True)

    # Declare main queue with dead-letter routing to DLQ.
    queue = await channel.declare_queue(
        name,
        durable=True,
        arguments={
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": dlq_name,
        },
    )
    return queue


async def consume_dlq(
    connection: aio_pika.abc.AbstractRobustConnection,
    dlq_name: str,
) -> None:
    """Consume messages from a DLQ and log each one as an error.

    DLQ messages represent processing failures. This consumer ensures they are
    never silently accumulated — every entry is logged so operators can triage.
    """
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    # DLQ is a plain durable queue (no dead-letter on the DLQ itself).
    queue = await channel.declare_queue(dlq_name, durable=True)

    log.info("starting DLQ consumer", queue=dlq_name)

    async with queue.iterator() as messages:
        async for message in messages:
            async with message.process(requeue=False):
                try:
                    body_str = message.body.decode(errors="replace")
                    headers = dict(message.headers or {})
                    log.error(
                        "DLQ message received — manual intervention required",
                        queue=dlq_name,
                        body=body_str[:500],
                        headers=headers,
                        message_id=message.message_id,
                    )
                except Exception:
                    log.exception("failed to log DLQ message", queue=dlq_name)


async def consume(
    connection: aio_pika.abc.AbstractRobustConnection,
    queue_name: str,
    handler: Callable[[dict[str, Any]], Awaitable[None]],
) -> None:
    """Consume messages from queue_name, calling handler for each message.

    Error handling strategy:
    - PermanentError: ack the message (no retry, no DLQ). The handler is
      responsible for recording failure in the DB before raising.
    - RetryableError: retry up to _MAX_RETRIES times with exponential back-off.
      If all retries are exhausted, nack → DLQ.
    - Any other Exception: treated as RetryableError (unknown errors may be
      transient) — same retry/DLQ path.
    """
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await _declare_queue_with_dlq(channel, queue_name)

    log.info("starting consumer", queue=queue_name)

    async with queue.iterator() as messages:
        async for message in messages:
            # Use manual ack/nack so we can control DLQ routing precisely.
            try:
                body = json.loads(message.body)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                log.error(
                    "unparseable message — acking to prevent DLQ loop",
                    queue=queue_name,
                    error=str(exc),
                )
                try:
                    await message.ack()
                except Exception:
                    log.exception("failed to ack unparseable message", queue=queue_name)
                continue

            last_exc: BaseException | None = None
            try:
                for attempt in range(1, _MAX_RETRIES + 1):
                    try:
                        await handler(body)
                        last_exc = None
                        break  # success
                    except PermanentError as exc:
                        log.error(
                            "permanent error — acking message without DLQ",
                            queue=queue_name,
                            attempt=attempt,
                            error=str(exc),
                        )
                        last_exc = exc
                        break  # do not retry permanent errors
                    except (RetryableError, Exception) as exc:
                        last_exc = exc
                        if attempt < _MAX_RETRIES:
                            delay = min(
                                _RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                                _RETRY_MAX_DELAY,
                            )
                            log.warning(
                                "retryable error — will retry",
                                queue=queue_name,
                                attempt=attempt,
                                max_retries=_MAX_RETRIES,
                                delay=delay,
                                error=str(exc),
                            )
                            await asyncio.sleep(delay)
                        else:
                            log.error(
                                "all retries exhausted — routing to DLQ",
                                queue=queue_name,
                                max_retries=_MAX_RETRIES,
                                error=str(exc),
                            )
            except BaseException as exc:
                # CancelledError, KeyboardInterrupt 등 — nack with requeue so
                # RabbitMQ re-queues the message and prevents silent message loss.
                last_exc = exc
                raise
            finally:
                # ack/nack은 BaseException(CancelledError 포함)이 발생하더라도
                # 반드시 시도한다. ack/nack 자체가 실패해도 예외를 전파하지 않는다.
                # connection 종료 시 RabbitMQ가 unacked 메시지를 자동으로 재큐잉한다.
                try:
                    if last_exc is None:
                        # Success
                        await message.ack()
                    elif isinstance(last_exc, PermanentError):
                        # Permanent failure: ack so it does NOT go to DLQ.
                        # The handler has already written the failure to the DB.
                        await message.ack()
                    else:
                        # Retryable failure exhausted all attempts, or BaseException:
                        # nack → DLQ (requeue=False sends to dead-letter queue).
                        await message.nack(requeue=False)
                except Exception:
                    log.exception(
                        "failed to ack/nack message — message may be requeued on connection close",
                        queue=queue_name,
                    )
