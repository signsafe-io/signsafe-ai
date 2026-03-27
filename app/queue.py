"""Message queue connection and helpers using aio-pika (RabbitMQ)."""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import aio_pika
import structlog

from app.config import settings

log = structlog.get_logger()

# Queue names — must match signsafe-api declarations
INGESTION_QUEUE = "ingestion.jobs"
ANALYSIS_QUEUE = "analysis.jobs"


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


async def consume(
    connection: aio_pika.abc.AbstractRobustConnection,
    queue_name: str,
    handler: Callable[[dict[str, Any]], Awaitable[None]],
) -> None:
    """Consume messages from queue_name, calling handler for each message.

    On handler success the message is acked.
    On any exception the message is nacked (requeue=False) so it goes to DLQ.
    """
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await _declare_queue_with_dlq(channel, queue_name)

    log.info("starting consumer", queue=queue_name)

    async with queue.iterator() as messages:
        async for message in messages:
            async with message.process(requeue=False):
                try:
                    body = json.loads(message.body)
                    await handler(body)
                except Exception:
                    log.exception("message processing failed", queue=queue_name)
                    raise  # aio-pika nacks and routes to DLQ
