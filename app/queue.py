"""Message queue connection using aio-pika (RabbitMQ)."""
import aio_pika

from app.config import settings


async def connect_queue() -> aio_pika.abc.AbstractRobustConnection:
    """Create and return a robust RabbitMQ connection."""
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    return connection
