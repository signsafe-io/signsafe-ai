"""Database connection using asyncpg."""
import asyncpg

from app.config import settings


async def connect_db() -> asyncpg.Pool:
    """Create and return an asyncpg connection pool."""
    pool = await asyncpg.create_pool(settings.database_url)
    return pool
