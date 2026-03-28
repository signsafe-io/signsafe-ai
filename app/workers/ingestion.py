"""Document ingestion worker: parses documents and stores embeddings.

Message format (from signsafe-api/internal/queue/rabbitmq.go):
    {
        "contractId": "<ULID>",
        "jobId":      "<ULID>",
        "filePath":   "<s3-key>"
    }

Processing pipeline:
    1. Receive message → confirm ingestion_job exists in DB
    2. Job status → parsing, progress 0%
    3. Download file from S3
    4. Parse into clauses
    5. Job status → chunking, progress 40%
    6. Batch-insert clauses to DB
    7. Job status → indexing, progress 70%
    8. Generate embeddings → upsert to Qdrant
    9. Job status → completed, progress 100%
   10. On failure → job status failed, store error message
"""

from __future__ import annotations

import asyncio
import functools
import os
import pathlib
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import asyncpg
import boto3
import structlog

from app.config import settings
from app.db import (
    get_org_id_for_contract,
    insert_clauses_batch,
    update_contract_status,
    update_ingestion_job,
)
from app.services import embeddings as emb_svc
from app.services import parser as parser_svc
from app.services import rag as rag_svc

log = structlog.get_logger()


def _make_s3_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name="us-east-1",
    )


def _download_file(file_path: str, dest: pathlib.Path) -> None:
    """Download a file from S3/SeaweedFS to dest."""
    s3 = _make_s3_client()
    s3.download_file(settings.s3_bucket, file_path, str(dest))


def _guess_suffix(file_path: str) -> str:
    """Return the file suffix from the S3 key."""
    _, ext = os.path.splitext(file_path)
    return ext.lower() if ext else ".pdf"


async def _process(pool: asyncpg.Pool, msg: dict[str, Any]) -> None:
    contract_id: str = msg["contractId"]
    job_id: str = msg["jobId"]
    file_path: str = msg["filePath"]

    log.info("ingestion started", job_id=job_id, contract_id=contract_id)

    # Step 1 → parsing
    await update_ingestion_job(
        pool, job_id, status="parsing", progress=0, current_step="파일 다운로드 중"
    )

    suffix = _guess_suffix(file_path)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name)

    try:
        # Offload blocking network I/O to a thread so the event loop
        # (shared with the analysis worker) is not blocked during download.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, functools.partial(_download_file, file_path, tmp_path)
        )
        log.info("file downloaded", path=str(tmp_path), size=tmp_path.stat().st_size)

        # Step 2 — parse (CPU-bound + sync C library; offload to thread pool
        # to avoid blocking the event loop shared with the analysis worker)
        clauses = await loop.run_in_executor(
            None, functools.partial(parser_svc.parse_sync, tmp_path)
        )
        log.info("parsing complete", clause_count=len(clauses))

        # Step 3 → chunking
        await update_ingestion_job(
            pool,
            job_id,
            status="chunking",
            progress=40,
            current_step="조항 분절 완료, DB 저장 중",
        )

        # Build clause dicts with ULID-style IDs.
        now_ts = datetime.now(timezone.utc)
        clause_dicts: list[dict[str, Any]] = []
        for idx, clause in enumerate(clauses):
            cid = str(uuid.uuid4()).replace("-", "")[:26]
            clause_dicts.append(
                {
                    "id": cid,
                    "clause_index": idx,
                    "label": clause.label,
                    "content": clause.text,
                    "page_start": clause.page_start,
                    "page_end": clause.page_end,
                    "anchor_x": clause.anchor.x if clause.anchor else None,
                    "anchor_y": clause.anchor.y if clause.anchor else None,
                    "anchor_width": clause.anchor.width if clause.anchor else None,
                    "anchor_height": clause.anchor.height if clause.anchor else None,
                    "start_offset": clause.start_offset,
                    "end_offset": clause.end_offset,
                }
            )

        await insert_clauses_batch(pool, contract_id, clause_dicts)
        log.info("clauses saved to DB", count=len(clause_dicts))

        # Step 4 → indexing
        await update_ingestion_job(
            pool,
            job_id,
            status="indexing",
            progress=70,
            current_step="임베딩 생성 및 Qdrant 저장 중",
        )

        # Ensure Qdrant collection exists.
        await rag_svc.ensure_collection()

        # Look up the organization ID so Qdrant payloads can be filtered per org.
        org_id = await get_org_id_for_contract(pool, contract_id)
        if org_id is None:
            log.warning(
                "org_id not found for contract — Qdrant points will have null org_id",
                contract_id=contract_id,
            )

        # Generate embeddings in batches.
        texts = [c["content"] for c in clause_dicts]
        vectors = await emb_svc.embed(texts)

        qdrant_points = [
            {
                "id": _clause_id_to_qdrant_id(c["id"]),
                "vector": vectors[i],
                "payload": {
                    "clause_id": c["id"],
                    "contract_id": contract_id,
                    "label": c.get("label"),
                    "content": c["content"][:500],  # truncated for RAG snippet display
                    "org_id": org_id,
                    "created_at": now_ts.isoformat(),
                    "created_at_ts": now_ts.timestamp(),
                },
            }
            for i, c in enumerate(clause_dicts)
        ]

        await rag_svc.upsert_clauses(qdrant_points)

        # Step 5 → completed
        await update_ingestion_job(
            pool, job_id, status="completed", progress=100, current_step="완료"
        )
        await update_contract_status(pool, contract_id, "ready")
        log.info("ingestion completed", job_id=job_id, contract_id=contract_id)

    finally:
        tmp_path.unlink(missing_ok=True)


def _clause_id_to_qdrant_id(clause_id: str) -> str:
    """Convert a 26-char alphanumeric ID to a UUID string for Qdrant.

    Qdrant requires point IDs to be unsigned integers or UUID strings.
    We pad/hash to produce a deterministic UUID.
    """
    # Use uuid5 with a fixed namespace for deterministic mapping.
    return str(uuid.uuid5(uuid.NAMESPACE_OID, clause_id))


def make_handler(
    pool: asyncpg.Pool,
) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """Return an async message handler bound to the given DB pool."""

    async def handler(msg: dict[str, Any]) -> None:
        job_id = msg.get("jobId", "unknown")
        contract_id = msg.get("contractId", "unknown")
        try:
            await _process(pool, msg)
        except Exception as exc:
            log.exception(
                "ingestion failed",
                job_id=job_id,
                contract_id=contract_id,
                error=str(exc),
            )
            try:
                await update_ingestion_job(
                    pool,
                    job_id,
                    status="failed",
                    progress=0,
                    error_message=str(exc),
                )
                await update_contract_status(pool, contract_id, "failed")
            except Exception:
                log.exception("failed to update job status to failed", job_id=job_id)
            raise  # re-raise so aio-pika nacks → DLQ

    return handler
