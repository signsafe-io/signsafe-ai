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

Error classification:
    PermanentError — missing required fields, unsupported file type, PDF parse
        errors. Message is acked; no DLQ routing.
    RetryableError — S3 download failures, DB connection errors, Qdrant
        unavailability. Consumer retries with exponential back-off.
"""

from __future__ import annotations

import asyncio
import functools
import pathlib
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import asyncpg
import boto3
import structlog
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings
from app.db import (
    get_org_id_for_contract,
    insert_clauses_batch,
    update_contract_status,
    update_ingestion_job,
)
from app.errors import PermanentError, RetryableError
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
    ext = pathlib.Path(file_path).suffix.lower()
    return ext if ext else ".pdf"


def _clause_id_to_qdrant_id(clause_id: str) -> str:
    """Convert a 26-char alphanumeric ID to a UUID string for Qdrant.

    Qdrant requires point IDs to be unsigned integers or UUID strings.
    We pad/hash to produce a deterministic UUID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, clause_id))


async def _download_and_parse(
    loop: asyncio.AbstractEventLoop,
    file_path: str,
    tmp_path: pathlib.Path,
) -> list[Any]:
    """Download file from S3 and parse it into raw clauses."""
    try:
        await loop.run_in_executor(
            None, functools.partial(_download_file, file_path, tmp_path)
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in ("NoSuchKey", "404", "AccessDenied", "403"):
            raise PermanentError(
                f"S3 object not accessible: {file_path} ({error_code})"
            ) from exc
        raise RetryableError(f"S3 download failed (retryable): {exc}") from exc
    except BotoCoreError as exc:
        raise RetryableError(f"S3 connection error: {exc}") from exc

    log.info("file downloaded", path=str(tmp_path), size=tmp_path.stat().st_size)

    # CPU-bound + sync C library; offload to thread pool to avoid blocking
    # the event loop shared with the analysis worker.
    try:
        clauses = await loop.run_in_executor(
            None, functools.partial(parser_svc.parse_sync, tmp_path)
        )
    except Exception as exc:
        # Parser errors are permanent — the file content won't change on retry.
        raise PermanentError(f"Document parse error: {exc}") from exc

    log.info("parsing complete", clause_count=len(clauses))
    return clauses


def _build_clause_dicts(
    clauses: list[Any],
    created_at: datetime,
) -> list[dict[str, Any]]:
    """Convert parsed Clause objects to DB-ready dicts with generated IDs."""
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
    return clause_dicts


def _build_qdrant_points(
    clause_dicts: list[dict[str, Any]],
    vectors: list[list[float]],
    contract_id: str,
    org_id: str | None,
    created_at: datetime,
) -> list[dict[str, Any]]:
    """Assemble Qdrant point dicts from clause dicts and embedding vectors."""
    return [
        {
            "id": _clause_id_to_qdrant_id(c["id"]),
            "vector": vectors[i],
            "payload": {
                "clause_id": c["id"],
                "contract_id": contract_id,
                "label": c.get("label"),
                "content": c["content"][:500],  # truncated for RAG snippet display
                "org_id": org_id,
                "created_at": created_at.isoformat(),
                "created_at_ts": created_at.timestamp(),
            },
        }
        for i, c in enumerate(clause_dicts)
    ]


async def _process(pool: asyncpg.Pool, msg: dict[str, Any]) -> None:
    # Validate required fields — raise PermanentError if missing
    try:
        contract_id: str = msg["contractId"]
        job_id: str = msg["jobId"]
        file_path: str = msg["filePath"]
    except KeyError as exc:
        raise PermanentError(f"Missing required message field: {exc}") from exc

    log.info("ingestion started", job_id=job_id, contract_id=contract_id)

    # Step 1 → parsing
    try:
        await update_ingestion_job(
            pool, job_id, status="parsing", progress=0, current_step="파일 다운로드 중"
        )
    except asyncpg.PostgresConnectionError as exc:
        raise RetryableError(f"DB connection error: {exc}") from exc

    suffix = _guess_suffix(file_path)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name)

    try:
        loop = asyncio.get_running_loop()
        clauses = await _download_and_parse(loop, file_path, tmp_path)

        # Step 2 → chunking
        await update_ingestion_job(
            pool,
            job_id,
            status="chunking",
            progress=40,
            current_step="조항 분절 완료, DB 저장 중",
        )

        now_ts = datetime.now(timezone.utc)
        clause_dicts = _build_clause_dicts(clauses, now_ts)

        try:
            await insert_clauses_batch(pool, contract_id, clause_dicts)
        except asyncpg.PostgresConnectionError as exc:
            raise RetryableError(
                f"DB connection error inserting clauses: {exc}"
            ) from exc

        log.info("clauses saved to DB", count=len(clause_dicts))

        # Step 3 → indexing
        await update_ingestion_job(
            pool,
            job_id,
            status="indexing",
            progress=70,
            current_step="임베딩 생성 및 Qdrant 저장 중",
        )

        await rag_svc.ensure_collection()

        try:
            org_id = await get_org_id_for_contract(pool, contract_id)
        except asyncpg.PostgresConnectionError as exc:
            raise RetryableError(f"DB connection error fetching org_id: {exc}") from exc

        if org_id is None:
            log.warning(
                "org_id not found for contract — Qdrant points will have null org_id",
                contract_id=contract_id,
            )

        texts = [c["content"] for c in clause_dicts]
        vectors = await emb_svc.embed(texts)
        qdrant_points = _build_qdrant_points(
            clause_dicts, vectors, contract_id, org_id, now_ts
        )

        await rag_svc.upsert_clauses(qdrant_points)

        # Step 4 → completed
        await update_ingestion_job(
            pool, job_id, status="completed", progress=100, current_step="완료"
        )
        await update_contract_status(pool, contract_id, "ready")
        log.info("ingestion completed", job_id=job_id, contract_id=contract_id)

    finally:
        tmp_path.unlink(missing_ok=True)


def make_handler(
    pool: asyncpg.Pool,
) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """Return an async message handler bound to the given DB pool."""

    async def handler(msg: dict[str, Any]) -> None:
        job_id = msg.get("jobId", "unknown")
        contract_id = msg.get("contractId", "unknown")
        try:
            await _process(pool, msg)
        except PermanentError as exc:
            log.error(
                "permanent ingestion failure — marking failed, acking message (no DLQ)",
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
            raise  # re-raise PermanentError; queue.consume acks without DLQ
        except (RetryableError, Exception) as exc:
            log.error(
                "retryable ingestion failure — will retry or route to DLQ",
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
            raise  # re-raise; queue.consume retries or nacks → DLQ

    return handler
