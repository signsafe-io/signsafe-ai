"""Document ingestion worker: parses documents and stores clauses in DB.

Message format (from signsafe-api/internal/queue/rabbitmq.go):
    {
        "contractId": "<ULID>",
        "jobId":      "<ULID>",
        "filePath":   "<s3-key>"
    }

Processing pipeline:
    1. Job status → parsing, progress 0%
    2. Download file from S3, extract paragraphs
    3. LLM-based clause boundary detection (regex fallback on failure)
    4. Job status → chunking, progress 60%
    5. Batch-insert clauses to DB
    6. Job status → completed, progress 100%
    7. On failure → job status failed, store error message

Error classification:
    PermanentError — missing required fields, unsupported file type, PDF parse
        errors. Message is acked; no DLQ routing.
    RetryableError — S3 download failures, DB connection errors. Consumer
        retries with exponential back-off.
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
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings
from app.db import (
    insert_clauses_batch,
    update_contract_status,
    update_ingestion_job,
)
from app.errors import PermanentError, RetryableError
from app.services import llm as llm_svc
from app.services import parser as parser_svc

log = structlog.get_logger()


def _make_s3_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name="us-east-1",
        config=Config(connect_timeout=10, read_timeout=120),
    )


def _download_file(file_path: str, dest: pathlib.Path) -> None:
    """Download a file from S3/SeaweedFS to dest."""
    s3 = _make_s3_client()
    s3.download_file(settings.s3_bucket, file_path, str(dest))


def _guess_suffix(file_path: str) -> str:
    """Return the file suffix from the S3 key."""
    ext = pathlib.Path(file_path).suffix.lower()
    return ext if ext else ".pdf"


async def _download_and_extract_paragraphs(
    loop: asyncio.AbstractEventLoop,
    file_path: str,
    tmp_path: pathlib.Path,
) -> list[Any]:
    """Download file from S3 and extract raw paragraphs (text, page, anchor)."""
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

    # CPU-bound + sync C library; offload to thread pool.
    try:
        paragraphs = await loop.run_in_executor(
            None, functools.partial(parser_svc.extract_paragraphs_sync, tmp_path)
        )
    except Exception as exc:
        # Parser errors are permanent — the file content won't change on retry.
        raise PermanentError(f"Document parse error: {exc}") from exc

    log.info("paragraph extraction complete", paragraph_count=len(paragraphs))
    return paragraphs


async def _split_clauses_with_llm(
    paragraphs: list[Any],
) -> list[Any]:
    """Use LLM to detect clause boundaries; fall back to regex on failure.

    Fast path: if the regex pass already finds ≥2 clauses (the common case
    after the sub-split fix), the LLM call (3-5 s) is skipped entirely.
    """
    # Fast path: regex first — if sufficient, skip the LLM round-trip.
    regex_clauses = parser_svc._split_into_clauses_regex(paragraphs)
    if len(regex_clauses) >= 2:
        log.info(
            "regex found sufficient clauses — skipping LLM boundary detection",
            clause_count=len(regex_clauses),
        )
        return regex_clauses

    # Regex found 0-1 clause — structure is ambiguous; try LLM for better segmentation.
    log.info(
        "regex found few clauses — attempting LLM boundary detection",
        regex_clause_count=len(regex_clauses),
    )
    paragraph_texts = [text for text, _, _ in paragraphs]

    try:
        boundaries = await llm_svc.extract_clause_boundaries(paragraph_texts)
        if boundaries:
            clauses = parser_svc.clauses_from_boundaries(paragraphs, boundaries)
            log.info(
                "LLM clause extraction complete",
                boundary_count=len(boundaries),
                clause_count=len(clauses),
            )
            return clauses
        log.warning("LLM returned no boundaries — falling back to regex")
    except Exception as exc:
        log.warning(
            "LLM clause boundary detection failed — falling back to regex",
            error=str(exc),
        )

    # Regex fallback (reuse already-computed result)
    log.info("using regex clause split", clause_count=len(regex_clauses))
    return regex_clauses


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
        paragraphs = await _download_and_extract_paragraphs(loop, file_path, tmp_path)

        # LLM 조항 분절 (실패 시 정규식 폴백)
        clauses = await _split_clauses_with_llm(paragraphs)

        # Step 2 → chunking
        await update_ingestion_job(
            pool,
            job_id,
            status="chunking",
            progress=60,
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

        # Step 3 → completed
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


def make_dlq_handler(
    pool: asyncpg.Pool,
) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """Return a DLQ callback that marks the ingestion job as failed in the DB.

    Called by consume_dlq for every message that could not be processed after
    all retries.  Idempotent: a job that is already 'failed' stays 'failed'.
    """

    async def on_dlq_message(msg: dict[str, Any]) -> None:
        job_id = msg.get("jobId")
        contract_id = msg.get("contractId")
        if not job_id:
            log.warning("ingestion DLQ message missing jobId", msg=str(msg)[:200])
            return
        try:
            await update_ingestion_job(
                pool,
                job_id,
                status="failed",
                progress=0,
                error_message="DLQ: 최대 재시도 횟수 초과",
            )
            if contract_id:
                await update_contract_status(pool, contract_id, "failed")
            log.info(
                "ingestion DLQ: job marked failed",
                job_id=job_id,
                contract_id=contract_id,
            )
        except Exception:
            log.exception(
                "ingestion DLQ: failed to update job status",
                job_id=job_id,
            )

    return on_dlq_message
