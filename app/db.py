"""Database connection and helper functions using asyncpg."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import asyncpg
import structlog

from app.config import settings

log = structlog.get_logger()


async def connect_db() -> asyncpg.Pool:
    """Create and return an asyncpg connection pool."""
    pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=2,
        max_size=10,
    )
    return pool


# ---------------------------------------------------------------------------
# Ingestion job helpers
# ---------------------------------------------------------------------------


async def update_ingestion_job(
    pool: asyncpg.Pool,
    job_id: str,
    status: str,
    progress: int,
    current_step: str | None = None,
    error_message: str | None = None,
) -> None:
    """Update ingestion_jobs status, progress, and optional step/error."""
    now = datetime.now(timezone.utc)
    # Build query dynamically based on which timestamps to set.
    params: list[Any] = [status, progress, current_step, error_message, now, job_id]

    if status == "parsing":
        query = """
            UPDATE ingestion_jobs
            SET status = $1, progress = $2, current_step = $3,
                error_message = $4, updated_at = $5, started_at = $5
            WHERE id = $6
        """
    elif status in ("completed", "failed"):
        query = """
            UPDATE ingestion_jobs
            SET status = $1, progress = $2, current_step = $3,
                error_message = $4, updated_at = $5, completed_at = $5
            WHERE id = $6
        """
    else:
        query = """
            UPDATE ingestion_jobs
            SET status = $1, progress = $2, current_step = $3,
                error_message = $4, updated_at = $5
            WHERE id = $6
        """

    async with pool.acquire() as conn:
        await conn.execute(query, *params)

    log.info(
        "ingestion_job updated",
        job_id=job_id,
        status=status,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------


async def get_org_id_for_contract(
    pool: asyncpg.Pool,
    contract_id: str,
) -> str | None:
    """Return the organization_id for a contract, or None if not found."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT organization_id FROM contracts WHERE id = $1",
            contract_id,
        )
    return row["organization_id"] if row else None


async def get_org_id_for_analysis(
    pool: asyncpg.Pool,
    analysis_id: str,
) -> str | None:
    """Return the organization_id for a risk analysis via contract join."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT c.organization_id
            FROM risk_analyses ra
            JOIN contracts c ON c.id = ra.contract_id
            WHERE ra.id = $1
            """,
            analysis_id,
        )
    return row["organization_id"] if row else None


async def update_contract_status(
    pool: asyncpg.Pool,
    contract_id: str,
    status: str,
) -> None:
    """Update contracts.status column."""
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE contracts SET status = $1, updated_at = $2 WHERE id = $3",
            status,
            now,
            contract_id,
        )
    log.info("contract status updated", contract_id=contract_id, status=status)


# ---------------------------------------------------------------------------
# Risk analysis helpers
# ---------------------------------------------------------------------------


async def update_risk_analysis(
    pool: asyncpg.Pool,
    analysis_id: str,
    status: str,
    error_message: str | None = None,
    model_version: str | None = None,
) -> None:
    """Update risk_analyses status and optional error/model fields."""
    now = datetime.now(timezone.utc)

    if status == "running":
        query = """
            UPDATE risk_analyses
            SET status = $1, started_at = $2, updated_at = $2
            WHERE id = $3
        """
        params: list[Any] = [status, now, analysis_id]
    elif status == "completed":
        query = """
            UPDATE risk_analyses
            SET status = $1, completed_at = $2, updated_at = $2,
                model_version = $3
            WHERE id = $4
        """
        params = [status, now, model_version, analysis_id]
    elif status == "failed":
        query = """
            UPDATE risk_analyses
            SET status = $1, error_message = $2, completed_at = $3,
                updated_at = $3
            WHERE id = $4
        """
        params = [status, error_message, now, analysis_id]
    else:
        query = """
            UPDATE risk_analyses
            SET status = $1, updated_at = $2
            WHERE id = $3
        """
        params = [status, now, analysis_id]

    async with pool.acquire() as conn:
        await conn.execute(query, *params)

    log.info(
        "risk_analysis updated",
        analysis_id=analysis_id,
        status=status,
    )


# ---------------------------------------------------------------------------
# Clause helpers
# ---------------------------------------------------------------------------


async def insert_clauses_batch(
    pool: asyncpg.Pool,
    contract_id: str,
    clauses: list[dict[str, Any]],
) -> list[str]:
    """Batch-insert clauses and return their IDs.

    Each item in clauses must have:
        id, clause_index, label, content,
        page_start, page_end,
        anchor_x, anchor_y, anchor_width, anchor_height,
        start_offset, end_offset
    """
    now = datetime.now(timezone.utc)
    records = [
        (
            c["id"],
            contract_id,
            c["clause_index"],
            c.get("label"),
            c["content"],
            c.get("page_start", 1),
            c.get("page_end", 1),
            c.get("anchor_x"),
            c.get("anchor_y"),
            c.get("anchor_width"),
            c.get("anchor_height"),
            c.get("start_offset", 0),
            c.get("end_offset", 0),
            now,
            now,
        )
        for c in clauses
    ]

    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO clauses (
                id, contract_id, clause_index, label, content,
                page_start, page_end,
                anchor_x, anchor_y, anchor_width, anchor_height,
                start_offset, end_offset,
                created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7,
                $8, $9, $10, $11,
                $12, $13,
                $14, $15
            )
            ON CONFLICT (id) DO NOTHING
            """,
            records,
        )

    return [c["id"] for c in clauses]


async def get_clauses_for_contract(
    pool: asyncpg.Pool,
    contract_id: str,
) -> list[asyncpg.Record]:
    """Return all clauses for a contract ordered by clause_index."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, contract_id, clause_index, label, content,
                   page_start, page_end,
                   anchor_x, anchor_y, anchor_width, anchor_height,
                   start_offset, end_offset
            FROM clauses
            WHERE contract_id = $1
            ORDER BY clause_index
            """,
            contract_id,
        )
    return rows


# ---------------------------------------------------------------------------
# Clause result helpers
# ---------------------------------------------------------------------------


async def insert_clause_result(
    pool: asyncpg.Pool,
    result: dict,
) -> str:
    """Insert a clause_result row and return its id.

    result dict fields:
        id, analysis_id, clause_id, risk_level, confidence (float, default 0.5),
        issue_type, summary, highlight_x, highlight_y, highlight_width,
        highlight_height, page_number
    """
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO clause_results (
                id, analysis_id, clause_id, risk_level, confidence, issue_type,
                summary, highlight_x, highlight_y,
                highlight_width, highlight_height, page_number,
                created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11, $12,
                $13, $13
            )
            """,
            result["id"],
            result["analysis_id"],
            result["clause_id"],
            result["risk_level"],
            float(result.get("confidence", 0.5)),
            result.get("issue_type"),
            result.get("summary"),
            result.get("highlight_x"),
            result.get("highlight_y"),
            result.get("highlight_width"),
            result.get("highlight_height"),
            result.get("page_number"),
            now,
        )
    return result["id"]


async def update_risk_analysis_summary(
    pool: asyncpg.Pool,
    analysis_id: str,
    document_summary: str,
    overall_risk: str,
    key_issues: list[Any],
) -> None:
    """Update document-level summary fields on risk_analyses.

    Requires migration: ALTER TABLE risk_analyses
        ADD COLUMN document_summary TEXT,
        ADD COLUMN overall_risk VARCHAR(10),
        ADD COLUMN key_issues JSONB;
    """
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE risk_analyses
            SET document_summary = $1, overall_risk = $2,
                key_issues = $3, updated_at = $4
            WHERE id = $5
            """,
            document_summary,
            overall_risk,
            json.dumps(key_issues, ensure_ascii=False),
            now,
            analysis_id,
        )
    log.info(
        "risk_analysis document summary updated",
        analysis_id=analysis_id,
        overall_risk=overall_risk,
    )


async def get_clause_by_id(
    pool: asyncpg.Pool,
    clause_id: str,
) -> asyncpg.Record | None:
    """Return a clause row by id."""
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT id, content, label FROM clauses WHERE id = $1",
            clause_id,
        )


async def get_evidence_set_with_clause(
    pool: asyncpg.Pool,
    evidence_set_id: str,
) -> asyncpg.Record | None:
    """Return evidence_set joined with its clause content and org_id."""
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            """
            SELECT es.id, es.clause_result_id, es.top_k,
                   cr.clause_id, c.content AS clause_content,
                   c.label AS clause_label,
                   co.organization_id AS org_id
            FROM evidence_sets es
            JOIN clause_results cr ON cr.id = es.clause_result_id
            JOIN clauses c ON c.id = cr.clause_id
            JOIN contracts co ON co.id = c.contract_id
            WHERE es.id = $1
            """,
            evidence_set_id,
        )


async def update_evidence_set_citations(
    pool: asyncpg.Pool,
    evidence_set_id: str,
    citations: list[Any],
) -> None:
    """Update citations and retrieved_at for an evidence_set."""
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE evidence_sets
            SET citations = $1, retrieved_at = $2, updated_at = $2
            WHERE id = $3
            """,
            json.dumps(citations, ensure_ascii=False),
            now,
            evidence_set_id,
        )
    log.info("evidence_set citations updated", evidence_set_id=evidence_set_id)


async def insert_evidence_set(
    pool: asyncpg.Pool,
    evidence: dict,
) -> str:
    """Insert an evidence_set row and return its id.

    evidence dict fields:
        id, clause_result_id, rationale, citations (list),
        recommended_actions (list), top_k, filter_params (dict)
    """
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO evidence_sets (
                id, clause_result_id, rationale,
                citations, recommended_actions,
                top_k, filter_params,
                retrieved_at, created_at, updated_at
            ) VALUES (
                $1, $2, $3,
                $4, $5,
                $6, $7,
                $8, $8, $8
            )
            """,
            evidence["id"],
            evidence["clause_result_id"],
            evidence.get("rationale", ""),
            json.dumps(evidence.get("citations", [])),
            json.dumps(evidence.get("recommended_actions", [])),
            evidence.get("top_k", 3),
            json.dumps(evidence.get("filter_params", {})),
            now,
        )
    return evidence["id"]
