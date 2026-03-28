"""Contract analysis worker: runs LLM + RAG pipeline and writes results to DB.

Message format (from signsafe-api/internal/queue/rabbitmq.go):
    {
        "contractId":  "<ULID>",
        "analysisId":  "<ULID>"
    }

Processing pipeline:
    1. Receive message → Analysis status running
    2. Load clauses from DB
    3. Parallel analysis (max 5 concurrent):
       a. LLM analysis → risk_level, issue_types, summary, rationale
       b. RAG search → topK=3 similar clauses
       c. Save clause_result to DB
       d. Save evidence_set to DB
    4. Analysis status completed
    5. On failure → status failed
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import asyncpg
import structlog

from app.db import (
    get_clauses_for_contract,
    insert_clause_result,
    insert_evidence_set,
    update_risk_analysis,
)
from app.services import rag as rag_svc
from app.services.llm import MODEL, analyze_clause

log = structlog.get_logger()

_MAX_CONCURRENCY = 5


def _new_id() -> str:
    """Generate a 26-char alphanumeric ID (similar to ULID format)."""
    return str(uuid.uuid4()).replace("-", "")[:26]


async def _analyze_single_clause(
    pool: asyncpg.Pool,
    analysis_id: str,
    clause: asyncpg.Record,
    semaphore: asyncio.Semaphore,
) -> None:
    """Run LLM + RAG for one clause and persist results."""
    async with semaphore:
        clause_id = clause["id"]
        clause_text = clause["content"]
        label = clause.get("label")
        page_start = clause.get("page_start", 1)

        log.info("analyzing clause", clause_id=clause_id, label=label)

        # LLM analysis.
        llm_result = await analyze_clause(clause_text)

        # RAG: find similar clauses.
        similar = await rag_svc.search_similar_clauses(
            query_text=clause_text,
            top_k=3,
        )

        # Build citations from similar clauses.
        # Field names must match the frontend Citation TypeScript interface:
        # id, type, title, snippet, whyRelevant, source?, score?
        citations = [
            {
                "id": s.get("clause_id") or "",
                "type": "clause",
                "title": s.get("label") or "Similar clause",
                "snippet": s.get("payload", {}).get("content", "")[:200],
                "whyRelevant": "",
                "source": s.get("contract_id"),
                "score": s.get("score"),
            }
            for s in similar
        ]

        # Persist clause_result.
        clause_result_id = _new_id()
        await insert_clause_result(
            pool,
            {
                "id": clause_result_id,
                "analysis_id": analysis_id,
                "clause_id": clause_id,
                "risk_level": llm_result.risk_level,
                "issue_type": (
                    llm_result.issue_types[0] if llm_result.issue_types else None
                ),
                "summary": llm_result.summary,
                "highlight_x": clause.get("anchor_x"),
                "highlight_y": clause.get("anchor_y"),
                "highlight_width": clause.get("anchor_width"),
                "highlight_height": clause.get("anchor_height"),
                "page_number": page_start,
            },
        )

        # Persist evidence_set.
        await insert_evidence_set(
            pool,
            {
                "id": _new_id(),
                "clause_result_id": clause_result_id,
                "rationale": llm_result.rationale,
                "citations": citations,
                "recommended_actions": _build_recommended_actions(
                    llm_result.issue_types
                ),
                "top_k": 3,
                "filter_params": {},
            },
        )

        log.info(
            "clause analysis saved",
            clause_id=clause_id,
            risk_level=llm_result.risk_level,
        )


def _build_recommended_actions(issue_types: list[str]) -> list[str]:
    """Map issue types to recommended actions (Korean)."""
    actions_map: dict[str, str] = {
        "LIABILITY_LIMITATION": "손해배상 한도 조항을 검토하고 상한액의 적절성을 확인하세요.",
        "TERMINATION_RIGHT": "일방적 해지권 조항의 요건과 사전 통지 기간을 협상하세요.",
        "IP_OWNERSHIP": "지식재산권 귀속 범위를 명확히 정의하고 공동 소유 조항을 검토하세요.",
        "PENALTY_CLAUSE": "위약금 금액과 적용 조건이 상호적으로 공평한지 확인하세요.",
        "FORCE_MAJEURE": "불가항력 사유의 범위와 통지 의무를 구체적으로 명시하도록 요청하세요.",
        "GOVERNING_LAW": "준거법 및 관할 법원이 자사에 불리하지 않은지 검토하세요.",
        "CONFIDENTIALITY": "기밀유지 의무의 기간과 예외 사항을 명확히 합의하세요.",
        "INDEMNITY": "면책 범위가 과도하게 광범위하지 않은지 법무팀에 검토 의뢰하세요.",
        "PAYMENT_TERMS": "지급 기한, 연체 이자, 이의제기 절차를 명시적으로 규정하세요.",
    }
    return [actions_map[it] for it in issue_types if it in actions_map]


async def _process(pool: asyncpg.Pool, msg: dict[str, Any]) -> None:
    contract_id: str = msg["contractId"]
    analysis_id: str = msg["analysisId"]

    log.info("analysis started", analysis_id=analysis_id, contract_id=contract_id)

    # Step 1 — mark running.
    await update_risk_analysis(pool, analysis_id, status="running")

    # Ensure Qdrant collection exists before running RAG searches.
    # This handles the case where Qdrant restarts and the collection is gone.
    await rag_svc.ensure_collection()

    # Step 2 — load clauses.
    clauses = await get_clauses_for_contract(pool, contract_id)
    if not clauses:
        log.warning("no clauses found for contract", contract_id=contract_id)
        await update_risk_analysis(
            pool,
            analysis_id,
            status="completed",
            model_version=MODEL,
        )
        return

    log.info("loaded clauses", count=len(clauses))

    # Step 3 — parallel analysis with bounded concurrency.
    # Use return_exceptions=True so a single clause failure does not abort others.
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    tasks = [
        _analyze_single_clause(pool, analysis_id, clause, semaphore)
        for clause in clauses
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Log per-clause failures but do not propagate; partial results are acceptable.
    failed_count = 0
    for clause, result in zip(clauses, results):
        if isinstance(result, BaseException):
            failed_count += 1
            log.error(
                "clause analysis failed",
                clause_id=clause["id"],
                analysis_id=analysis_id,
                error=str(result),
            )

    if failed_count:
        log.warning(
            "analysis completed with clause failures",
            analysis_id=analysis_id,
            failed=failed_count,
            total=len(clauses),
        )

    # Step 4 — mark completed.
    await update_risk_analysis(
        pool,
        analysis_id,
        status="completed",
        model_version=MODEL,
    )
    log.info("analysis completed", analysis_id=analysis_id)


def make_handler(
    pool: asyncpg.Pool,
) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """Return an async message handler bound to the given DB pool.

    Handles two message types on analysis.jobs:
      1. Analysis jobs: {"contractId": "...", "analysisId": "..."}
      2. RETRIEVE_EVIDENCE: {"type": "RETRIEVE_EVIDENCE", "evidenceSetId": "...", ...}

    RETRIEVE_EVIDENCE messages are acknowledged and ignored here because the
    actual retrieval logic (re-running RAG) is not yet implemented as a background
    step — evidence is already populated during the initial analysis pass. Silently
    dropping the message prevents the queue from being poisoned with unhandled types.
    """

    async def handler(msg: dict[str, Any]) -> None:
        # Route by message type. RETRIEVE_EVIDENCE messages are sent to this
        # queue by evidenceSvc.RetrieveEvidence and must not be processed as
        # analysis jobs (they lack contractId/analysisId fields).
        msg_type = msg.get("type")
        if msg_type == "RETRIEVE_EVIDENCE":
            log.info(
                "RETRIEVE_EVIDENCE message received and acknowledged (no-op)",
                evidence_set_id=msg.get("evidenceSetId"),
            )
            return

        analysis_id = msg.get("analysisId", "unknown")
        contract_id = msg.get("contractId", "unknown")
        try:
            await _process(pool, msg)
        except Exception as exc:
            log.exception(
                "analysis failed",
                analysis_id=analysis_id,
                contract_id=contract_id,
                error=str(exc),
            )
            try:
                await update_risk_analysis(
                    pool,
                    analysis_id,
                    status="failed",
                    error_message=str(exc),
                )
            except Exception:
                log.exception(
                    "failed to update analysis status to failed",
                    analysis_id=analysis_id,
                )
            raise  # re-raise so aio-pika nacks → DLQ

    return handler
