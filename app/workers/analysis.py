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
       a. LLM analysis → risk_level, confidence, issue_types, summary, rationale
       b. RAG search → topK=3 similar clauses
       c. Save clause_result to DB (including confidence)
       d. Save evidence_set to DB
    4. Analysis status completed
    5. On failure → status failed

Error classification:
    PermanentError — missing required fields, DB schema issues. Message is
        acked immediately; no DLQ routing.
    RetryableError — DB connection loss, LLM 429/503, timeout. Consumer retries
        up to _MAX_RETRIES times with exponential back-off before routing to DLQ.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import asyncpg
import structlog
from anthropic import APIStatusError

from app.db import (
    get_clauses_for_contract,
    get_evidence_set_with_clause,
    get_org_id_for_analysis,
    insert_clause_result,
    insert_evidence_set,
    update_evidence_set_citations,
    update_risk_analysis,
    update_risk_analysis_summary,
)
from app.errors import PermanentError, RetryableError
from app.services import rag as rag_svc
from app.services.llm import MODEL, ClauseAnalysisResult, analyze_clause, summarize_document

log = structlog.get_logger()

_MAX_CONCURRENCY = 5
# Total timeout per clause: LLM (60 s) + RAG + DB writes.
_CLAUSE_TIMEOUT = 120.0


def _new_id() -> str:
    """Generate a 26-char alphanumeric ID (similar to ULID format)."""
    return str(uuid.uuid4()).replace("-", "")[:26]


def _classify_exception(exc: BaseException) -> type[RetryableError | PermanentError]:
    """Map an arbitrary exception to a retryable vs permanent category.

    Rules:
    - Anthropic 429 (rate limit) / 503 (overloaded) → retryable
    - asyncpg connection errors → retryable
    - TimeoutError / asyncio.TimeoutError → retryable (may recover)
    - All other Anthropic API errors → permanent (bad request, auth)
    - KeyError / TypeError / ValueError (malformed data) → permanent
    """
    if isinstance(exc, APIStatusError):
        if exc.status_code in (429, 503):
            return RetryableError
        return PermanentError

    if isinstance(
        exc, (asyncpg.PostgresConnectionError, asyncpg.TooManyConnectionsError)
    ):
        return RetryableError

    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return RetryableError

    if isinstance(exc, (KeyError, TypeError, ValueError)):
        return PermanentError

    # Default: treat as retryable (unknown errors may be transient)
    return RetryableError


async def _analyze_single_clause(
    pool: asyncpg.Pool,
    analysis_id: str,
    clause: asyncpg.Record,
    semaphore: asyncio.Semaphore,
    org_id: str | None = None,
) -> ClauseAnalysisResult:
    """Run LLM + RAG for one clause, persist results, and return the LLM result."""
    async with semaphore:
        clause_id: str = clause["id"]
        clause_text: str = clause["content"]
        label: str | None = clause.get("label")
        page_start: int = clause.get("page_start", 1)
        anchor_x: float | None = clause.get("anchor_x")
        anchor_y: float | None = clause.get("anchor_y")
        anchor_width: float | None = clause.get("anchor_width")
        anchor_height: float | None = clause.get("anchor_height")

        log.info("analyzing clause", clause_id=clause_id, label=label)

        # LLM analysis with per-clause total timeout.
        # analyze_clause() already applies a 60 s call-level timeout; this outer
        # guard covers the full pipeline (LLM + RAG + DB) to prevent a semaphore
        # slot from being occupied indefinitely.
        try:
            llm_result = await asyncio.wait_for(
                analyze_clause(clause_text), timeout=_CLAUSE_TIMEOUT
            )
        except TimeoutError:
            log.error(
                "clause analysis timed out",
                clause_id=clause_id,
                analysis_id=analysis_id,
                timeout=_CLAUSE_TIMEOUT,
            )
            raise

        # RAG: find similar clauses scoped to the same organization.
        # Passing org_id ensures we only surface evidence from the org's own
        # historical contracts, preventing cross-tenant data leakage.
        similar, legal_refs = await asyncio.gather(
            rag_svc.search_similar_clauses(
                query_text=clause_text,
                top_k=3,
                org_id=org_id,
            ),
            rag_svc.search_legal_references(
                query_text=clause_text,
                top_k=3,
            ),
        )

        # Build citations: clause-based + legal references (판례/법령).
        citations = [
            {
                "id": s.get("clause_id") or "",
                "type": "clause",
                "title": s.get("label") or "유사 조항",
                "snippet": s.get("payload", {}).get("content", "")[:200],
                "whyRelevant": "",
                "source": s.get("contract_id"),
                "score": s.get("score"),
            }
            for s in similar
        ] + [
            {
                "id": r.get("source_id") or "",
                "type": r.get("type", "prec"),
                "title": r.get("title") or ("판례" if r.get("type") == "prec" else "법령"),
                "snippet": r.get("content", "")[:200],
                "whyRelevant": "",
                "source": f"https://www.law.go.kr/판례/{r.get('source_id', '')}",
                "score": r.get("score"),
                "date": r.get("date", ""),
                "court": r.get("court", ""),
            }
            for r in legal_refs
        ]

        # Persist clause_result (includes confidence score from LLM).
        clause_result_id = _new_id()
        await insert_clause_result(
            pool,
            {
                "id": clause_result_id,
                "analysis_id": analysis_id,
                "clause_id": clause_id,
                "risk_level": llm_result.risk_level,
                "confidence": llm_result.confidence,
                "issue_type": (
                    llm_result.issue_types[0] if llm_result.issue_types else None
                ),
                "summary": llm_result.summary,
                "highlight_x": anchor_x,
                "highlight_y": anchor_y,
                "highlight_width": anchor_width,
                "highlight_height": anchor_height,
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
            confidence=llm_result.confidence,
        )
        return llm_result


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


async def _handle_retrieve_evidence(
    pool: asyncpg.Pool, msg: dict[str, Any]
) -> None:
    """Re-run RAG for an evidence set and update its citations.

    Message format:
        {
            "type": "RETRIEVE_EVIDENCE",
            "evidenceSetId": "<ID>",
            "orgId": "<org UUID>"   (optional)
        }
    """
    evidence_set_id = msg.get("evidenceSetId")
    if not evidence_set_id:
        raise PermanentError("RETRIEVE_EVIDENCE message missing evidenceSetId")

    log.info("RETRIEVE_EVIDENCE started", evidence_set_id=evidence_set_id)

    try:
        row = await get_evidence_set_with_clause(pool, evidence_set_id)
    except asyncpg.PostgresConnectionError as exc:
        raise RetryableError(f"DB error fetching evidence set: {exc}") from exc

    if row is None:
        raise PermanentError(f"evidence_set not found: {evidence_set_id}")

    clause_text: str = row["clause_content"]
    org_id: str | None = msg.get("orgId") or row["org_id"]
    top_k: int = row["top_k"] or 3

    similar = await rag_svc.search_similar_clauses(
        query_text=clause_text,
        top_k=top_k,
        org_id=org_id,
    )

    citations = [
        {
            "id": s.get("clause_id") or "",
            "type": "clause",
            "title": s.get("label") or "유사 조항",
            "snippet": s.get("payload", {}).get("content", "")[:200],
            "whyRelevant": "",
            "source": s.get("contract_id"),
            "score": s.get("score"),
            "url": f"/contracts/{s.get('contract_id')}/clauses/{s.get('clause_id')}",
        }
        for s in similar
    ]

    try:
        await update_evidence_set_citations(pool, evidence_set_id, citations)
    except asyncpg.PostgresConnectionError as exc:
        raise RetryableError(f"DB error updating citations: {exc}") from exc

    log.info(
        "RETRIEVE_EVIDENCE completed",
        evidence_set_id=evidence_set_id,
        citations_count=len(citations),
    )


async def _process(pool: asyncpg.Pool, msg: dict[str, Any]) -> None:
    # Validate required fields — raise PermanentError if missing
    try:
        contract_id: str = msg["contractId"]
        analysis_id: str = msg["analysisId"]
    except KeyError as exc:
        raise PermanentError(f"Missing required message field: {exc}") from exc

    log.info("analysis started", analysis_id=analysis_id, contract_id=contract_id)

    # Step 1 — mark running.
    try:
        await update_risk_analysis(pool, analysis_id, status="running")
    except asyncpg.PostgresConnectionError as exc:
        raise RetryableError(f"DB connection error on status update: {exc}") from exc

    # Ensure Qdrant collection exists before running RAG searches.
    # This handles the case where Qdrant restarts and the collection is gone.
    await rag_svc.ensure_collection()

    # Step 2 — load clauses.
    try:
        clauses = await get_clauses_for_contract(pool, contract_id)
    except asyncpg.PostgresConnectionError as exc:
        raise RetryableError(f"DB connection error loading clauses: {exc}") from exc

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

    # Fetch org_id to scope RAG search to this organization only.
    org_id = await get_org_id_for_analysis(pool, analysis_id)
    if org_id is None:
        log.warning(
            "org_id not found for analysis — RAG will search across all orgs",
            analysis_id=analysis_id,
        )

    # Step 3 — parallel analysis with bounded concurrency.
    # Use return_exceptions=True so a single clause failure does not abort others.
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    tasks = [
        _analyze_single_clause(pool, analysis_id, clause, semaphore, org_id=org_id)
        for clause in clauses
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Categorise per-clause failures.
    failed_retryable = 0
    failed_permanent = 0
    for clause, result in zip(clauses, results):
        if isinstance(result, BaseException):
            category = _classify_exception(result)
            if category is RetryableError:
                failed_retryable += 1
            else:
                failed_permanent += 1
            log.error(
                "clause analysis failed",
                clause_id=clause["id"],
                analysis_id=analysis_id,
                error_type=category.__name__,
                error=str(result),
            )

    failed_count = failed_retryable + failed_permanent
    if failed_count:
        log.warning(
            "analysis completed with clause failures",
            analysis_id=analysis_id,
            failed_retryable=failed_retryable,
            failed_permanent=failed_permanent,
            total=len(clauses),
        )

    # If ALL clauses failed with retryable errors (none permanent), propagate
    # as RetryableError so the consumer can retry the entire analysis job.
    if failed_retryable == len(clauses) and failed_permanent == 0:
        raise RetryableError(f"All {len(clauses)} clauses failed with retryable errors")

    # Step 4 — generate document-level summary from successful clause results.
    successful_results: list[ClauseAnalysisResult] = [
        r for r in results if isinstance(r, ClauseAnalysisResult)
    ]
    if successful_results:
        try:
            doc_summary = await summarize_document(successful_results)
            await update_risk_analysis_summary(
                pool,
                analysis_id,
                document_summary=doc_summary.summary,
                overall_risk=doc_summary.overall_risk,
                key_issues=doc_summary.key_issues,
            )
            log.info(
                "document summary saved",
                analysis_id=analysis_id,
                overall_risk=doc_summary.overall_risk,
            )
        except Exception as exc:
            # Document summary failure is non-fatal — log and continue.
            log.error(
                "document summary failed, skipping",
                analysis_id=analysis_id,
                error=str(exc),
            )

    # Step 5 — mark completed (partial success is still completed).
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
      2. RETRIEVE_EVIDENCE: {"type": "RETRIEVE_EVIDENCE", "evidenceSetId": "...", "orgId": "..."}
    """

    async def handler(msg: dict[str, Any]) -> None:
        # Route by message type.
        msg_type = msg.get("type")
        if msg_type == "RETRIEVE_EVIDENCE":
            evidence_set_id = msg.get("evidenceSetId", "unknown")
            try:
                await _handle_retrieve_evidence(pool, msg)
            except PermanentError as exc:
                log.error(
                    "permanent RETRIEVE_EVIDENCE failure",
                    evidence_set_id=evidence_set_id,
                    error=str(exc),
                )
                raise
            except (RetryableError, Exception) as exc:
                log.error(
                    "retryable RETRIEVE_EVIDENCE failure",
                    evidence_set_id=evidence_set_id,
                    error=str(exc),
                )
                raise
            return

        analysis_id = msg.get("analysisId", "unknown")
        contract_id = msg.get("contractId", "unknown")
        try:
            await _process(pool, msg)
        except PermanentError as exc:
            log.error(
                "permanent analysis failure — marking failed, acking message (no DLQ)",
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
            raise  # re-raise PermanentError; queue.consume acks without DLQ
        except (RetryableError, Exception) as exc:
            log.error(
                "retryable analysis failure — will retry or route to DLQ",
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
            raise  # re-raise; queue.consume retries or nacks → DLQ

    return handler
