"""LLM service: clause risk analysis using OpenAI API."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog
from openai import APIStatusError, AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

log = structlog.get_logger()

MODEL = "gpt-4o"

# Timeout in seconds for a single LLM API call.
_LLM_CALL_TIMEOUT = 60.0

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def _is_retryable(exc: BaseException) -> bool:
    """Return True for OpenAI 429 (rate limit) and 503 (overloaded) errors."""
    if isinstance(exc, APIStatusError):
        return exc.status_code in (429, 503)
    return False


_ANALYSIS_PROMPT_TEMPLATE = """계약서 조항을 분석하여 리스크를 평가해주세요.

조항: {clause_text}

다음 JSON 형식으로만 응답하세요:
{{
  "risk_level": "HIGH|MEDIUM|LOW",
  "confidence": 0.85,
  "issue_types": ["LIABILITY_LIMITATION", ...],
  "summary": "한국어 리스크 요약 (2-3문장)",
  "rationale": "판단 근거 상세 설명"
}}

confidence 필드 설명:
- 0.0~1.0 사이의 숫자로, 위험도 판단에 대한 신뢰도를 나타냅니다.
- 1.0에 가까울수록 명확하고 확실한 판단, 0.5에 가까울수록 해석이 불분명합니다.
- 예시: 명확한 고위험 조항 → 0.95, 해석 여지가 있는 조항 → 0.65

이슈 유형 목록:
- LIABILITY_LIMITATION: 손해배상 제한
- TERMINATION_RIGHT: 일방적 계약 해지권
- IP_OWNERSHIP: 지식재산권 귀속
- PENALTY_CLAUSE: 위약금/페널티
- FORCE_MAJEURE: 불가항력 조항
- GOVERNING_LAW: 준거법/관할
- CONFIDENTIALITY: 기밀유지
- INDEMNITY: 면책 조항
- PAYMENT_TERMS: 지급 조건"""


@dataclass
class ClauseAnalysisResult:
    """Result of LLM clause risk analysis."""

    risk_level: str  # HIGH | MEDIUM | LOW
    confidence: float  # 0.0 ~ 1.0
    issue_types: list[str]
    summary: str
    rationale: str
    raw: dict[str, Any] = field(default_factory=dict)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, tolerating markdown code fences."""
    # Strip markdown code fences if present.
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1)

    text = text.strip()
    return json.loads(text)


def _normalize_risk_level(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in ("HIGH", "MEDIUM", "LOW"):
        return normalized
    # Fallback mapping for unexpected values.
    mapping = {"높음": "HIGH", "중간": "MEDIUM", "낮음": "LOW"}
    return mapping.get(value.strip(), "MEDIUM")


def _normalize_confidence(value: Any) -> float:
    """Clamp confidence to [0.0, 1.0]. Returns 0.5 on invalid input."""
    try:
        f = float(value)
        return max(0.0, min(1.0, f))
    except (TypeError, ValueError):
        return 0.5


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _call_llm(client: AsyncOpenAI, prompt: str) -> str:
    """Call OpenAI API with timeout and retry on 429/503."""
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ),
        timeout=_LLM_CALL_TIMEOUT,
    )
    return response.choices[0].message.content


_DOCUMENT_SUMMARY_PROMPT_TEMPLATE = """다음은 계약서의 조항별 리스크 분석 결과입니다.
전체 계약서를 종합하여 문서 수준의 리스크 요약을 작성해주세요.

조항별 분석 결과:
{clause_summaries}

통계: 전체 {total}개 조항 / 고위험 {high_count}개 / 중위험 {medium_count}개 / 저위험 {low_count}개

다음 JSON 형식으로만 응답하세요:
{{
  "overall_risk": "HIGH|MEDIUM|LOW",
  "summary": "계약서 전체 리스크에 대한 2-4문장 한국어 요약",
  "key_issues": ["주요 이슈 1", "주요 이슈 2", ...]
}}"""


@dataclass
class DocumentSummaryResult:
    """Document-level risk summary aggregated from clause results."""

    overall_risk: str  # HIGH | MEDIUM | LOW
    summary: str
    key_issues: list[str]


async def summarize_document(
    clause_results: list[ClauseAnalysisResult],
) -> DocumentSummaryResult:
    """Produce a document-level risk summary from clause analysis results."""
    if not clause_results:
        return DocumentSummaryResult(
            overall_risk="LOW", summary="분석된 조항이 없습니다.", key_issues=[]
        )

    client = _get_client()

    high = sum(1 for r in clause_results if r.risk_level == "HIGH")
    medium = sum(1 for r in clause_results if r.risk_level == "MEDIUM")
    low = sum(1 for r in clause_results if r.risk_level == "LOW")

    clause_summaries = "\n".join(
        f"[{r.risk_level}] {r.summary}" for r in clause_results if r.summary
    )

    prompt = _DOCUMENT_SUMMARY_PROMPT_TEMPLATE.format(
        clause_summaries=clause_summaries or "(요약 없음)",
        total=len(clause_results),
        high_count=high,
        medium_count=medium,
        low_count=low,
    )

    try:
        raw_text = await _call_llm(client, prompt)
    except TimeoutError:
        log.error("document summary LLM call timed out")
        raise
    except APIStatusError as exc:
        log.error("document summary LLM API error", status_code=exc.status_code)
        raise

    try:
        data = _extract_json(raw_text)
    except (json.JSONDecodeError, AttributeError) as exc:
        log.warning(
            "document summary non-JSON response", error=str(exc), raw=raw_text[:200]
        )
        fallback_risk = "HIGH" if high > 0 else ("MEDIUM" if medium > 0 else "LOW")
        return DocumentSummaryResult(
            overall_risk=fallback_risk,
            summary="문서 전체 요약을 생성하지 못했습니다.",
            key_issues=[],
        )

    return DocumentSummaryResult(
        overall_risk=_normalize_risk_level(data.get("overall_risk", "MEDIUM")),
        summary=data.get("summary", ""),
        key_issues=data.get("key_issues", []),
    )


async def analyze_clause(clause_text: str) -> ClauseAnalysisResult:
    """Run LLM risk analysis on a single clause and return structured result."""
    client = _get_client()
    prompt = _ANALYSIS_PROMPT_TEMPLATE.format(clause_text=clause_text)

    try:
        raw_text = await _call_llm(client, prompt)
    except TimeoutError:
        log.error(
            "LLM call timed out",
            timeout=_LLM_CALL_TIMEOUT,
            clause_preview=clause_text[:100],
        )
        raise
    except APIStatusError as exc:
        log.error(
            "LLM API error after retries",
            status_code=exc.status_code,
            clause_preview=clause_text[:100],
        )
        raise

    try:
        data = _extract_json(raw_text)
    except (json.JSONDecodeError, AttributeError) as exc:
        log.warning(
            "LLM returned non-JSON response", error=str(exc), raw=raw_text[:200]
        )
        # Return safe defaults on parse failure.
        return ClauseAnalysisResult(
            risk_level="MEDIUM",
            confidence=0.5,
            issue_types=[],
            summary="분석 결과를 파싱하지 못했습니다.",
            rationale=raw_text,
            raw={"raw_text": raw_text},
        )

    return ClauseAnalysisResult(
        risk_level=_normalize_risk_level(data.get("risk_level", "MEDIUM")),
        confidence=_normalize_confidence(data.get("confidence", 0.5)),
        issue_types=data.get("issue_types", []),
        summary=data.get("summary", ""),
        rationale=data.get("rationale", ""),
        raw=data,
    )
