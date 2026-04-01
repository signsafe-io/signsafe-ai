"""LLM service: clause risk analysis using Anthropic Claude API."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog
from anthropic import APIStatusError, AsyncAnthropic
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

log = structlog.get_logger()

MODEL = "claude-3-5-sonnet-20241022"

# Timeout in seconds for a single LLM API call.
_LLM_CALL_TIMEOUT = 60.0

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


def _is_retryable(exc: BaseException) -> bool:
    """Return True for Anthropic 429 (rate limit) and 503 (overloaded) errors."""
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
async def _call_llm(client: AsyncAnthropic, prompt: str) -> str:
    """Call Claude API with timeout and retry on 429/503."""
    message = await asyncio.wait_for(
        client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ),
        timeout=_LLM_CALL_TIMEOUT,
    )
    return message.content[0].text


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
