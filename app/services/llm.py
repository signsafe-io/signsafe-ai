"""LLM service: clause risk analysis using Anthropic Claude API."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import structlog
from anthropic import AsyncAnthropic

from app.config import settings

log = structlog.get_logger()

MODEL = "claude-3-5-sonnet-20241022"

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


_ANALYSIS_PROMPT_TEMPLATE = """계약서 조항을 분석하여 리스크를 평가해주세요.

조항: {clause_text}

다음 JSON 형식으로만 응답하세요:
{{
  "risk_level": "HIGH|MEDIUM|LOW",
  "issue_types": ["LIABILITY_LIMITATION", ...],
  "summary": "한국어 리스크 요약 (2-3문장)",
  "rationale": "판단 근거 상세 설명"
}}

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
    issue_types: list[str]
    summary: str
    rationale: str
    raw: dict[str, Any]


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
        return normalized.lower()
    # Fallback mapping for unexpected values.
    mapping = {"높음": "high", "중간": "medium", "낮음": "low"}
    return mapping.get(value.strip(), "medium")


async def analyze_clause(clause_text: str) -> ClauseAnalysisResult:
    """Run LLM risk analysis on a single clause and return structured result."""
    client = _get_client()
    prompt = _ANALYSIS_PROMPT_TEMPLATE.format(clause_text=clause_text)

    message = await client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = message.content[0].text

    try:
        data = _extract_json(raw_text)
    except (json.JSONDecodeError, AttributeError) as exc:
        log.warning("LLM returned non-JSON response", error=str(exc), raw=raw_text[:200])
        # Return safe defaults on parse failure.
        return ClauseAnalysisResult(
            risk_level="medium",
            issue_types=[],
            summary="분석 결과를 파싱하지 못했습니다.",
            rationale=raw_text,
            raw={"raw_text": raw_text},
        )

    return ClauseAnalysisResult(
        risk_level=_normalize_risk_level(data.get("risk_level", "MEDIUM")),
        issue_types=data.get("issue_types", []),
        summary=data.get("summary", ""),
        rationale=data.get("rationale", ""),
        raw=data,
    )
