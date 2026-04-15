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
            response_format={"type": "json_object"},
        ),
        timeout=_LLM_CALL_TIMEOUT,
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI returned empty content")
    return content


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


@dataclass
class ClauseBoundary:
    """LLM이 감지한 단일 조항의 경계 정보."""

    start: int  # 단락 인덱스 (0-based)
    label: str | None  # 조항명, 없으면 None


_CLAUSE_BOUNDARY_PROMPT_TEMPLATE = """다음은 계약서에서 추출된 단락 목록입니다. 각 단락 앞에 인덱스 번호가 표시됩니다.
계약서의 각 조항이 시작되는 단락의 인덱스와 조항명을 식별해주세요.

단락 목록:
{paragraphs}

다음 JSON 형식으로만 응답하세요:
[
  {{"start": <단락 인덱스>, "label": "<조항 제목 또는 null>"}},
  ...
]

주의사항:
- start 인덱스는 0부터 시작합니다
- 조항 제목(제N조, Article N, Section N 등)이 있으면 label에 포함하세요
- 구분이 불분명한 경우 의미 단위로 묶어 하나의 조항으로 처리하세요
- 짧은 머리말 단독 단락은 다음 본문과 같은 조항으로 처리하세요"""

# 단락당 LLM에 전달할 최대 문자 수 (긴 단락은 잘라서 구조 파악용으로만 사용)
_PARA_PREVIEW_LEN = 150
# 한 번에 LLM에 보낼 최대 단락 수
_CHUNK_SIZE = 80


async def extract_clause_boundaries(
    paragraph_texts: list[str],
) -> list[ClauseBoundary]:
    """LLM을 사용해 단락 목록에서 조항 경계를 감지한다.

    단락이 _CHUNK_SIZE보다 많은 경우 청크로 나눠 처리하고 인덱스를 보정한다.
    """
    if not paragraph_texts:
        return []

    client = _get_client()
    boundaries: list[ClauseBoundary] = []

    # 청크 단위로 처리 (긴 문서 대응)
    for chunk_start in range(0, len(paragraph_texts), _CHUNK_SIZE):
        chunk = paragraph_texts[chunk_start : chunk_start + _CHUNK_SIZE]

        # 단락 목록을 "인덱스: 텍스트 미리보기" 형태로 직렬화
        para_list = "\n".join(
            f"[{chunk_start + i}] {text[:_PARA_PREVIEW_LEN].replace(chr(10), ' ')}"
            + ("..." if len(text) > _PARA_PREVIEW_LEN else "")
            for i, text in enumerate(chunk)
        )

        prompt = _CLAUSE_BOUNDARY_PROMPT_TEMPLATE.format(paragraphs=para_list)

        try:
            raw_text = await _call_llm(client, prompt)
        except (TimeoutError, APIStatusError) as exc:
            log.warning(
                "clause boundary LLM call failed for chunk",
                chunk_start=chunk_start,
                error=str(exc),
            )
            # 청크 실패 시 청크의 첫 단락을 단일 조항으로 처리
            boundaries.append(ClauseBoundary(start=chunk_start, label=None))
            continue

        try:
            data = _extract_json(raw_text)
            if not isinstance(data, list):
                raise ValueError("Expected JSON array")
            for item in data:
                start_idx = int(item.get("start", chunk_start))
                # 인덱스 범위 검증
                if chunk_start <= start_idx < chunk_start + len(chunk):
                    boundaries.append(
                        ClauseBoundary(
                            start=start_idx,
                            label=item.get("label") or None,
                        )
                    )
        except (json.JSONDecodeError, AttributeError, ValueError, TypeError) as exc:
            log.warning(
                "clause boundary JSON parse failed",
                chunk_start=chunk_start,
                error=str(exc),
                raw=raw_text[:200],
            )
            boundaries.append(ClauseBoundary(start=chunk_start, label=None))

    # 중복 제거 및 정렬
    seen: set[int] = set()
    unique: list[ClauseBoundary] = []
    for b in sorted(boundaries, key=lambda x: x.start):
        if b.start not in seen:
            seen.add(b.start)
            unique.append(b)

    return unique


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
