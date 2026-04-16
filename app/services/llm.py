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
{legal_context_section}
다음 JSON 형식으로만 응답하세요:
{{
  "risk_level": "HIGH|MEDIUM|LOW",
  "confidence": 0.85,
  "issue_types": ["LIABILITY_LIMITATION", ...],
  "summary": "한국어 리스크 요약 (2-3문장)",
  "rationale": "판단 근거 상세 설명 — 관련 판례/법령이 제공된 경우 이를 구체적으로 인용할 것"
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


def _build_legal_context_section(legal_refs: list[dict]) -> str:
    """Format retrieved 판례/법령 as a prompt section for the LLM."""
    if not legal_refs:
        return ""
    lines = ["\n관련 판례/법령 (분석 시 참고하여 rationale에 인용):"]
    for i, ref in enumerate(legal_refs[:3], 1):  # 최대 3개만 주입
        ref_type = "판례" if ref.get("type") == "prec" else "법령"
        title = ref.get("title", "")
        content = ref.get("content", "")[:300].replace("\n", " ")
        date = ref.get("date", "")
        court = ref.get("court", "")
        meta = f"{court} {date}".strip() if (court or date) else ""
        lines.append(
            f"[{i}] [{ref_type}] {title}"
            + (f" ({meta})" if meta else "")
            + f"\n    {content}"
        )
    return "\n".join(lines) + "\n"


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
각 조항이 **시작되는** 단락의 인덱스만 골라주세요.

단락 목록:
{paragraphs}

다음 JSON 형식으로만 응답하세요 (boundaries 키 아래에 배열로):
{{
  "boundaries": [
    {{"start": <단락 인덱스>, "label": "<조항 제목 또는 null>"}},
    ...
  ]
}}

[경계 선택 원칙]
1. 조항 헤더(제N조, 제N항, Article N, Section N, 1. 제목, (1) 등)가 시작되는 단락만 경계로 표시합니다.
2. 조항 본문(을은 갑에게..., 다음 각 호..., 단, ... 등 내용 텍스트)은 절대 경계로 표시하지 않습니다.
   본문은 앞선 조항 헤더와 동일한 조항에 속합니다.
3. 헤더 단락과 바로 뒤따르는 본문 단락은 같은 조항으로 묶어야 합니다.
   예) [0] "제 1조 계약 목적"  [1] "을은 갑에게 서비스를 제공한다." → [0]만 경계, [1]은 본문
4. 줄바꿈으로 인해 헤더가 짧게 나뉘더라도 조항 번호·제목이 포함된 단락만 경계입니다.
5. 문서 형식 다양성 대응:
   - 한국어: 제N조, 제N항, 제N목, 제N절, N. 제목, (N) 등
   - 영문: Article N, Section N, Clause N, N. Title
   - 번호 없는 의미 단위: 문맥상 명확히 새 주제가 시작될 때만 경계
6. 구분이 불분명하면 이전 조항에 포함시키세요 (경계를 적게 잡는 것이 많이 잡는 것보다 안전합니다).
- start 인덱스는 0부터 시작합니다."""

# 단락당 LLM에 전달할 최대 문자 수 (긴 단락은 잘라서 구조 파악용으로만 사용)
_PARA_PREVIEW_LEN = 150
# 한 번에 LLM에 보낼 최대 단락 수.
# GPT-4o는 128k 컨텍스트. 150자 × 500단락 ≈ 2만 토큰 → 여유롭게 전체 전송 가능.
# 대부분의 계약서는 500단락 미만이므로 실질적으로 전체 문서를 한 번에 봄.
_CHUNK_SIZE = 500
# 청크 간 overlap: 이전 청크 마지막 N개 단락을 다음 청크 앞에 포함해 경계 맥락 유지.
_OVERLAP = 10


def _serialize_paragraphs(texts: list[str], offset: int) -> str:
    """단락 목록을 LLM 프롬프트용 '[인덱스] 미리보기' 형태로 직렬화."""
    return "\n".join(
        f"[{offset + i}] {text[:_PARA_PREVIEW_LEN].replace(chr(10), ' ')}"
        + ("..." if len(text) > _PARA_PREVIEW_LEN else "")
        for i, text in enumerate(texts)
    )


def _parse_boundaries_from_response(
    raw_text: str,
    valid_start: int,
    valid_end: int,
) -> list[ClauseBoundary]:
    """LLM 응답에서 boundaries 파싱. valid_start..valid_end 범위 외 인덱스 무시."""
    data = _extract_json(raw_text)
    if isinstance(data, dict):
        data = data.get("boundaries", [])
    if not isinstance(data, list):
        raise ValueError("Expected list under 'boundaries' key")
    result = []
    for item in data:
        start_idx = int(item.get("start", valid_start))
        if valid_start <= start_idx < valid_end:
            result.append(
                ClauseBoundary(start=start_idx, label=item.get("label") or None)
            )
    return result


async def extract_clause_boundaries(
    paragraph_texts: list[str],
) -> list[ClauseBoundary]:
    """LLM을 사용해 단락 목록에서 조항 경계를 감지한다.

    500단락 이하(대부분의 계약서)는 전체를 한 번에 전송해 LLM이 문서 전체 맥락을
    보고 경계를 잡도록 한다. 500단락 초과 시에는 _OVERLAP만큼 겹치는 슬라이딩
    윈도우로 청크를 나눠 처리하여 청크 경계에서의 누락을 방지한다.
    """
    if not paragraph_texts:
        return []

    client = _get_client()
    all_boundaries: list[ClauseBoundary] = []

    # 청크 시작 인덱스 목록 생성 (overlap 적용)
    chunk_starts = list(range(0, len(paragraph_texts), _CHUNK_SIZE - _OVERLAP))
    # 마지막 청크가 문서 끝까지 포함되도록 보정
    if not chunk_starts or chunk_starts[-1] + _CHUNK_SIZE < len(paragraph_texts):
        if len(paragraph_texts) > _CHUNK_SIZE:
            pass  # range()가 이미 처리
    # 단일 청크인 경우
    if len(paragraph_texts) <= _CHUNK_SIZE:
        chunk_starts = [0]

    for chunk_idx, chunk_start in enumerate(chunk_starts):
        chunk = paragraph_texts[chunk_start : chunk_start + _CHUNK_SIZE]
        if not chunk:
            continue

        # overlap 구간(이전 청크 마지막 _OVERLAP개)은 경계 감지 대상에서 제외.
        # 이미 이전 청크에서 처리된 인덱스이므로 중복 방지.
        new_start = chunk_start if chunk_idx == 0 else chunk_start + _OVERLAP
        new_end = chunk_start + len(chunk)

        para_list = _serialize_paragraphs(chunk, chunk_start)
        prompt = _CLAUSE_BOUNDARY_PROMPT_TEMPLATE.format(paragraphs=para_list)

        log.info(
            "clause boundary LLM call",
            chunk_idx=chunk_idx,
            para_range=f"{chunk_start}-{chunk_start + len(chunk) - 1}",
            new_range=f"{new_start}-{new_end - 1}",
            total=len(paragraph_texts),
        )

        try:
            raw_text = await _call_llm(client, prompt)
        except (TimeoutError, APIStatusError) as exc:
            log.warning(
                "clause boundary LLM call failed for chunk",
                chunk_start=chunk_start,
                error=str(exc),
            )
            # 실패 시 새 구간 첫 단락을 단일 조항 시작으로 처리
            all_boundaries.append(ClauseBoundary(start=new_start, label=None))
            continue

        try:
            chunk_boundaries = _parse_boundaries_from_response(
                raw_text,
                valid_start=new_start,
                valid_end=new_end,
            )
            all_boundaries.extend(chunk_boundaries)
            log.info(
                "clause boundary chunk done",
                chunk_start=chunk_start,
                found=len(chunk_boundaries),
            )
        except (json.JSONDecodeError, AttributeError, ValueError, TypeError) as exc:
            log.warning(
                "clause boundary JSON parse failed",
                chunk_start=chunk_start,
                error=str(exc),
                raw=raw_text[:200],
            )
            all_boundaries.append(ClauseBoundary(start=new_start, label=None))

    # 중복 제거 및 정렬
    seen: set[int] = set()
    unique: list[ClauseBoundary] = []
    for b in sorted(all_boundaries, key=lambda x: x.start):
        if b.start not in seen:
            seen.add(b.start)
            unique.append(b)

    log.info(
        "clause boundaries total",
        count=len(unique),
        total_paragraphs=len(paragraph_texts),
    )
    return unique


async def analyze_clause(
    clause_text: str,
    legal_refs: list[dict] | None = None,
) -> ClauseAnalysisResult:
    """Run LLM risk analysis on a single clause and return structured result.

    Args:
        clause_text: the raw clause to analyze.
        legal_refs: optional list of retrieved 판례/법령 from RAG search.
            When provided, injected into the prompt so the LLM can cite them
            in the rationale instead of reasoning from scratch.
    """
    client = _get_client()
    legal_context_section = _build_legal_context_section(legal_refs or [])
    prompt = _ANALYSIS_PROMPT_TEMPLATE.format(
        clause_text=clause_text,
        legal_context_section=legal_context_section,
    )

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
