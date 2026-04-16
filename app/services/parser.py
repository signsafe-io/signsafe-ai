"""Document parser: extracts clauses from PDF and DOCX files.

조항 분절 전략:
1. PDF/DOCX에서 원시 단락(text, page, anchor)을 추출한다.
2. LLM이 단락 목록을 분석해 조항 경계(start 인덱스, label)를 반환한다.
3. 경계 정보를 바탕으로 단락들을 Clause 객체로 조립한다.
4. LLM 호출 실패 시 정규식 기반 폴백(_KO_HEADER)을 사용한다.
5. 최소 길이 30자, 최대 길이 3000자로 조항 크기 제어.
"""

from __future__ import annotations

import asyncio
import functools
import io
import re
from dataclasses import dataclass
from pathlib import Path

import structlog

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Anchor:
    """Bounding box coordinates on a page (PDF only)."""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0


@dataclass
class Clause:
    """A single extracted clause from a contract."""

    text: str
    label: str | None
    page_start: int
    page_end: int
    anchor: Anchor | None
    start_offset: int = 0
    end_offset: int = 0


# ---------------------------------------------------------------------------
# Regex fallback patterns for clause header detection
# ---------------------------------------------------------------------------

_KO_HEADER = re.compile(
    r"^(제\s*\d+\s*[조항목절]"
    r"|제\s*[일이삼사오육칠팔구십백]+\s*[조항목절]"
    r"|\d+\.\s+[가-힣A-Za-z]"
    r"|\d+\)\s+[가-힣A-Za-z]"
    r"|\(\s*\d+\s*\)\s+"
    r"|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]"
    r"|Article\s+\d+"
    r"|Section\s+\d+"
    r"|Clause\s+\d+)",
    re.IGNORECASE,
)

# 최소 길이는 페이지 번호·공백 같은 노이즈만 걸러낼 수 있을 만큼 낮게 유지한다.
# 30자로 설정하면 짧은 정상 조항(예: "제1조\n계약 당사자는 갑·을이다.")도 드롭됨.
_MIN_CLAUSE_LEN = 5
_MAX_CLAUSE_LEN = 3000

# ---------------------------------------------------------------------------
# Raw paragraph types
# ---------------------------------------------------------------------------

# (text, page_number, anchor_or_none)
RawParagraph = tuple[str, int, Anchor | None]


def _sub_split_block(
    text: str, page_num: int, anchor: Anchor | None
) -> list[RawParagraph]:
    """PyMuPDF 블록 내에 여러 조항 헤더가 포함된 경우 서브스플릿.

    get_text("blocks") 는 레이아웃 블록 단위라 "제1조\n...\n제2조\n..." 같이
    여러 조항이 하나의 블록으로 묶이는 경우가 있다. 이 경우 LLM/정규식 모두
    첫 번째 헤더만 인식하므로, 헤더 패턴이 등장하는 줄마다 새 단락을 시작한다.
    """
    lines = text.split("\n")
    result: list[RawParagraph] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if _KO_HEADER.match(stripped) and current:
            # 헤더 등장 → 직전 단락 flush 후 새 단락 시작
            combined = "\n".join(current).strip()
            if combined:
                result.append((combined, page_num, anchor))
            current = [line]
        else:
            current.append(line)

    if current:
        combined = "\n".join(current).strip()
        if combined:
            result.append((combined, page_num, anchor))

    return result if result else [(text, page_num, anchor)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_label(text: str) -> str | None:
    first_line = text.strip().split("\n")[0].strip()
    if len(first_line) <= 100:
        return first_line or None
    return first_line[:100]


def _flush_lines(
    lines: list[str],
    page_start: int,
    page_end: int,
    anchor: Anchor | None,
    label: str | None,
    char_offset: int,
    out: list[Clause],
) -> int:
    """Merge lines into Clause(s), appending to out. Returns updated char_offset."""
    merged = "\n".join(lines).strip()
    # 헤더 패턴으로 시작하는 조항은 길이 무관하게 보존한다.
    # 짧아도 정상 조항(예: "제4조\n기간: 미정")이 누락되는 것을 방지.
    has_header = (
        bool(_KO_HEADER.match(merged.split("\n")[0].strip())) if merged else False
    )
    if len(merged) < _MIN_CLAUSE_LEN and not has_header:
        return char_offset

    if len(merged) > _MAX_CLAUSE_LEN:
        sub_parts = [
            merged[i : i + _MAX_CLAUSE_LEN]
            for i in range(0, len(merged), _MAX_CLAUSE_LEN)
        ]
    else:
        sub_parts = [merged]

    for part in sub_parts:
        out.append(
            Clause(
                text=part,
                label=label,
                page_start=page_start,
                page_end=page_end,
                anchor=anchor,
                start_offset=char_offset,
                end_offset=char_offset + len(part),
            )
        )
        char_offset += len(part) + 1

    return char_offset


# ---------------------------------------------------------------------------
# Regex-based clause splitter (fallback)
# ---------------------------------------------------------------------------


def _split_into_clauses_regex(
    paragraphs: list[RawParagraph],
) -> list[Clause]:
    """Split paragraphs into clauses using regex header detection (fallback)."""
    clauses: list[Clause] = []
    current_lines: list[str] = []
    current_page_start = 1
    current_page_end = 1
    current_anchor: Anchor | None = None
    current_label: str | None = None
    char_offset = 0

    for text, page_num, anchor in paragraphs:
        if not text.strip():
            continue

        first_line = text.strip().split("\n")[0]
        is_header = bool(_KO_HEADER.match(first_line.strip()))

        if is_header and current_lines:
            char_offset = _flush_lines(
                current_lines,
                current_page_start,
                current_page_end,
                current_anchor,
                current_label,
                char_offset,
                clauses,
            )
            current_lines = []
            current_page_start = page_num
            current_page_end = page_num
            current_anchor = anchor
            current_label = _extract_label(text)
            current_lines = [text]
        else:
            if not current_lines:
                current_page_start = page_num
                current_anchor = anchor
                if is_header:
                    current_label = _extract_label(text)
            current_page_end = page_num
            current_lines.append(text)

    if current_lines:
        _flush_lines(
            current_lines,
            current_page_start,
            current_page_end,
            current_anchor,
            current_label,
            char_offset,
            clauses,
        )

    return clauses


# ---------------------------------------------------------------------------
# LLM-boundary-based clause assembler
# ---------------------------------------------------------------------------


def clauses_from_boundaries(
    paragraphs: list[RawParagraph],
    boundaries: list[Any],  # list[ClauseBoundary] — avoid circular import
) -> list[Clause]:
    """Assemble Clause objects using LLM-detected boundaries.

    boundaries must be sorted by .start (ascending) and contain no duplicates.
    Each boundary marks the first paragraph of a new clause.

    Short chunks (e.g. header-only lines) are carried forward and merged into
    the next chunk instead of being dropped, so that a clause header and its
    body always end up in the same Clause object even if the LLM accidentally
    placed an extra boundary between them.
    """
    if not boundaries or not paragraphs:
        return []

    clauses: list[Clause] = []
    char_offset = 0

    # Convert boundaries to (start_idx, label) pairs and add a sentinel.
    breakpoints = [(b.start, b.label) for b in boundaries]
    breakpoints.append((len(paragraphs), None))  # sentinel

    # Carry-over state: lines from a too-short previous chunk that must be
    # prepended to the next chunk rather than discarded.
    carry_lines: list[str] = []
    carry_label: str | None = None
    carry_page_start: int = 1
    carry_page_end: int = 1
    carry_anchor: Anchor | None = None

    for i, (start_idx, label) in enumerate(breakpoints[:-1]):
        end_idx = breakpoints[i + 1][0]

        chunk = paragraphs[start_idx:end_idx]
        if not chunk:
            continue

        lines = [t for t, _, _ in chunk if t.strip()]
        pages = [p for _, p, _ in chunk]
        first_anchor = next((a for _, _, a in chunk if a is not None), None)
        page_start = pages[0] if pages else 1
        page_end = pages[-1] if pages else 1

        # Merge carry-over from previous too-short chunk.
        if carry_lines:
            lines = carry_lines + lines
            page_start = carry_page_start
            page_end = max(carry_page_end, page_end)
            first_anchor = carry_anchor or first_anchor
            # Carry-over label takes priority (it was the actual clause header).
            label = carry_label or label
            carry_lines = []
            carry_label = None
            carry_anchor = None

        effective_label = label or (_extract_label("\n".join(lines)) if lines else None)
        merged = "\n".join(lines).strip()

        if not merged:
            continue

        if len(merged) < _MIN_CLAUSE_LEN:
            # Too short on its own — likely a bare header line that the LLM
            # split from its body. Carry forward to the next iteration.
            carry_lines = lines
            carry_label = effective_label
            carry_page_start = page_start
            carry_page_end = page_end
            carry_anchor = first_anchor
            continue

        # Normal path: write out (splitting oversized clauses if needed).
        char_offset = _flush_lines(
            lines,
            page_start,
            page_end,
            first_anchor,
            effective_label,
            char_offset,
            clauses,
        )

    # Flush any remaining carry-over (last segment was a bare header with no body).
    if carry_lines:
        merged = "\n".join(carry_lines).strip()
        if merged:
            clauses.append(
                Clause(
                    text=merged,
                    label=carry_label,
                    page_start=carry_page_start,
                    page_end=carry_page_end,
                    anchor=carry_anchor,
                    start_offset=char_offset,
                    end_offset=char_offset + len(merged),
                )
            )

    return clauses


# ---------------------------------------------------------------------------
# Raw paragraph extractors (sync, CPU-bound)
# ---------------------------------------------------------------------------


def _extract_paragraphs_pdf(data: bytes) -> list[RawParagraph]:
    """Extract raw (text, page, anchor) tuples from PDF bytes using PyMuPDF."""
    import fitz  # type: ignore[import-untyped]

    doc = fitz.open(stream=data, filetype="pdf")
    paragraphs: list[RawParagraph] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_width = page.rect.width or 1.0
        page_height = page.rect.height or 1.0

        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()
            if not text:
                continue
            anchor = Anchor(
                x=float(x0) / page_width,
                y=float(y0) / page_height,
                width=float(x1 - x0) / page_width,
                height=float(y1 - y0) / page_height,
            )
            # 하나의 블록에 여러 조항 헤더가 포함될 수 있으므로 서브스플릿.
            paragraphs.extend(_sub_split_block(text, page_num, anchor))

    doc.close()
    return paragraphs


def _extract_paragraphs_docx(data: bytes) -> list[RawParagraph]:
    """Extract raw (text, page, anchor) tuples from DOCX bytes using python-docx."""
    from docx import Document  # type: ignore[import-untyped]

    doc = Document(io.BytesIO(data))
    paragraphs: list[RawParagraph] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append((text, 1, None))
    return paragraphs


def extract_paragraphs_sync(file_path: Path) -> list[RawParagraph]:
    """Extract raw paragraphs from a PDF or DOCX file (sync, run in thread pool).

    Returns a list of (text, page_number, anchor) tuples. Does NOT perform
    clause splitting — use LLM boundaries + clauses_from_boundaries(), or
    fall back to _split_into_clauses_regex() if LLM is unavailable.
    """
    data = file_path.read_bytes()
    suffix = file_path.suffix.lower()

    log.info(
        "extracting paragraphs", path=str(file_path), suffix=suffix, size=len(data)
    )

    if suffix == ".pdf":
        paragraphs = _extract_paragraphs_pdf(data)
    elif suffix in (".docx", ".doc"):
        paragraphs = _extract_paragraphs_docx(data)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    log.info("paragraph extraction complete", count=len(paragraphs))
    return paragraphs


# ---------------------------------------------------------------------------
# Public API (regex-only, kept for backward compatibility / testing)
# ---------------------------------------------------------------------------

# Import Any here to satisfy clauses_from_boundaries signature without
# a circular import from llm.py.
from typing import Any  # noqa: E402


def parse_sync(file_path: Path) -> list[Clause]:
    """Parse a PDF or DOCX file synchronously using regex-based clause splitting.

    This is the regex-only fallback path. The primary ingestion pipeline calls
    extract_paragraphs_sync() then uses LLM-based boundary detection instead.
    """
    paragraphs = extract_paragraphs_sync(file_path)
    clauses = _split_into_clauses_regex(paragraphs)
    log.info("regex parse complete", clause_count=len(clauses))
    return clauses


async def parse(file_path: Path) -> list[Clause]:
    """Async shim — kept for backward compatibility.

    Prefer calling parse_sync via loop.run_in_executor so that the
    synchronous C library (PyMuPDF) does not block the event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(parse_sync, file_path))
