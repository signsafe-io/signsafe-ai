"""Document parser: extracts clauses from PDF and DOCX files.

조항 분절 전략:
1. 한국어 계약서 조항 헤더 패턴 감지:
   - 제N조, 제N항, 제N목 (한글 숫자 포함)
   - 숫자+점 패턴: 1. / 1) / (1) / ①
   - 영문 Article / Section / Clause + 숫자
2. 헤더가 감지되면 새 조항 시작으로 처리
3. 헤더가 없는 경우 빈 줄 기준 단락 분절 후 의미 단위 병합
4. 최소 길이 30자, 최대 길이 3000자로 조항 크기 제어
"""
from __future__ import annotations

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
# Regex patterns for clause header detection
# ---------------------------------------------------------------------------

# Korean article/section numbers (제1조, 제1항, 제1목, 제1절)
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

_MIN_CLAUSE_LEN = 30
_MAX_CLAUSE_LEN = 3000


def _extract_label(text: str) -> str | None:
    """Extract a short label from the first line of a clause."""
    first_line = text.strip().split("\n")[0].strip()
    if len(first_line) <= 100:
        return first_line or None
    return first_line[:100]


def _split_into_clauses(
    paragraphs: list[tuple[str, int, Anchor | None]],
) -> list[Clause]:
    """Split paragraphs into clauses based on header detection.

    Each paragraph is (text, page_number, anchor).
    """
    clauses: list[Clause] = []
    current_lines: list[str] = []
    current_page_start: int = 1
    current_page_end: int = 1
    current_anchor: Anchor | None = None
    current_label: str | None = None
    char_offset = 0

    def flush() -> None:
        nonlocal current_lines, current_page_start, current_page_end
        nonlocal current_anchor, current_label, char_offset

        merged = "\n".join(current_lines).strip()
        if len(merged) >= _MIN_CLAUSE_LEN:
            # Split oversized clauses at paragraph boundaries.
            if len(merged) > _MAX_CLAUSE_LEN:
                sub_parts = [
                    merged[i : i + _MAX_CLAUSE_LEN]
                    for i in range(0, len(merged), _MAX_CLAUSE_LEN)
                ]
            else:
                sub_parts = [merged]

            for part in sub_parts:
                clauses.append(
                    Clause(
                        text=part,
                        label=current_label,
                        page_start=current_page_start,
                        page_end=current_page_end,
                        anchor=current_anchor,
                        start_offset=char_offset,
                        end_offset=char_offset + len(part),
                    )
                )
                char_offset += len(part) + 1

        current_lines = []
        current_label = None
        current_anchor = None

    for text, page_num, anchor in paragraphs:
        if not text.strip():
            continue

        first_line = text.strip().split("\n")[0]
        is_header = bool(_KO_HEADER.match(first_line.strip()))

        if is_header and current_lines:
            flush()
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
        flush()

    return clauses


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------


def _parse_pdf(data: bytes) -> list[Clause]:
    """Parse PDF bytes into clauses using PyMuPDF."""
    import fitz  # type: ignore[import-untyped]

    doc = fitz.open(stream=data, filetype="pdf")
    paragraphs: list[tuple[str, int, Anchor | None]] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1

        # Extract text blocks with position info.
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()
            if not text:
                continue
            anchor = Anchor(
                x=float(x0),
                y=float(y0),
                width=float(x1 - x0),
                height=float(y1 - y0),
            )
            paragraphs.append((text, page_num, anchor))

    doc.close()
    return _split_into_clauses(paragraphs)


# ---------------------------------------------------------------------------
# DOCX parser
# ---------------------------------------------------------------------------


def _parse_docx(data: bytes) -> list[Clause]:
    """Parse DOCX bytes into clauses using python-docx."""
    from docx import Document  # type: ignore[import-untyped]

    doc = Document(io.BytesIO(data))
    paragraphs: list[tuple[str, int, Anchor | None]] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append((text, 1, None))

    return _split_into_clauses(paragraphs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def parse(file_path: Path) -> list[Clause]:
    """Parse a PDF or DOCX file and return a list of Clause objects."""
    data = file_path.read_bytes()
    suffix = file_path.suffix.lower()

    log.info("parsing document", path=str(file_path), suffix=suffix, size=len(data))

    if suffix == ".pdf":
        clauses = _parse_pdf(data)
    elif suffix in (".docx", ".doc"):
        clauses = _parse_docx(data)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    log.info("parsing complete", clause_count=len(clauses))
    return clauses
