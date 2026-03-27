"""Document parser: extracts plain text from PDF and DOCX files."""
from pathlib import Path


async def parse(file_path: Path) -> str:
    """Return plain text content from a PDF or DOCX file."""
    # TODO: implement parsing using pymupdf / python-docx
    raise NotImplementedError
