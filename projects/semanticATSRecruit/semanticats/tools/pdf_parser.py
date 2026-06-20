from __future__ import annotations

from pathlib import Path


def parse_pdf(source: str | Path | bytes) -> str:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required to parse PDF resumes.") from exc

    if isinstance(source, bytes):
        doc = fitz.open(stream=source, filetype="pdf")
    else:
        doc = fitz.open(source)
    text = "\n".join(page.get_text("text") for page in doc).strip()
    if not text:
        raise RuntimeError(
            "No selectable text was found in this PDF. Upload a text-based resume PDF or a TXT file."
        )
    return text
