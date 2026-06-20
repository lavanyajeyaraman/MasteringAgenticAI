import pytest

from semanticats.tools.pdf_parser import parse_pdf


def test_parse_pdf_reads_uploaded_pdf_bytes() -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Alex Rivera\nSkills: Selenium, Python")
    pdf_bytes = doc.tobytes()

    text = parse_pdf(pdf_bytes)

    assert "Alex Rivera" in text
    assert "Selenium" in text


def test_parse_pdf_rejects_pdf_without_selectable_text() -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page()

    with pytest.raises(RuntimeError, match="No selectable text"):
        parse_pdf(doc.tobytes())
