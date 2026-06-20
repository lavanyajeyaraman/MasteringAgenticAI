from __future__ import annotations

from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def report_to_pdf(report: dict) -> bytes:
    buffer = BytesIO()
    page = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 48
    page.setFont("Helvetica-Bold", 14)
    page.drawString(48, y, f"semanticATSRecruit Report: {report.get('candidate_name', '')}")
    y -= 28
    page.setFont("Helvetica", 10)
    lines = [
        f"Recommendation: {report.get('hiring_recommendation')}",
        f"Transferability score: {report.get('transferability_score')}",
        f"Ramp-up: {report.get('ramp_up_estimate')}",
        f"Summary: {report.get('recruiter_summary')}",
        f"Gaps: {', '.join(report.get('gaps', [])) or 'None'}",
    ]
    for line in lines:
        for wrapped in _wrap(line, 95):
            page.drawString(48, y, wrapped)
            y -= 14
            if y < 48:
                page.showPage()
                page.setFont("Helvetica", 10)
                y = height - 48
    page.save()
    return buffer.getvalue()


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
