from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import pdfplumber

from spendwise_rag.core.config import get_bank_config
from spendwise_rag.core.models import BankConfig, ParsedStatement


CARD_PATTERNS = {
    "chase_sapphire": ["chase sapphire", "sapphire preferred", "sapphire reserve"],
    "chase": ["chase"],
    "amex": ["american express", "amex"],
    "bofa": ["bank of america", "bofa"],
    "capital_one": ["capital one"],
    "discover": ["discover"],
    "citi": ["citibank", "citi"],
}


def read_upload_bytes(uploaded_file: BinaryIO | bytes) -> bytes:
    if isinstance(uploaded_file, bytes):
        return uploaded_file
    position = uploaded_file.tell() if hasattr(uploaded_file, "tell") else None
    data = uploaded_file.read()
    if position is not None and hasattr(uploaded_file, "seek"):
        uploaded_file.seek(position)
    return data


def detect_card_type(filename: str, first_page_text: str) -> str:
    haystack = f"{filename} {first_page_text}".lower().replace("_", " ")
    for card_type, needles in CARD_PATTERNS.items():
        if any(needle in haystack for needle in needles):
            return card_type
    return "generic_card"


def infer_statement_year(text: str) -> int:
    current = datetime.now().year
    years = [int(year) for year in re.findall(r"\b(20\d{2})\b", text)]
    reasonable = [year for year in years if 2000 <= year <= current + 1]
    return max(set(reasonable or [current]), key=(reasonable or [current]).count)


def infer_statement_month(text: str, year: int) -> str:
    month_names = (
        "January February March April May June July August September October November December"
    ).split()
    for month in month_names:
        if re.search(rf"\b{month}\b", text, flags=re.I):
            return f"{month} {year}"
    numeric = re.search(r"\b(?P<month>\d{1,2})[/-]\d{1,2}[/-](?:\d{2}|\d{4})\b", text)
    if numeric:
        month_idx = max(1, min(12, int(numeric.group("month"))))
        return f"{month_names[month_idx - 1]} {year}"
    return f"Unknown {year}"


def build_namespace(card_type: str, year: int) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", card_type.lower()).strip("_")
    return f"{clean}_{year}"


def upload_node(pdf_bytes: bytes, filename: str, card_type_override: str | None = None) -> dict[str, object]:
    first_page_text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        if pdf.pages:
            first_page_text = pdf.pages[0].extract_text() or ""
    card_type = card_type_override or detect_card_type(filename, first_page_text)
    year = infer_statement_year(first_page_text)
    month = infer_statement_month(first_page_text, year)
    return {
        "card_type": card_type,
        "statement_month": month,
        "statement_year": year,
        "card_namespace": build_namespace(card_type, year),
        "bank_config": get_bank_config(card_type),
    }


def _table_settings(config: BankConfig) -> dict[str, object]:
    settings: dict[str, object] = {}
    if config.explicit_vertical_lines:
        settings["vertical_strategy"] = "explicit"
        settings["explicit_vertical_lines"] = config.explicit_vertical_lines
    return settings


def parse_node(pdf_bytes: bytes, filename: str, card_type_override: str | None = None) -> ParsedStatement:
    upload = upload_node(pdf_bytes, filename, card_type_override=card_type_override)
    config = upload["bank_config"]
    raw_tables: list[list[str]] = []
    raw_text: list[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            table_source = page.crop(config.table_region) if config.table_region else page
            for table in table_source.extract_tables(_table_settings(config)) or []:
                for row in table:
                    cleaned = [str(cell or "").strip() for cell in row]
                    if any(cleaned):
                        raw_tables.append(cleaned)

            rotated_chars = [char for char in page.chars if int(char.get("upright", 1)) == 0]
            if rotated_chars:
                raw_text.append(f"[page {page_number}] discarded {len(rotated_chars)} rotated character(s)")

            page_text = page.extract_text() or ""
            if page_text.strip():
                raw_text.append(page_text)

    return ParsedStatement(
        card_type=str(upload["card_type"]),
        statement_month=str(upload["statement_month"]),
        statement_year=int(upload["statement_year"]),
        namespace=str(upload["card_namespace"]),
        raw_tables=raw_tables,
        raw_text=raw_text,
    )


def parse_pdf_path(path: str | Path, card_type_override: str | None = None) -> ParsedStatement:
    file_path = Path(path)
    return parse_node(file_path.read_bytes(), file_path.name, card_type_override=card_type_override)
