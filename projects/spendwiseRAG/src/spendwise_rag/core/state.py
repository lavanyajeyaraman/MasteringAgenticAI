from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document

from .models import BankConfig, Chunk, LocalIndex, ParsedStatement


class IngestionState(TypedDict, total=False):
    pdf_bytes: bytes
    filename: str
    card_type_override: str | None
    card_type: str
    statement_month: str
    statement_year: int
    card_namespace: str
    bank_config: BankConfig
    parsed_statement: ParsedStatement
    raw_tables: list[list[str]]
    raw_text: list[str]
    image_extractions: list[dict[str, Any]]
    excluded_regions: list[tuple[float, float, float, float]]
    chunks: list[Chunk]
    documents: list[Document]
    pinecone_records: list[dict[str, Any]]
    embedding_model: str
    upsert_count: int
    pinecone_stats: dict[str, Any]
    bm25_path: str
    local_index: LocalIndex
    summary: dict[str, Any]
    errors: list[str]
