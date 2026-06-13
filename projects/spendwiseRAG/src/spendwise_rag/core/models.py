from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class BankConfig:
    name: str
    date_format: str = "%m/%d/%Y"
    table_region: BBox | None = None
    explicit_vertical_lines: list[float] = field(default_factory=list)
    amount_column_index: int = 3
    merchant_column_index: int = 1
    category_column_index: int | None = None


@dataclass
class ParsedStatement:
    card_type: str
    statement_month: str
    statement_year: int
    namespace: str
    raw_tables: list[list[str]]
    raw_text: list[str]
    image_extractions: list[dict[str, Any]] = field(default_factory=list)
    excluded_regions: list[BBox] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict[str, Any]


@dataclass
class LocalIndex:
    namespace: str
    chunks: list[Chunk]
    tokenized: list[list[str]]

