from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from spendwise_rag.core.models import Chunk, ParsedStatement
from spendwise_rag.processing.transaction_parser import (
    ParsedTransaction,
    clean_merchant,
    extract_transactions_from_text_block,
    infer_category,
    is_transaction_dense_text,
    parse_amount,
    parse_date,
)


ROLLUP_RE = re.compile(r"(?P<category>[A-Za-z &]+)\s+Total:?\s+\$?(?P<amount>[\d,]+\.\d{2})", re.I)
def _looks_like_header(row: list[str]) -> bool:
    joined = " ".join(row).lower()
    return any(word in joined for word in ("date description amount", "transaction date", "payments credits"))


def _append_transaction_chunk(
    chunks: list[Chunk],
    seen: set[tuple[str, str, float]],
    statement: ParsedStatement,
    date: str | None,
    merchant: str,
    amount: float | None,
    category: str = "Uncategorized",
) -> None:
    merchant = clean_merchant(merchant)
    if not date or not merchant or amount is None:
        return
    category = category if category and category != "Uncategorized" else infer_category(merchant)
    key = (date, merchant.lower(), round(amount, 2))
    if key in seen:
        return
    seen.add(key)
    text = f"{date} | {merchant} | ${amount:.2f} | {category}"
    chunks.append(
        Chunk(
            text=text,
            metadata={
                "chunk_type": "transaction",
                "date": date,
                "merchant": merchant,
                "amount": amount,
                "category": category,
                "statement_month": statement.statement_month,
                "card_type": statement.card_type,
                "namespace": statement.namespace,
            },
        )
    )


def _table_transaction_chunks(statement: ParsedStatement, seen: set[tuple[str, str, float]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for row in statement.raw_tables:
        if len(row) < 3 or _looks_like_header(row):
            continue
        date = parse_date(row[0], statement.statement_year)
        merchant = row[1].strip() if len(row) > 1 else ""
        amount = parse_amount(row[-1])
        category = row[2].strip() if len(row) > 3 else "Uncategorized"
        _append_transaction_chunk(chunks, seen, statement, date, merchant, amount, category)
    return chunks


def _append_parsed_transaction(
    chunks: list[Chunk],
    seen: set[tuple[str, str, float]],
    statement: ParsedStatement,
    transaction: ParsedTransaction,
) -> None:
    _append_transaction_chunk(
        chunks,
        seen,
        statement,
        transaction.date,
        transaction.merchant,
        transaction.amount,
        transaction.category,
    )


def _text_transaction_chunks(statement: ParsedStatement, seen: set[tuple[str, str, float]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for text_block in statement.raw_text:
        for transaction in extract_transactions_from_text_block(text_block, statement.statement_year):
            _append_parsed_transaction(chunks, seen, statement, transaction)
    return chunks


def transaction_chunks(statement: ParsedStatement) -> list[Chunk]:
    seen: set[tuple[str, str, float]] = set()
    chunks = _table_transaction_chunks(statement, seen)
    if chunks:
        return chunks
    return _text_transaction_chunks(statement, seen)


def _sentence_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return splitter.split_text(re.sub(r"\s+", " ", text).strip())


def summary_chunks(statement: ParsedStatement) -> list[Chunk]:
    chunks: list[Chunk] = []
    for text_block in statement.raw_text:
        if text_block.startswith("[page ") and "discarded" in text_block:
            continue
        if is_transaction_dense_text(text_block):
            continue
        for text in _sentence_chunks(text_block):
            chunks.append(
                Chunk(
                    text=text,
                    metadata={
                        "chunk_type": "summary",
                        "statement_month": statement.statement_month,
                        "card_type": statement.card_type,
                        "namespace": statement.namespace,
                    },
                )
            )
    return chunks


def rollup_chunks(statement: ParsedStatement) -> list[Chunk]:
    chunks: list[Chunk] = []
    for text_block in statement.raw_text:
        for match in ROLLUP_RE.finditer(text_block):
            category = re.sub(r"\s+", " ", match.group("category")).strip().title()
            amount = parse_amount(match.group("amount"))
            if amount is None:
                continue
            text = f"Total {category} - {statement.statement_month} - ${amount:.2f}"
            chunks.append(
                Chunk(
                    text=text,
                    metadata={
                        "chunk_type": "rollup",
                        "category": category,
                        "amount": amount,
                        "statement_month": statement.statement_month,
                        "card_type": statement.card_type,
                        "namespace": statement.namespace,
                    },
                )
            )
    return chunks


def image_chunks(statement: ParsedStatement) -> list[Chunk]:
    chunks = []
    for extraction in statement.image_extractions:
        chunks.append(
            Chunk(
                text=str(extraction),
                metadata={
                    "chunk_type": "image_extract",
                    "statement_month": statement.statement_month,
                    "card_type": statement.card_type,
                    "namespace": statement.namespace,
                },
            )
        )
    return chunks


def chunk_node(statement: ParsedStatement) -> list[Chunk]:
    return transaction_chunks(statement) + rollup_chunks(statement) + summary_chunks(statement) + image_chunks(statement)
