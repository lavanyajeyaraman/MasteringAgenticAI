from __future__ import annotations

import pandas as pd

from spendwise_rag.core.models import LocalIndex


def transaction_frame(index: LocalIndex | None) -> pd.DataFrame:
    if index is None:
        return pd.DataFrame(columns=["date", "merchant", "category", "amount", "statement_month", "namespace"])

    rows = []
    for chunk in index.chunks:
        metadata = chunk.metadata
        if metadata.get("chunk_type") != "transaction":
            continue
        rows.append(
            {
                "date": metadata.get("date"),
                "merchant": metadata.get("merchant", "Unknown"),
                "category": metadata.get("category", "Uncategorized"),
                "amount": float(metadata.get("amount", 0.0) or 0.0),
                "statement_month": metadata.get("statement_month", "Unknown"),
                "namespace": metadata.get("namespace", index.namespace),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "merchant", "category", "amount", "statement_month", "namespace"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


def spending_by_category(index: LocalIndex | None) -> pd.DataFrame:
    df = transaction_frame(index)
    if df.empty:
        return pd.DataFrame(columns=["category", "amount"])
    return df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)


def monthly_spend_trend(index: LocalIndex | None) -> pd.DataFrame:
    df = transaction_frame(index)
    if df.empty:
        return pd.DataFrame(columns=["month", "amount"])
    return df.groupby("month", as_index=False)["amount"].sum().sort_values("month")


def top_merchants(index: LocalIndex | None, limit: int = 10) -> pd.DataFrame:
    df = transaction_frame(index)
    if df.empty:
        return pd.DataFrame(columns=["merchant", "amount"])
    return df.groupby("merchant", as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(limit)


def uncategorized_merchants(index: LocalIndex | None) -> pd.DataFrame:
    df = transaction_frame(index)
    if df.empty:
        return pd.DataFrame(columns=["merchant", "amount", "count"])
    uncategorized = df[df["category"].eq("Uncategorized")]
    if uncategorized.empty:
        return pd.DataFrame(columns=["merchant", "amount", "count"])
    return (
        uncategorized.groupby("merchant", as_index=False)
        .agg(amount=("amount", "sum"), count=("merchant", "size"))
        .sort_values(["amount", "count"], ascending=False)
    )
