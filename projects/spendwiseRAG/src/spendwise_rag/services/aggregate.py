from __future__ import annotations

import re
from calendar import month_name
from typing import Any

import pandas as pd

from spendwise_rag.core.models import LocalIndex
from spendwise_rag.processing.analytics import transaction_frame


MONTHS = {name.lower(): idx for idx, name in enumerate(month_name) if name}
CATEGORY_ALIASES = {
    "grocery": "Groceries",
    "groceries": "Groceries",
    "dining": "Dining",
    "dining out": "Dining",
    "restaurant": "Dining",
    "restaurants": "Dining",
    "travel": "Travel",
    "subscription": "Subscriptions",
    "subscriptions": "Subscriptions",
    "shopping": "Shopping",
    "entertainment": "Entertainment",
    "utilities": "Utilities",
    "donation": "Donations",
    "donations": "Donations",
    "auto": "Auto",
}


def answer_aggregate_question(
    index: LocalIndex,
    question: str,
    query_plan: dict[str, object] | None = None,
) -> dict[str, Any] | None:
    lowered = question.lower()
    df = transaction_frame(index)
    if df.empty:
        return None

    if _is_top_merchants_query(lowered):
        return _answer_top_merchants(df, _requested_limit(lowered, default=5))

    planned_comparison = _planned_comparison_categories(query_plan)
    if planned_comparison:
        return _answer_category_comparison(df, planned_comparison, planner_used=True)

    comparison = _comparison_categories(df, lowered)
    if comparison:
        return _answer_category_comparison(df, comparison)

    if not _is_total_spending_query(lowered):
        return None

    filtered = df.copy()
    filters = []

    month_filter = _month_filter(filtered, lowered)
    if month_filter:
        month_period, label = month_filter
        filtered = filtered[filtered["month"].eq(month_period)]
        filters.append(label)

    category = _category_filter(filtered, lowered)
    if category:
        filtered = filtered[filtered["category"].str.lower().eq(category.lower())]
        filters.append(category)

    merchant = _merchant_filter(filtered, lowered)
    if merchant:
        filtered = filtered[filtered["merchant"].str.lower().eq(merchant.lower())]
        filters.append(merchant)

    if not filters and _has_unresolved_semantic_noun(lowered):
        return None

    if filtered.empty:
        return {
            "answer": f"I found no matching transactions for {' / '.join(filters) if filters else 'that spending query'}.",
            "model_provider": "deterministic_analytics",
            "confidence": 0.99,
            "faithfulness": 1.0,
            "rerank_used": False,
            "matches": [],
            "retrieval_diagnostics": {
                "pinecone_used": False,
                "pinecone_namespaces": [],
                "vector_error": "",
                "vector_warning": "Deterministic aggregate answered from local transaction chunks; Pinecone vector search was not needed.",
            },
        }

    total = float(filtered["amount"].sum())
    count = int(len(filtered))
    scope = " for " + " / ".join(filters) if filters else ""
    answer = f"You spent ${total:,.2f}{scope} across {count} transaction(s)."
    if count > 10:
        answer += " The sources show the first 10 matching transactions; the total uses all matching transaction chunks."

    return {
        "answer": answer,
        "model_provider": "deterministic_analytics",
        "confidence": 0.99,
        "faithfulness": 1.0,
        "rerank_used": False,
        "matches": _source_matches(filtered),
        "retrieval_diagnostics": {
            "pinecone_used": False,
            "pinecone_namespaces": sorted(filtered["namespace"].dropna().astype(str).unique().tolist()),
            "vector_error": "",
            "vector_warning": "Deterministic aggregate answered from local transaction chunks; Pinecone vector search was not needed.",
        },
    }


def _is_total_spending_query(lowered: str) -> bool:
    total_signal = any(term in lowered for term in ("total", "how much", "sum"))
    spend_signal = any(term in lowered for term in ("spend", "spent", "spending", "amount"))
    return total_signal and spend_signal


def _has_unresolved_semantic_noun(lowered: str) -> bool:
    semantic_terms = re.sub(
        r"\b(show|me|all|what|is|are|my|the|total|amount|how|much|did|i|we|spend|spent|spending|charges|charge|transactions|transaction|for|on|in|month)\b",
        " ",
        lowered,
    )
    semantic_terms = re.sub(r"[^a-z0-9 ]", " ", semantic_terms)
    return bool(re.sub(r"\s+", "", semantic_terms))


def _is_top_merchants_query(lowered: str) -> bool:
    return "merchant" in lowered and any(term in lowered for term in ("top", "highest", "biggest", "most"))


def _requested_limit(lowered: str, default: int) -> int:
    match = re.search(r"\btop\s+(\d{1,2})\b", lowered)
    if not match:
        return default
    return max(1, min(20, int(match.group(1))))


def _answer_top_merchants(df: pd.DataFrame, limit: int) -> dict[str, Any]:
    grouped = (
        df.groupby("merchant", as_index=False)
        .agg(amount=("amount", "sum"), count=("merchant", "size"))
        .sort_values(["amount", "count", "merchant"], ascending=[False, False, True])
        .head(limit)
    )
    lines = [
        f"{idx}. {row['merchant']}: ${float(row['amount']):,.2f} across {int(row['count'])} transaction(s)"
        for idx, (_, row) in enumerate(grouped.iterrows(), start=1)
    ]
    return {
        "answer": "Top merchants by total spend:\n" + "\n".join(lines),
        "model_provider": "deterministic_analytics",
        "confidence": 0.99,
        "faithfulness": 1.0,
        "rerank_used": False,
        "matches": _top_merchant_sources(df, grouped["merchant"].tolist()),
        "retrieval_diagnostics": {
            "pinecone_used": False,
            "pinecone_namespaces": sorted(df["namespace"].dropna().astype(str).unique().tolist()),
            "vector_error": "",
            "vector_warning": "Deterministic merchant ranking answered from local transaction chunks; Pinecone vector search was not needed.",
        },
    }


def _comparison_categories(df: pd.DataFrame, lowered: str) -> list[str]:
    if not any(term in lowered for term in ("compare", " vs ", " versus ", "difference between")):
        return []

    categories = []
    for alias, category in CATEGORY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered) and category not in categories:
            categories.append(category)
    for category in sorted(df["category"].dropna().astype(str).unique(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(category.lower())}\b", lowered) and category not in categories:
            categories.append(category)
    return categories[:2] if len(categories) >= 2 else []


def _planned_comparison_categories(query_plan: dict[str, object] | None) -> list[str]:
    if not query_plan:
        return []
    intent = str(query_plan.get("intent", ""))
    operation = str(query_plan.get("operation", ""))
    if intent != "compare_categories" and operation != "highest_spending":
        return []
    categories = [str(category) for category in query_plan.get("categories", [])]
    return categories[:2] if len(categories) >= 2 else []


def _answer_category_comparison(
    df: pd.DataFrame,
    categories: list[str],
    planner_used: bool = False,
) -> dict[str, Any]:
    left, right = categories
    left_rows = df[df["category"].str.lower().eq(left.lower())]
    right_rows = df[df["category"].str.lower().eq(right.lower())]
    left_total = float(left_rows["amount"].sum())
    right_total = float(right_rows["amount"].sum())
    difference = abs(left_total - right_total)
    higher = left if left_total >= right_total else right
    answer = (
        f"{left}: ${left_total:,.2f} across {len(left_rows)} transaction(s). "
        f"{right}: ${right_total:,.2f} across {len(right_rows)} transaction(s). "
        f"{higher} is higher by ${difference:,.2f}."
    )
    sources = _source_matches(pd.concat([left_rows, right_rows], ignore_index=True))
    return {
        "answer": answer,
        "model_provider": "llm_planned_deterministic_analytics" if planner_used else "deterministic_analytics",
        "confidence": 0.99,
        "faithfulness": 1.0,
        "rerank_used": False,
        "matches": sources,
        "retrieval_diagnostics": {
            "pinecone_used": False,
            "pinecone_namespaces": sorted(df["namespace"].dropna().astype(str).unique().tolist()),
            "vector_error": "",
            "vector_warning": (
                "LLM planner mapped the question to categories; deterministic comparison answered from local transaction chunks."
                if planner_used
                else "Deterministic comparison answered from local transaction chunks; Pinecone vector search was not needed."
            ),
        },
    }


def _month_filter(df: pd.DataFrame, lowered: str) -> tuple[str, str] | None:
    for name, month_number in MONTHS.items():
        if re.search(rf"\b{name}\b", lowered):
            year_match = re.search(r"\b(20\d{2})\b", lowered)
            year = int(year_match.group(1)) if year_match else _latest_year_for_month(df, month_number)
            if not year:
                return None
            return f"{year}-{month_number:02d}", f"{month_name[month_number]} {year}"
    return None


def _latest_year_for_month(df: pd.DataFrame, month_number: int) -> int | None:
    matches = df[df["date"].dt.month.eq(month_number)]
    if matches.empty:
        return None
    return int(matches["date"].dt.year.max())


def _category_filter(df: pd.DataFrame, lowered: str) -> str | None:
    for alias, category in CATEGORY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return category
    for category in sorted(df["category"].dropna().astype(str).unique(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(category.lower())}\b", lowered):
            return category
    return None


def _merchant_filter(df: pd.DataFrame, lowered: str) -> str | None:
    for merchant in sorted(df["merchant"].dropna().astype(str).unique(), key=len, reverse=True):
        if len(merchant) < 4:
            continue
        if re.search(rf"\b{re.escape(merchant.lower())}\b", lowered):
            return merchant
    return None


def _source_matches(df: pd.DataFrame) -> list[dict[str, Any]]:
    sources = []
    ordered = df.sort_values(["date", "merchant", "amount"]).head(10)
    for _, row in ordered.iterrows():
        text = f"{row['date'].date().isoformat()} | {row['merchant']} | ${float(row['amount']):.2f} | {row['category']}"
        sources.append(
            {
                "text": text,
                "score": 1.0,
                "metadata": {
                    "chunk_type": "transaction",
                    "date": row["date"].date().isoformat(),
                    "merchant": row["merchant"],
                    "amount": float(row["amount"]),
                    "category": row["category"],
                    "statement_month": row.get("statement_month", ""),
                    "namespace": row.get("namespace", ""),
                },
            }
        )
    return sources


def _top_merchant_sources(df: pd.DataFrame, merchants: list[str]) -> list[dict[str, Any]]:
    rows = df[df["merchant"].isin(merchants)].copy()
    return _source_matches(rows.sort_values(["merchant", "amount"], ascending=[True, False]))
