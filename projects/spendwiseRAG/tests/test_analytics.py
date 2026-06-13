from spendwise_rag.core.models import Chunk
from spendwise_rag.processing.analytics import monthly_spend_trend, spending_by_category, top_merchants
from spendwise_rag.retrieval import build_index


def test_analytics_are_derived_from_transaction_chunks():
    chunks = [
        Chunk(
            "2026-05-03 | Spotify | $9.99 | Subscriptions",
            {
                "chunk_type": "transaction",
                "date": "2026-05-03",
                "merchant": "Spotify",
                "category": "Subscriptions",
                "amount": 9.99,
                "statement_month": "May 2026",
                "namespace": "test_2026",
            },
        ),
        Chunk(
            "2026-05-07 | Trader Joe's | $42.10 | Groceries",
            {
                "chunk_type": "transaction",
                "date": "2026-05-07",
                "merchant": "Trader Joe's",
                "category": "Groceries",
                "amount": 42.10,
                "statement_month": "May 2026",
                "namespace": "test_2026",
            },
        ),
    ]
    index = build_index(chunks, "test_2026")

    assert spending_by_category(index).iloc[0]["category"] == "Groceries"
    assert monthly_spend_trend(index).iloc[0]["amount"] == 52.09
    assert top_merchants(index).iloc[0]["merchant"] == "Trader Joe's"
