from spendwise_rag.core.models import Chunk
from spendwise_rag.processing.analytics import uncategorized_merchants
from spendwise_rag.retrieval import build_index


def test_uncategorized_merchants_summarizes_active_index():
    index = build_index(
        [
            Chunk(
                "2026-01-01 | Mystery | $10.00 | Uncategorized",
                {
                    "chunk_type": "transaction",
                    "date": "2026-01-01",
                    "merchant": "Mystery",
                    "category": "Uncategorized",
                    "amount": 10.0,
                },
            ),
            Chunk(
                "2026-01-02 | Mystery | $5.00 | Uncategorized",
                {
                    "chunk_type": "transaction",
                    "date": "2026-01-02",
                    "merchant": "Mystery",
                    "category": "Uncategorized",
                    "amount": 5.0,
                },
            ),
        ],
        "test",
    )

    result = uncategorized_merchants(index)

    assert result.to_dict(orient="records") == [{"merchant": "Mystery", "amount": 15.0, "count": 2}]
