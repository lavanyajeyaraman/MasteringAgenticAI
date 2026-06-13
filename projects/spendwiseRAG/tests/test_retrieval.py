from spendwise_rag.core.models import Chunk
from spendwise_rag.retrieval import build_index, search


def test_search_returns_relevant_financial_chunks():
    chunks = [
        Chunk("2026-05-03 | Spotify | $9.99 | Subscriptions", {"chunk_type": "transaction"}),
        Chunk("2026-05-07 | Trader Joe's | $42.10 | Groceries", {"chunk_type": "transaction"}),
    ]
    index = build_index(chunks, "test_2026")

    matches = search(index, "groceries trader", top_k=1)

    assert len(matches) == 1
    assert "Trader Joe" in matches[0][0].text
