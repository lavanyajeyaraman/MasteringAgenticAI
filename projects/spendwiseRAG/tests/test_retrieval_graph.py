from spendwise_rag.core.models import Chunk
from spendwise_rag.graphs.retrieval_graph import (
    confidence_check_node,
    context_validation_node,
    expand_financial_query,
    hybrid_search_node,
    query_node,
)
from spendwise_rag.retrieval import build_index
from spendwise_rag.services.vector_store import PineconeQueryResult


def test_query_node_expands_financial_aliases_and_detects_intent():
    state = query_node({"query": "How much did I spend on food delivery last month?"})

    assert state["query_intent"] == "category_total"
    assert "Uber Eats" in state["expanded_query"]
    assert "DoorDash" in state["expanded_query"]
    assert "statement_month" in state["metadata_filters"]


def test_expand_financial_query_adds_category_synonyms():
    expanded, intent, filters = expand_financial_query("entertainment total")

    assert intent == "category_total"
    assert "Netflix" in expanded
    assert "streaming" in expanded
    assert filters["category"] == "Entertainment"


def test_expand_financial_query_adds_airfare_semantic_hints_and_transaction_filter():
    expanded, intent, filters = expand_financial_query("Show me all airfare charges and the total amount.")

    assert intent == "category_total"
    assert "Frontier Airlines" in expanded
    assert "flight" in expanded
    assert filters["chunk_type"] == "transaction"
    assert filters["merchant"] == {
        "$in": ["Frontier Airlines", "Delta Air Lines", "American Airlines", "Spirit Airlines"]
    }


def test_hybrid_search_filters_airfare_to_airline_merchants(monkeypatch):
    chunks = [
        Chunk(
            "2026-05-08 | Frontier Airlines | $11.20 | Travel",
            {"chunk_type": "transaction", "merchant": "Frontier Airlines", "category": "Travel", "namespace": "amex_2026"},
        ),
        Chunk(
            "2026-04-24 | The Human Bean | $7.82 | Dining",
            {"chunk_type": "transaction", "merchant": "The Human Bean", "category": "Dining", "namespace": "amex_2026"},
        ),
    ]
    index = build_index(chunks, "active_session")
    state = query_node(
        {
            "query": "Show me all airfare charges and the total amount.",
            "local_index": index,
            "namespace": "active_session",
        }
    )

    def fake_pinecone_search(query, namespace, top_k=20, metadata_filter=None):
        assert metadata_filter["merchant"] == {
            "$in": ["Frontier Airlines", "Delta Air Lines", "American Airlines", "Spirit Airlines"]
        }
        return PineconeQueryResult(
            [
                {
                    "id": "amex_2026_1",
                    "text": "2026-05-08 | Frontier Airlines | $11.20 | Travel",
                    "metadata": {
                        "chunk_type": "transaction",
                        "merchant": "Frontier Airlines",
                        "amount": 11.2,
                        "category": "Travel",
                        "namespace": namespace,
                    },
                    "score": 0.9,
                    "rank": 1,
                    "source": "vector",
                }
            ],
            used=True,
        )

    monkeypatch.setattr("spendwise_rag.graphs.retrieval_graph.query_pinecone_records_detailed", fake_pinecone_search)

    result = hybrid_search_node(state)

    assert result["merged_results"][0]["text"] == "2026-05-08 | Frontier Airlines | $11.20 | Travel"
    assert all(item["metadata"]["merchant"] == "Frontier Airlines" for item in result["merged_results"])


def test_hybrid_search_does_not_retry_unfiltered_for_airfare_when_pinecone_has_no_matches(monkeypatch):
    chunks = [
        Chunk(
            "2026-04-24 | The Human Bean | $7.82 | Dining",
            {"chunk_type": "transaction", "merchant": "The Human Bean", "category": "Dining", "namespace": "amex_2026"},
        )
    ]
    index = build_index(chunks, "active_session")
    state = query_node(
        {
            "query": "Show me all airfare charges and the total amount.",
            "local_index": index,
            "namespace": "active_session",
        }
    )
    calls = []

    def fake_pinecone_search(query, namespace, top_k=20, metadata_filter=None):
        calls.append(metadata_filter)
        return PineconeQueryResult([], warning=f"Pinecone returned no vector matches for namespace `{namespace}`.", used=True)

    monkeypatch.setattr("spendwise_rag.graphs.retrieval_graph.query_pinecone_records_detailed", fake_pinecone_search)

    result = hybrid_search_node(state)

    assert len(calls) == 1
    assert calls[0]["merchant"] == {
        "$in": ["Frontier Airlines", "Delta Air Lines", "American Airlines", "Spirit Airlines"]
    }
    assert result["merged_results"] == []
    assert "no vector matches" in result["vector_warning"]


def test_context_validation_rejects_airfare_query_with_unrelated_the_human_bean_source():
    state = query_node(
        {
            "query": "Show me all airfare charges and the total amount.",
            "merged_results": [
                {
                    "text": "2026-04-24 | The Human Bean | $7.82 | Dining",
                    "metadata": {
                        "chunk_type": "transaction",
                        "merchant": "The Human Bean",
                        "amount": 7.82,
                        "category": "Dining",
                    },
                    "score": 1.0,
                }
            ],
            "rerank_used": False,
        }
    )

    result = context_validation_node(state)

    assert result["final_context"] == []
    assert result["response_message"] == "No relevant data found for that question in the indexed statements."


def test_hybrid_search_merges_bm25_and_pinecone_vector_results_with_confidence(monkeypatch):
    chunks = [
        Chunk(
            "2026-05-03 | Spotify | $9.99 | Subscriptions",
            {"chunk_type": "transaction", "merchant": "Spotify", "category": "Subscriptions", "namespace": "test"},
        ),
        Chunk(
            "2026-05-07 | Trader Joe's | $42.10 | Groceries",
            {"chunk_type": "transaction", "merchant": "Trader Joe's", "category": "Groceries", "namespace": "test"},
        ),
    ]
    index = build_index(chunks, "test")
    state = query_node({"query": "Trader Joe $42.10 groceries", "local_index": index, "namespace": "test"})

    def fake_pinecone_search(query, namespace, top_k=20, metadata_filter=None):
        return PineconeQueryResult(
            [
                {
                    "id": "pinecone_1",
                    "text": "2026-05-07 | Trader Joe's | $42.10 | Groceries",
                    "metadata": {"chunk_type": "transaction", "merchant": "Trader Joe's", "category": "Groceries"},
                    "score": 0.91,
                    "rank": 1,
                    "source": "vector",
                }
            ],
            used=True,
        )

    monkeypatch.setattr("spendwise_rag.graphs.retrieval_graph.query_pinecone_records_detailed", fake_pinecone_search)

    result = hybrid_search_node(state)

    assert result["merged_results"][0]["text"] == "2026-05-07 | Trader Joe's | $42.10 | Groceries"
    assert result["merged_results"][0]["sources"] == ["bm25", "vector"]
    assert result["pinecone_used"] is True
    assert result["vector_error"] == ""
    assert result["vector_warning"] == ""
    assert result["confidence_score"] >= 0.70
    assert confidence_check_node(result) == "generate_node"


def test_hybrid_search_uses_chunk_namespaces_instead_of_active_session(monkeypatch):
    chunks = [
        Chunk(
            "2026-01-22 | Dunkin | $13.54 | Dining",
            {"chunk_type": "transaction", "merchant": "Dunkin", "category": "Dining", "namespace": "amex_2026"},
        )
    ]
    index = build_index(chunks, "active_session")
    state = query_node({"query": "dining total", "local_index": index, "namespace": "active_session"})
    queried_namespaces = []

    def fake_pinecone_search(query, namespace, top_k=20, metadata_filter=None):
        queried_namespaces.append(namespace)
        return PineconeQueryResult(
            [
                {
                    "id": "amex_2026_1",
                    "text": "2026-01-22 | Dunkin | $13.54 | Dining",
                    "metadata": {"chunk_type": "transaction", "merchant": "Dunkin", "category": "Dining", "namespace": namespace},
                    "score": 0.92,
                    "rank": 1,
                    "source": "vector",
                }
            ],
            used=True,
        )

    monkeypatch.setattr("spendwise_rag.graphs.retrieval_graph.query_pinecone_records_detailed", fake_pinecone_search)

    result = hybrid_search_node(state)

    assert queried_namespaces == ["amex_2026"]
    assert result["pinecone_namespaces"] == ["amex_2026"]
    assert result["vector_warning"] == ""


def test_hybrid_search_queries_multiple_chunk_namespaces(monkeypatch):
    chunks = [
        Chunk(
            "2026-01-22 | Dunkin | $13.54 | Dining",
            {"chunk_type": "transaction", "merchant": "Dunkin", "category": "Dining", "namespace": "amex_2026"},
        ),
        Chunk(
            "2026-05-03 | Spotify | $9.99 | Subscriptions",
            {"chunk_type": "transaction", "merchant": "Spotify", "category": "Subscriptions", "namespace": "chase_2026"},
        ),
    ]
    index = build_index(chunks, "active_session")
    state = query_node({"query": "dining spotify", "local_index": index, "namespace": "active_session"})
    queried_namespaces = []

    def fake_pinecone_search(query, namespace, top_k=20, metadata_filter=None):
        queried_namespaces.append(namespace)
        return PineconeQueryResult([], warning=f"Pinecone returned no vector matches for namespace `{namespace}`.", used=True)

    monkeypatch.setattr("spendwise_rag.graphs.retrieval_graph.query_pinecone_records_detailed", fake_pinecone_search)

    result = hybrid_search_node(state)

    assert queried_namespaces == ["amex_2026", "amex_2026", "chase_2026", "chase_2026"]
    assert result["pinecone_namespaces"] == ["amex_2026", "chase_2026"]
    assert "amex_2026" in result["vector_warning"]
    assert "chase_2026" in result["vector_warning"]


def test_hybrid_search_sets_vector_error_and_uses_bm25_only_when_pinecone_fails(monkeypatch):
    chunks = [
        Chunk(
            "2026-05-07 | Trader Joe's | $42.10 | Groceries",
            {"chunk_type": "transaction", "merchant": "Trader Joe's", "category": "Groceries", "namespace": "test"},
        )
    ]
    index = build_index(chunks, "test")
    state = query_node({"query": "Trader Joe groceries", "local_index": index, "namespace": "test"})

    def fake_pinecone_search(query, namespace, top_k=20, metadata_filter=None):
        return PineconeQueryResult([], error="Pinecone vector search failed", used=True)

    monkeypatch.setattr("spendwise_rag.graphs.retrieval_graph.query_pinecone_records_detailed", fake_pinecone_search)

    result = hybrid_search_node(state)

    assert result["vector_results"] == []
    assert result["vector_error"] == "Pinecone vector search failed"
    assert result["merged_results"][0]["sources"] == ["bm25"]
    assert result["merged_results"][0]["text"] == "2026-05-07 | Trader Joe's | $42.10 | Groceries"


def test_hybrid_search_sets_vector_warning_for_zero_pinecone_hits(monkeypatch):
    chunks = [
        Chunk(
            "2026-05-03 | Spotify | $9.99 | Subscriptions",
            {"chunk_type": "transaction", "merchant": "Spotify", "category": "Subscriptions", "namespace": "test"},
        )
    ]
    index = build_index(chunks, "test")
    state = query_node({"query": "Spotify subscription", "local_index": index, "namespace": "test"})

    def fake_pinecone_search(query, namespace, top_k=20, metadata_filter=None):
        return PineconeQueryResult([], warning="Pinecone returned no vector matches for namespace `test`.", used=True)

    monkeypatch.setattr("spendwise_rag.graphs.retrieval_graph.query_pinecone_records_detailed", fake_pinecone_search)

    result = hybrid_search_node(state)

    assert result["vector_results"] == []
    assert result["vector_warning"] == "Pinecone returned no vector matches for namespace `test`."
    assert result["merged_results"][0]["sources"] == ["bm25"]


def test_hybrid_search_retries_pinecone_without_filters_when_filtered_search_is_empty(monkeypatch):
    chunks = [
        Chunk(
            "2026-04-25 | Frontier Airlines | $68.98 | Travel",
            {"chunk_type": "transaction", "merchant": "Frontier Airlines", "category": "Travel", "namespace": "amex_2026"},
        )
    ]
    index = build_index(chunks, "active_session")
    state = query_node({"query": "travel total", "local_index": index, "namespace": "active_session"})
    calls = []

    def fake_pinecone_search(query, namespace, top_k=20, metadata_filter=None):
        calls.append(metadata_filter)
        if metadata_filter:
            return PineconeQueryResult([], warning=f"Pinecone returned no vector matches for namespace `{namespace}`.", used=True)
        return PineconeQueryResult(
            [
                {
                    "id": "amex_2026_1",
                    "text": "2026-04-25 | Frontier Airlines | $68.98 | Travel",
                    "metadata": {"chunk_type": "transaction", "merchant": "Frontier Airlines", "category": "Travel", "namespace": namespace},
                    "score": 0.9,
                    "rank": 1,
                    "source": "vector",
                }
            ],
            used=True,
        )

    monkeypatch.setattr("spendwise_rag.graphs.retrieval_graph.query_pinecone_records_detailed", fake_pinecone_search)

    result = hybrid_search_node(state)

    assert calls == [{"category": "Travel"}, {}]
    assert result["vector_results"][0]["text"] == "2026-04-25 | Frontier Airlines | $68.98 | Travel"
    assert result["vector_warning"] == ""


def test_hybrid_search_does_not_use_local_vector_fallback_when_bm25_has_no_matches(monkeypatch):
    chunks = [
        Chunk(
            "2026-05-03 | Spotify | $9.99 | Subscriptions",
            {"chunk_type": "transaction", "merchant": "Spotify", "category": "Subscriptions", "namespace": "test"},
        )
    ]
    index = build_index(chunks, "test")
    state = query_node({"query": "semantic streaming bill", "local_index": index, "namespace": "test"})

    def fake_pinecone_search(query, namespace, top_k=20, metadata_filter=None):
        return PineconeQueryResult([], error="Pinecone unavailable", used=False)

    monkeypatch.setattr("spendwise_rag.graphs.retrieval_graph.query_pinecone_records_detailed", fake_pinecone_search)

    result = hybrid_search_node(state)

    assert result["bm25_results"] == []
    assert result["vector_results"] == []
    assert result["merged_results"] == []
    assert result["vector_error"] == "Pinecone unavailable"


def test_confidence_check_routes_low_scores_to_reranker():
    assert confidence_check_node({"original_query": "unknown", "confidence_score": 0.31}) == "rerank_node"


def test_context_validation_blocks_off_topic_chunks():
    state = context_validation_node(
        {
            "original_query": "Spotify subscription",
            "merged_results": [
                {
                    "text": "2026-05-07 | Trader Joe's | $42.10 | Groceries",
                    "metadata": {"chunk_type": "transaction"},
                    "score": 1.0,
                }
            ],
            "rerank_used": False,
        }
    )

    assert state["final_context"] == []
    assert state["response_message"] == "No relevant data found for that question in the indexed statements."


def test_context_validation_numbers_citations():
    state = context_validation_node(
        {
            "original_query": "Spotify subscription",
            "merged_results": [
                {
                    "text": "2026-05-03 | Spotify | $9.99 | Subscriptions",
                    "metadata": {"chunk_type": "transaction"},
                    "score": 1.0,
                }
            ],
            "rerank_used": False,
        }
    )

    assert state["final_context"][0]["citation"] == 1
    assert state["context_string"] == "[1] 2026-05-03 | Spotify | $9.99 | Subscriptions"
