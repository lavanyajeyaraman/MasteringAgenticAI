from spendwise_rag.core.models import Chunk
from spendwise_rag.retrieval import build_index
from spendwise_rag.services.aggregate import answer_aggregate_question
from spendwise_rag.services.pipeline import answer_question
from spendwise_rag.services.vector_store import PineconeQueryResult


def _index():
    chunks = [
        Chunk(
            "2026-05-02 | Publix | $32.74 | Groceries",
            {
                "chunk_type": "transaction",
                "date": "2026-05-02",
                "merchant": "Publix",
                "amount": 32.74,
                "category": "Groceries",
                "statement_month": "May 2026",
                "namespace": "amex_2026",
            },
        ),
        Chunk(
            "2026-05-07 | Publix | $38.71 | Groceries",
            {
                "chunk_type": "transaction",
                "date": "2026-05-07",
                "merchant": "Publix",
                "amount": 38.71,
                "category": "Groceries",
                "statement_month": "May 2026",
                "namespace": "amex_2026",
            },
        ),
        Chunk(
            "2026-05-08 | Frontier Airlines | $11.20 | Travel",
            {
                "chunk_type": "transaction",
                "date": "2026-05-08",
                "merchant": "Frontier Airlines",
                "amount": 11.20,
                "category": "Travel",
                "statement_month": "May 2026",
                "namespace": "amex_2026",
            },
        ),
        Chunk(
            "2026-04-16 | Publix | $30.28 | Groceries",
            {
                "chunk_type": "transaction",
                "date": "2026-04-16",
                "merchant": "Publix",
                "amount": 30.28,
                "category": "Groceries",
                "statement_month": "May 2026",
                "namespace": "amex_2026",
            },
        ),
        Chunk(
            "2026-05-09 | Dunkin | $13.54 | Dining",
            {
                "chunk_type": "transaction",
                "date": "2026-05-09",
                "merchant": "Dunkin",
                "amount": 13.54,
                "category": "Dining",
                "statement_month": "May 2026",
                "namespace": "amex_2026",
            },
        ),
        Chunk(
            "2026-05-10 | Hashtag India | $38.50 | Dining",
            {
                "chunk_type": "transaction",
                "date": "2026-05-10",
                "merchant": "Hashtag India",
                "amount": 38.50,
                "category": "Dining",
                "statement_month": "May 2026",
                "namespace": "amex_2026",
            },
        ),
        Chunk(
            "2026-05-12 | Ponce De Leon Music | $33.82 | Entertainment",
            {
                "chunk_type": "transaction",
                "date": "2026-05-12",
                "merchant": "Ponce De Leon Music",
                "amount": 33.82,
                "category": "Entertainment",
                "statement_month": "May 2026",
                "namespace": "amex_2026",
            },
        ),
        Chunk(
            "2026-05-11 | Frontier Airlines | $134.98 | Travel",
            {
                "chunk_type": "transaction",
                "date": "2026-05-11",
                "merchant": "Frontier Airlines",
                "amount": 134.98,
                "category": "Travel",
                "statement_month": "May 2026",
                "namespace": "amex_2026",
            },
        ),
    ]
    return build_index(chunks, "active_session")


def test_aggregate_answers_total_spend_for_calendar_month():
    result = answer_aggregate_question(_index(), "how much I total i spend for may month?")

    assert result is not None
    assert result["answer"] == "You spent $303.49 for May 2026 across 7 transaction(s)."
    assert result["model_provider"] == "deterministic_analytics"
    assert len(result["matches"]) == 7


def test_aggregate_answers_category_total_across_all_transactions():
    result = answer_aggregate_question(_index(), "what is my total groceries spend?")

    assert result is not None
    assert result["answer"] == "You spent $101.73 for Groceries across 3 transaction(s)."
    assert len(result["matches"]) == 3


def test_pipeline_uses_aggregate_before_retrieval_for_total_questions():
    result = answer_question(_index(), "how much did I spend on groceries?")

    assert result["answer"] == "You spent $101.73 for Groceries across 3 transaction(s)."
    assert result["confidence"] == 0.99
    assert result["retrieval_diagnostics"]["pinecone_used"] is False


def test_aggregate_compares_two_categories_across_all_transactions():
    result = answer_aggregate_question(_index(), "Compare my Groceries spend vs Dining spend.")

    assert result is not None
    assert result["answer"] == (
        "Groceries: $101.73 across 3 transaction(s). "
        "Dining: $52.04 across 2 transaction(s). "
        "Groceries is higher by $49.69."
    )
    assert result["model_provider"] == "deterministic_analytics"
    assert result["retrieval_diagnostics"]["pinecone_used"] is False


def test_pipeline_uses_aggregate_for_comparison_questions():
    result = answer_question(_index(), "Compare my Groceries spend vs Dining spend.")

    assert "Groceries: $101.73" in result["answer"]
    assert "Dining: $52.04" in result["answer"]
    assert result["model_provider"] == "deterministic_analytics"


def test_aggregate_answers_top_merchants_by_total_spend():
    result = answer_aggregate_question(_index(), "What are my top 5 merchants by total spend?")

    assert result is not None
    assert result["answer"] == (
        "Top merchants by total spend:\n"
        "1. Frontier Airlines: $146.18 across 2 transaction(s)\n"
        "2. Publix: $101.73 across 3 transaction(s)\n"
        "3. Hashtag India: $38.50 across 1 transaction(s)\n"
        "4. Ponce De Leon Music: $33.82 across 1 transaction(s)\n"
        "5. Dunkin: $13.54 across 1 transaction(s)"
    )
    assert result["model_provider"] == "deterministic_analytics"
    assert result["retrieval_diagnostics"]["pinecone_used"] is False


def test_pipeline_uses_aggregate_for_top_merchants_questions():
    result = answer_question(_index(), "What are my top 5 merchants by total spend?")

    assert result["answer"].startswith("Top merchants by total spend:")
    assert "1. Frontier Airlines: $146.18" in result["answer"]
    assert result["model_provider"] == "deterministic_analytics"


def test_aggregate_answers_llm_planned_category_comparison():
    result = answer_aggregate_question(
        _index(),
        "Which one is the highest in spending? Eating out or entertainment?",
        query_plan={
            "intent": "compare_categories",
            "operation": "highest_spending",
            "categories": ["Dining", "Entertainment"],
            "planner": "ollama",
        },
    )

    assert result is not None
    assert result["answer"] == (
        "Dining: $52.04 across 2 transaction(s). "
        "Entertainment: $33.82 across 1 transaction(s). "
        "Dining is higher by $18.22."
    )
    assert result["model_provider"] == "llm_planned_deterministic_analytics"


def test_pipeline_uses_llm_plan_before_retrieval_for_ambiguous_comparison(monkeypatch):
    def fake_planner(question, allowed_categories):
        assert "Dining" in allowed_categories
        assert "Entertainment" in allowed_categories
        return {
            "intent": "compare_categories",
            "operation": "highest_spending",
            "categories": ["Dining", "Entertainment"],
            "planner": "ollama",
        }

    monkeypatch.setattr("spendwise_rag.services.pipeline.plan_spending_query_with_ollama", fake_planner)

    result = answer_question(_index(), "Which one is highest in spending? Eating out or entertainment?")

    assert "Dining: $52.04" in result["answer"]
    assert "Entertainment: $33.82" in result["answer"]
    assert result["model_provider"] == "llm_planned_deterministic_analytics"
    assert result["retrieval_diagnostics"]["pinecone_used"] is False


def test_pipeline_computes_total_from_semantically_retrieved_transactions(monkeypatch):
    def fake_pinecone_search(query, namespace, top_k=20, metadata_filter=None):
        return PineconeQueryResult(
            [
                {
                    "id": "amex_2026_1",
                    "text": "2026-05-08 | Frontier Airlines | $11.20 | Travel",
                    "metadata": {
                        "chunk_type": "transaction",
                        "date": "2026-05-08",
                        "merchant": "Frontier Airlines",
                        "amount": 11.2,
                        "category": "Travel",
                        "namespace": namespace,
                    },
                    "score": 0.9,
                    "rank": 1,
                    "source": "vector",
                },
                {
                    "id": "amex_2026_2",
                    "text": "2026-05-11 | Frontier Airlines | $134.98 | Travel",
                    "metadata": {
                        "chunk_type": "transaction",
                        "date": "2026-05-11",
                        "merchant": "Frontier Airlines",
                        "amount": 134.98,
                        "category": "Travel",
                        "namespace": namespace,
                    },
                    "score": 0.88,
                    "rank": 2,
                    "source": "vector",
                },
            ],
            used=True,
        )

    monkeypatch.setattr("spendwise_rag.graphs.retrieval_graph.query_pinecone_records_detailed", fake_pinecone_search)

    result = answer_question(_index(), "Show me all airfare charges and the total amount.")

    assert result["model_provider"] == "deterministic_retrieval_math"
    assert "2026-05-08 | Frontier Airlines | $11.20 | Travel" in result["answer"]
    assert "2026-05-11 | Frontier Airlines | $134.98 | Travel" in result["answer"]
    assert "Total: $146.18 across 2 transaction(s)." in result["answer"]


def test_aggregate_skips_unresolved_semantic_charge_queries():
    result = answer_aggregate_question(_index(), "Show me all airfare charges and the total amount.")

    assert result is None
