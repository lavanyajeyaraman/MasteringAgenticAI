from spendwise_rag.core.models import Chunk
from spendwise_rag.services.llm import answer_with_configured_model, ollama_status, suggest_categories_with_ollama
from spendwise_rag.services.vector_store import (
    clear_namespace_enabled,
    chunks_to_pinecone_records,
    query_pinecone_records_detailed,
    upsert_records_to_pinecone,
)


def test_configured_model_uses_ollama(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:1")

    answer, provider = answer_with_configured_model("How much?", ["2026-05-03 | Spotify | $9.99"])

    assert answer == ""
    assert provider == "ollama"


def test_chunks_to_pinecone_records_maps_text_field():
    chunks = [Chunk("hello", {"chunk_type": "summary", "namespace": "test"})]

    records = chunks_to_pinecone_records(chunks, "test")

    assert records == [
        {
            "_id": "test_0000",
            "text": "hello",
            "chunk_type": "summary",
            "namespace": "test",
        }
    ]


def test_pinecone_upsert_skips_without_api_key(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    result = upsert_records_to_pinecone([{"_id": "1", "text": "hello"}], "test")

    assert result.count == 0
    assert result.error == ""
    assert result.namespace_cleared is False


def test_pinecone_query_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    result = query_pinecone_records_detailed("spotify", "test")

    assert result.matches == []
    assert result.error == "PINECONE_API_KEY is not configured."
    assert result.used is False


def test_pinecone_query_uses_integrated_search_signature(monkeypatch):
    calls = {}

    class FakeIndex:
        def search(self, **kwargs):
            calls.update(kwargs)
            return {
                "result": {
                    "hits": [
                        {
                            "_id": "amex_2026_0001",
                            "_score": 0.91,
                            "fields": {
                                "text": "2026-04-25 | Frontier Airlines | $68.98 | Travel",
                                "category": "Travel",
                            },
                        }
                    ]
                }
            }

    class FakePinecone:
        def __init__(self, api_key):
            self.api_key = api_key

        def Index(self, index_name):
            calls["index_name"] = index_name
            return FakeIndex()

    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX", "smartspend")
    monkeypatch.setattr("pinecone.Pinecone", FakePinecone)

    result = query_pinecone_records_detailed("travel total", "amex_2026", metadata_filter={"category": "Travel"})

    assert calls["index_name"] == "smartspend"
    assert calls["namespace"] == "amex_2026"
    assert calls["top_k"] == 20
    assert calls["inputs"] == {"text": "travel total"}
    assert calls["filter"] == {"category": {"$eq": "Travel"}}
    assert result.matches[0]["text"] == "2026-04-25 | Frontier Airlines | $68.98 | Travel"
    assert result.matches[0]["metadata"] == {"category": "Travel"}
    assert result.warning == ""


def test_pinecone_query_parses_object_hits_with_fields(monkeypatch):
    calls = {}

    class FakeHit:
        id = "amex_2026_0018"
        score = 0.50
        fields = {
            "text": "2026-04-16 | Publix | $30.28 | Groceries",
            "amount": 30.28,
            "category": "Groceries",
        }

    class FakeResult:
        hits = [FakeHit()]

    class FakeResponse:
        result = FakeResult()

    class FakeIndex:
        def search(self, **kwargs):
            calls.update(kwargs)
            return FakeResponse()

    class FakePinecone:
        def __init__(self, api_key):
            self.api_key = api_key

        def Index(self, index_name):
            return FakeIndex()

    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setattr("pinecone.Pinecone", FakePinecone)

    result = query_pinecone_records_detailed("groceries publix total", "amex_2026")

    assert calls["namespace"] == "amex_2026"
    assert result.matches == [
        {
            "id": "amex_2026_0018",
            "text": "2026-04-16 | Publix | $30.28 | Groceries",
            "metadata": {"amount": 30.28, "category": "Groceries"},
            "score": 0.5,
            "rank": 1,
            "source": "vector",
        }
    ]
    assert result.warning == ""


def test_clear_namespace_defaults_to_true(monkeypatch):
    monkeypatch.delenv("PINECONE_CLEAR_NAMESPACE_BEFORE_UPSERT", raising=False)

    assert clear_namespace_enabled() is True


def test_category_suggestions_fail_closed_when_ollama_unavailable(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:1")

    suggestions = suggest_categories_with_ollama(["Mystery Merchant"], ["Groceries", "Uncategorized"])

    assert suggestions == []


def test_category_suggestions_accept_case_variants(monkeypatch):
    class FakeResponse:
        content = '[{"merchant":"shell gas","category":"auto","reason":"gas station"}]'

    class FakeChat:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt):
            return FakeResponse()

    monkeypatch.setattr("spendwise_rag.services.llm.ChatOllama", FakeChat)

    suggestions = suggest_categories_with_ollama(["Shell Gas"], ["Auto", "Uncategorized"])

    assert suggestions == [{"merchant": "Shell Gas", "suggested_category": "Auto", "reason": "gas station"}]


def test_ollama_status_reports_unreachable_server(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:1")

    ready, message = ollama_status()

    assert ready is False
    assert "not reachable" in message
