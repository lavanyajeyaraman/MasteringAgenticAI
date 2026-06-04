from src.llm import available_providers, build_chat_model, get_model_name, get_provider, get_provider_spec, provider_label


def test_groq_provider_defaults(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    assert get_provider() == "groq"
    assert provider_label() == "Groq"
    assert get_model_name() == "llama-3.3-70b-versatile"


def test_openai_provider_uses_openai_model(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    assert get_provider() == "openai"
    assert provider_label() == "OpenAI"
    assert get_model_name() == "gpt-test"


def test_gemini_provider_uses_gemini_model(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")

    assert get_provider() == "gemini"
    assert provider_label() == "Gemini"
    assert get_model_name() == "gemini-test"


def test_gemini_provider_default_model(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    assert get_model_name() == "gemini-2.0-flash"


def test_gemini_provider_sets_google_api_key_alias(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    model = build_chat_model()

    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert get_model_name() == "gemini-2.0-flash"


def test_ollama_provider_is_local_and_keyless(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    assert get_provider() == "ollama"
    assert provider_label() == "Ollama Local"
    assert get_model_name() == "llama3.1:8b"


def test_ollama_provider_builds_chat_model(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "test-local-model")

    model = build_chat_model()

    assert type(model).__name__ == "ChatOllama"


def test_provider_registry_exposes_env_contract():
    assert available_providers() == ["groq", "openai", "gemini", "ollama"]
    groq = get_provider_spec("groq")
    gemini = get_provider_spec("gemini")
    ollama = get_provider_spec("ollama")

    assert groq.api_key_env == "GROQ_API_KEY"
    assert groq.model_env == "GROQ_MODEL"
    assert gemini.api_key_env == "GEMINI_API_KEY"
    assert gemini.model_env == "GEMINI_MODEL"
    assert ollama.api_key_env is None
    assert ollama.model_env == "OLLAMA_MODEL"
