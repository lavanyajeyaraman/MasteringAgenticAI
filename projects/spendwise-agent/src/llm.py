from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    api_key_env: str | None
    model_env: str
    default_model: str


PROVIDERS = {
    "groq": ProviderSpec(
        name="groq",
        label="Groq",
        api_key_env="GROQ_API_KEY",
        model_env="GROQ_MODEL",
        default_model="llama-3.3-70b-versatile",
    ),
    "openai": ProviderSpec(
        name="openai",
        label="OpenAI",
        api_key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        default_model="gpt-4.1-mini",
    ),
    "gemini": ProviderSpec(
        name="gemini",
        label="Gemini",
        api_key_env="GEMINI_API_KEY",
        model_env="GEMINI_MODEL",
        default_model="gemini-2.0-flash",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        label="Ollama Local",
        api_key_env=None,
        model_env="OLLAMA_MODEL",
        default_model="llama3.1:8b",
    ),
}


def available_providers() -> list[str]:
    return list(PROVIDERS)


def get_provider() -> str:
    load_dotenv()
    provider = os.getenv("AI_PROVIDER", "groq").strip().lower()
    return provider if provider in PROVIDERS else "groq"


def get_provider_spec(provider: str | None = None) -> ProviderSpec:
    return PROVIDERS.get(provider or get_provider(), PROVIDERS["groq"])


def provider_label(provider: str | None = None) -> str:
    return get_provider_spec(provider).label


def get_model_name(provider: str | None = None) -> str:
    load_dotenv()
    spec = get_provider_spec(provider)
    return os.getenv(spec.model_env, spec.default_model)


def has_api_key(provider: str | None = None) -> bool:
    load_dotenv()
    spec = get_provider_spec(provider)
    if spec.api_key_env is None:
        return True
    return bool(os.getenv(spec.api_key_env))


def get_api_key(provider: str | None = None) -> str | None:
    load_dotenv()
    spec = get_provider_spec(provider)
    if spec.api_key_env is None:
        return None
    return os.getenv(spec.api_key_env)


def build_chat_model(temperature: float = 0.2, max_tokens: int = 450) -> BaseChatModel:
    load_dotenv()
    provider = get_provider()
    model = get_model_name(provider)
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = get_api_key("gemini")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=api_key,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, temperature=temperature, base_url=base_url)
    raise ValueError(f"Unsupported AI provider: {provider}")
