from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    nebius_api_key: str | None = None
    llm_model_extract: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm_model_reasoning: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    llm_model_verify: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pinecone_api_key: str | None = None
    pinecone_index: str = "semanticatsrecruit"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60
    top_k: int = 20
    faithfulness_threshold: float = 0.70

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
