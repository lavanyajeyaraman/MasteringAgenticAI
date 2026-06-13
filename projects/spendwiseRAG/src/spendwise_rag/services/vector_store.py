from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from spendwise_rag.core.models import Chunk


@dataclass(frozen=True)
class PineconeUpsertResult:
    count: int
    error: str = ""
    namespace_cleared: bool = False


@dataclass(frozen=True)
class PineconeQueryResult:
    matches: list[dict[str, Any]]
    error: str = ""
    warning: str = ""
    used: bool = False


def chunks_to_pinecone_records(chunks: list[Chunk], namespace: str) -> list[dict[str, object]]:
    records = []
    for idx, chunk in enumerate(chunks):
        records.append(
            {
                "_id": f"{namespace}_{idx:04d}",
                "text": chunk.text,
                **chunk.metadata,
            }
        )
    return records


def clear_namespace_enabled() -> bool:
    return os.getenv("PINECONE_CLEAR_NAMESPACE_BEFORE_UPSERT", "true").lower().strip() in {"1", "true", "yes", "on"}


def upsert_records_to_pinecone(records: list[dict[str, object]], namespace: str) -> PineconeUpsertResult:
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "smartspend")
    if not api_key or not records:
        return PineconeUpsertResult(0)

    from pinecone import Pinecone

    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        index.describe_index_stats()
    except Exception as exc:
        return PineconeUpsertResult(
            0,
            f"Pinecone index `{index_name}` is unavailable: {exc}",
        )

    upserted = 0
    namespace_cleared = False
    try:
        if clear_namespace_enabled():
            index.delete(delete_all=True, namespace=namespace)
            namespace_cleared = True
        for start in range(0, len(records), 96):
            batch = records[start : start + 96]
            response = index.upsert_records(records=batch, namespace=namespace)
            upserted += int(getattr(response, "record_count", len(batch)))
    except Exception as exc:
        return PineconeUpsertResult(
            upserted,
            f"Pinecone integrated upsert failed after {upserted} records: {exc}",
            namespace_cleared=namespace_cleared,
        )
    return PineconeUpsertResult(upserted, namespace_cleared=namespace_cleared)


def query_pinecone_records(
    query: str,
    namespace: str,
    top_k: int = 20,
    metadata_filter: dict[str, object] | None = None,
) -> list[dict[str, Any]]:
    return query_pinecone_records_detailed(query, namespace, top_k, metadata_filter).matches


def query_pinecone_records_detailed(
    query: str,
    namespace: str,
    top_k: int = 20,
    metadata_filter: dict[str, object] | None = None,
) -> PineconeQueryResult:
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "smartspend")
    if not api_key:
        return PineconeQueryResult([], error="PINECONE_API_KEY is not configured.")
    if not namespace:
        return PineconeQueryResult([], error="Pinecone namespace is not available for vector search.")

    from pinecone import Pinecone

    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        response = index.search(
            namespace=namespace,
            top_k=top_k,
            inputs={"text": query},
            filter=_pinecone_filter(metadata_filter or {}),
        )
    except Exception as exc:
        return PineconeQueryResult([], error=f"Pinecone vector search failed for `{index_name}/{namespace}`: {exc}", used=True)

    matches = _pinecone_hits(response)
    results: list[dict[str, Any]] = []
    for rank, match in enumerate(matches, start=1):
        fields = _pinecone_hit_fields(match)
        text = str(fields.get("text", ""))
        if not text:
            continue
        results.append(
            {
                "id": _pinecone_hit_id(match, rank),
                "text": text,
                "metadata": {key: value for key, value in fields.items() if key != "text"},
                "score": _pinecone_hit_score(match),
                "rank": rank,
                "source": "vector",
            }
        )
    warning = "" if results else f"Pinecone returned no vector matches for namespace `{namespace}`."
    return PineconeQueryResult(results, warning=warning, used=True)


def _pinecone_filter(metadata_filter: dict[str, object]) -> dict[str, object]:
    normalized = {}
    for key, value in metadata_filter.items():
        if value in ("", None):
            continue
        if isinstance(value, dict):
            normalized[key] = value
        else:
            normalized[key] = {"$eq": value}
    return normalized


def _pinecone_hits(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return list(response.get("result", {}).get("hits", []))
    result = getattr(response, "result", None)
    if result is not None:
        hits = getattr(result, "hits", None)
        if hits is not None:
            return list(hits)
    matches = getattr(response, "matches", None)
    return list(matches or [])


def _pinecone_hit_fields(match: Any) -> dict[str, Any]:
    if isinstance(match, dict):
        return dict(match.get("fields") or match.get("metadata") or {})
    return dict(getattr(match, "fields", None) or getattr(match, "metadata", None) or {})


def _pinecone_hit_id(match: Any, rank: int) -> str:
    if isinstance(match, dict):
        return str(match.get("_id", match.get("id", f"pinecone_{rank}")))
    return str(getattr(match, "id", f"pinecone_{rank}"))


def _pinecone_hit_score(match: Any) -> float:
    if isinstance(match, dict):
        return float(match.get("_score", match.get("score", 0.0)))
    return float(getattr(match, "score", 0.0))
