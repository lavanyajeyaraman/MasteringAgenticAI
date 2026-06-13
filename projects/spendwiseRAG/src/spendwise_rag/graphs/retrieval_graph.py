from __future__ import annotations

import logging
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from spendwise_rag.core.models import LocalIndex
from spendwise_rag.retrieval import search, tokenize
from spendwise_rag.retrieval.bm25_store import load_bm25_payload, search_bm25_payload
from spendwise_rag.services.vector_store import query_pinecone_records_detailed


logger = logging.getLogger(__name__)


AIRFARE_MERCHANTS = ["Frontier Airlines", "Delta Air Lines", "American Airlines", "Spirit Airlines"]
QUERY_STOP_WORDS = {
    "all",
    "and",
    "are",
    "did",
    "for",
    "how",
    "into",
    "much",
    "show",
    "the",
    "total",
    "what",
    "with",
    "you",
}

MERCHANT_ALIASES = {
    "food delivery": ["Uber Eats", "DoorDash", "Grubhub", "Postmates"],
    "airfare": [*AIRFARE_MERCHANTS, "flight", "airline"],
    "airline": [*AIRFARE_MERCHANTS, "flight", "airfare"],
    "flight": [*AIRFARE_MERCHANTS, "airline", "airfare"],
    "rideshare": ["Uber", "Lyft"],
    "ride share": ["Uber", "Lyft"],
    "coffee": ["Starbucks", "Dunkin"],
}

CATEGORY_SYNONYMS = {
    "entertainment": ["Netflix", "Spotify", "movies", "concerts", "streaming"],
    "dining": ["restaurants", "food delivery", "Uber Eats", "DoorDash"],
    "dining out": ["restaurants", "food delivery", "Uber Eats", "DoorDash"],
    "groceries": ["Publix", "Trader Joe", "Whole Foods", "Costco", "Aldi"],
    "subscriptions": ["Netflix", "Spotify", "Hulu", "Apple"],
    "travel": ["airlines", "hotel", "parking", "Uber", "Lyft"],
}

INTENT_PATTERNS = {
    "comparison": ("compare", "vs", "versus", "difference", "more than"),
    "category_total": ("total", "sum", "spend", "spent", "category"),
    "merchant_search": ("merchant", "store", "vendor", "where", "uber", "spotify", "publix", "amazon"),
    "date_range": ("last month", "this month", "between", "from", "to", "january", "february", "march", "april", "may", "june"),
}


class RetrievalState(TypedDict, total=False):
    query: str
    original_query: str
    expanded_query: str
    query_intent: str
    namespace: str
    card_type: str
    local_index: LocalIndex
    bm25_index_path: str
    bm25_results: list[dict[str, Any]]
    vector_results: list[dict[str, Any]]
    pinecone_namespaces: list[str]
    merged_results: list[dict[str, Any]]
    confidence_score: float
    reranked_results: list[dict[str, Any]]
    rerank_used: bool
    final_context: list[dict[str, Any]]
    context_string: str
    response_message: str
    metadata_filters: dict[str, object]
    retrieval_log: list[dict[str, Any]]
    pinecone_used: bool
    vector_error: str
    vector_warning: str


def _month_name_for_offset(today: date, month_offset: int) -> str:
    month = today.month + month_offset
    year = today.year
    while month <= 0:
        month += 12
        year -= 1
    while month > 12:
        month -= 12
        year += 1
    return date(year, month, 1).strftime("%B %Y")


def detect_intent(query: str) -> str:
    lowered = query.lower()
    for intent, needles in INTENT_PATTERNS.items():
        if any(needle in lowered for needle in needles):
            return intent
    if re.search(r"\$\d|\b\d+\.\d{2}\b", lowered):
        return "spending_lookup"
    return "spending_lookup"


def expand_financial_query(query: str, today: date | None = None) -> tuple[str, str, dict[str, object]]:
    today = today or date.today()
    lowered = query.lower()
    additions: list[str] = []
    metadata_filters: dict[str, object] = {}

    for phrase, aliases in MERCHANT_ALIASES.items():
        if phrase in lowered:
            additions.extend(aliases)

    for category, synonyms in CATEGORY_SYNONYMS.items():
        if category in lowered:
            additions.extend(synonyms)
            metadata_filters["category"] = category.title() if category != "dining" else "Dining"

    if "last month" in lowered:
        last_month = _month_name_for_offset(today, -1)
        additions.append(last_month)
        metadata_filters["statement_month"] = last_month
    elif "this month" in lowered:
        this_month = _month_name_for_offset(today, 0)
        additions.append(this_month)
        metadata_filters["statement_month"] = this_month

    for card_type in ("amex", "chase", "discover", "generic_card"):
        if card_type.replace("_", " ") in lowered or card_type in lowered:
            metadata_filters["card_type"] = card_type

    if any(term in lowered for term in ("airfare", "airline", "flight")):
        metadata_filters["merchant"] = {"$in": AIRFARE_MERCHANTS}

    if any(term in lowered for term in ("charge", "charges", "transaction", "transactions", "spend", "spent", "amount")):
        metadata_filters["chunk_type"] = "transaction"

    expanded = " ".join([query, *additions])
    expanded = re.sub(r"\s+", " ", expanded).strip()
    return expanded, detect_intent(query), metadata_filters


def query_node(state: RetrievalState) -> RetrievalState:
    original_query = str(state.get("query") or state.get("original_query") or "").strip()
    expanded_query, query_intent, metadata_filters = expand_financial_query(original_query)
    return {
        **state,
        "original_query": original_query,
        "expanded_query": expanded_query,
        "query_intent": query_intent,
        "metadata_filters": {**metadata_filters, **state.get("metadata_filters", {})},
    }


def _result_from_chunk(chunk: Any, score: float, rank: int, source: str) -> dict[str, Any]:
    return {
        "id": f"{chunk.metadata.get('namespace', 'local')}_{rank}_{abs(hash(chunk.text))}",
        "text": chunk.text,
        "metadata": chunk.metadata,
        "score": float(score),
        "rank": rank,
        "source": source,
    }


def _metadata_matches(metadata: dict[str, Any], filters: dict[str, object]) -> bool:
    for key, expected in filters.items():
        if not expected:
            continue
        value = metadata.get(key)
        if isinstance(expected, dict):
            if "$in" in expected and value not in expected["$in"]:
                return False
            if "$eq" in expected and value != expected["$eq"]:
                return False
            continue
        if value != expected:
            return False
    return True


def _local_bm25_results(state: RetrievalState, query: str, filters: dict[str, object]) -> list[dict[str, Any]]:
    path = state.get("bm25_index_path")
    if path and Path(path).exists():
        results = search_bm25_payload(load_bm25_payload(path), query, top_k=20)
        return [item for item in results if _metadata_matches(item.get("metadata", {}), filters)]

    index = state.get("local_index")
    if not index:
        return []
    results = []
    for rank, (chunk, score) in enumerate(search(index, query, top_k=20), start=1):
        if _metadata_matches(chunk.metadata, filters):
            results.append(_result_from_chunk(chunk, score, rank, "bm25"))
    return results


def _rrf_merge(bm25_results: list[dict[str, Any]], vector_results: list[dict[str, Any]], k: int = 60) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for ranked_list in (bm25_results, vector_results):
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("text", "") or item.get("id", "")
            if key not in merged:
                merged[key] = {**item, "rrf_score": 0.0, "sources": set()}
            merged[key]["rrf_score"] += 1 / (k + rank)
            merged[key]["sources"].add(item.get("source", "unknown"))

    ranked = sorted(merged.values(), key=lambda item: float(item["rrf_score"]), reverse=True)
    if not ranked:
        return []
    top_score = float(ranked[0]["rrf_score"]) or 1.0
    normalized = []
    for rank, item in enumerate(ranked[:10], start=1):
        sources = sorted(item.pop("sources", set()))
        normalized.append(
            {
                **item,
                "rank": rank,
                "score": float(item["rrf_score"]) / top_score,
                "sources": sources,
            }
        )
    return normalized


def _confidence_from_results(query: str, merged_results: list[dict[str, Any]]) -> float:
    if not merged_results:
        return 0.0
    top = merged_results[0]
    source_count = len(top.get("sources", []))
    base = 0.62 + (0.18 if source_count >= 2 else 0.0)
    query_terms = _query_keywords(query)
    if query_terms:
        overlap = len(query_terms & set(tokenize(top.get("text", "")))) / len(query_terms)
        base += min(0.18, overlap * 0.18)
    return min(0.99, base)


def _pinecone_namespaces(state: RetrievalState) -> list[str]:
    explicit = str(state.get("namespace") or "").strip()
    index = state.get("local_index")
    chunk_namespaces = []
    if index:
        chunk_namespaces = sorted(
            {
                str(chunk.metadata.get("namespace", "")).strip()
                for chunk in index.chunks
                if str(chunk.metadata.get("namespace", "")).strip()
            }
        )
    if chunk_namespaces:
        return chunk_namespaces
    return [explicit] if explicit else []


def _pinecone_vector_results(
    query: str,
    namespaces: list[str],
    filters: dict[str, object],
) -> tuple[list[dict[str, Any]], bool, str, str]:
    if not namespaces:
        return [], False, "Pinecone namespace is not available for vector search.", ""

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    used = False
    for namespace in namespaces:
        pinecone_result = query_pinecone_records_detailed(
            query,
            namespace=namespace,
            top_k=20,
            metadata_filter=filters,
        )
        if not pinecone_result.matches and filters and not pinecone_result.error and not _has_strict_semantic_filter(filters):
            pinecone_result = query_pinecone_records_detailed(
                query,
                namespace=namespace,
                top_k=20,
                metadata_filter={},
            )
        used = used or pinecone_result.used
        results.extend(pinecone_result.matches)
        if pinecone_result.error:
            errors.append(pinecone_result.error)
        if pinecone_result.warning:
            warnings.append(pinecone_result.warning)

    if results:
        return results, used, "; ".join(errors), ""
    if errors:
        return [], used, "; ".join(errors), ""
    return [], used, "", "; ".join(warnings) or f"Pinecone returned no vector matches for namespaces: {', '.join(namespaces)}."


def _has_strict_semantic_filter(filters: dict[str, object]) -> bool:
    merchant_filter = filters.get("merchant")
    return isinstance(merchant_filter, dict) and bool(merchant_filter.get("$in"))


def hybrid_search_node(state: RetrievalState) -> RetrievalState:
    expanded_query = str(state.get("expanded_query", ""))
    filters = dict(state.get("metadata_filters", {}))
    if state.get("card_type"):
        filters["card_type"] = state["card_type"]
    bm25_results = _local_bm25_results(state, expanded_query, filters)
    pinecone_namespaces = _pinecone_namespaces(state)
    vector_results, pinecone_used, vector_error, vector_warning = _pinecone_vector_results(
        expanded_query,
        pinecone_namespaces,
        filters,
    )

    merged_results = _rrf_merge(bm25_results, vector_results)
    confidence_score = _confidence_from_results(expanded_query, merged_results)
    return {
        **state,
        "bm25_results": bm25_results,
        "vector_results": vector_results,
        "pinecone_namespaces": pinecone_namespaces,
        "merged_results": merged_results,
        "confidence_score": confidence_score,
        "pinecone_used": pinecone_used,
        "vector_error": vector_error,
        "vector_warning": vector_warning,
    }


def confidence_check_node(state: RetrievalState) -> str:
    score = float(state.get("confidence_score", 0.0))
    path_taken = "direct" if score >= 0.70 else "reranked"
    logger.info(
        "retrieval_confidence query=%r confidence_score=%.3f path_taken=%s",
        state.get("original_query", ""),
        score,
        path_taken,
    )
    return "generate_node" if score >= 0.70 else "rerank_node"


def generate_node(state: RetrievalState) -> RetrievalState:
    return {**state, "rerank_used": False}


def _lexical_rerank_score(query: str, text: str) -> float:
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(text))
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def rerank_node(state: RetrievalState) -> RetrievalState:
    start = time.perf_counter()
    expanded_query = str(state.get("expanded_query", ""))
    candidates = list(state.get("merged_results", []))[:10]
    scored: list[tuple[dict[str, Any], float]] = []
    try:
        from sentence_transformers import CrossEncoder

        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        model = CrossEncoder(model_name)
        scores = model.predict([(expanded_query, item["text"]) for item in candidates])
        scored = [(item, float(score)) for item, score in zip(candidates, scores)]
    except Exception:
        scored = [(item, _lexical_rerank_score(expanded_query, item.get("text", ""))) for item in candidates]

    scored.sort(key=lambda item: item[1], reverse=True)
    latency = time.perf_counter() - start
    if latency > 2.0:
        logger.warning("reranker_latency_seconds=%.3f exceeds target threshold", latency)
    reranked = [
        {**item, "cross_encoder_score": score, "rank": rank, "source": "reranker"}
        for rank, (item, score) in enumerate(scored[:5], start=1)
    ]
    return {**state, "reranked_results": reranked, "rerank_used": True}


def _query_keywords(query: str) -> set[str]:
    return {term for term in tokenize(query) if len(term) > 2 and term not in QUERY_STOP_WORDS}


def context_validation_node(state: RetrievalState) -> RetrievalState:
    candidates = state.get("reranked_results") if state.get("rerank_used") else state.get("merged_results")
    selected = list(candidates or [])[:5]
    if not selected:
        return {
            **state,
            "final_context": [],
            "context_string": "",
            "response_message": "No data found for that question in the indexed statements.",
        }

    keywords = _query_keywords(str(state.get("original_query", "") + " " + state.get("expanded_query", "")))
    if keywords:
        selected = [
            item
            for item in selected
            if keywords
            & set(
                tokenize(
                    item.get("text", "")
                    + " "
                    + " ".join(str(value) for value in item.get("metadata", {}).values())
                )
            )
        ]
    if not selected:
        return {
            **state,
            "final_context": [],
            "context_string": "",
            "response_message": "No relevant data found for that question in the indexed statements.",
        }

    final_context = []
    context_lines = []
    for idx, item in enumerate(selected[:5], start=1):
        numbered = {**item, "citation": idx}
        final_context.append(numbered)
        context_lines.append(f"[{idx}] {item['text']}")
    return {
        **state,
        "final_context": final_context,
        "context_string": "\n".join(context_lines),
        "response_message": "",
    }


def build_retrieval_graph():
    graph = StateGraph(RetrievalState)
    graph.add_node("query_node", query_node)
    graph.add_node("hybrid_search_node", hybrid_search_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("rerank_node", rerank_node)
    graph.add_node("context_validation_node", context_validation_node)
    graph.add_edge(START, "query_node")
    graph.add_edge("query_node", "hybrid_search_node")
    graph.add_conditional_edges(
        "hybrid_search_node",
        confidence_check_node,
        {
            "generate_node": "generate_node",
            "rerank_node": "rerank_node",
        },
    )
    graph.add_edge("generate_node", "context_validation_node")
    graph.add_edge("rerank_node", "context_validation_node")
    graph.add_edge("context_validation_node", END)
    return graph.compile()
