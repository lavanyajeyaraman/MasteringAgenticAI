from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spendwise_rag.core.models import Chunk, LocalIndex
from spendwise_rag.graphs.ingestion_graph import build_ingestion_graph
from spendwise_rag.graphs.retrieval_graph import build_retrieval_graph
from spendwise_rag.processing.analytics import transaction_frame
from spendwise_rag.retrieval import build_index, save_index
from spendwise_rag.services.aggregate import answer_aggregate_question
from spendwise_rag.services.llm import answer_with_configured_model, plan_spending_query_with_ollama


@dataclass(frozen=True)
class IngestionSummary:
    filename: str
    namespace: str
    chunks_indexed: int
    transaction_rows: int
    summary_chunks: int
    rollup_chunks: int
    embedding_model: str = "local BM25"
    pinecone_upserts: int = 0
    pinecone_namespace_cleared: bool = False
    errors: tuple[str, ...] = ()


def summarize_chunks(filename: str, namespace: str, chunks: list[Chunk]) -> IngestionSummary:
    chunk_types = [chunk.metadata.get("chunk_type") for chunk in chunks]
    return IngestionSummary(
        filename=filename,
        namespace=namespace,
        chunks_indexed=len(chunks),
        transaction_rows=chunk_types.count("transaction"),
        summary_chunks=chunk_types.count("summary"),
        rollup_chunks=chunk_types.count("rollup"),
    )


def build_local_index(
    pdf_bytes: bytes,
    filename: str,
    index_dir: str | Path = "data/indexes",
    card_type_override: str | None = None,
) -> tuple[LocalIndex, IngestionSummary]:
    graph = build_ingestion_graph()
    result = graph.invoke(
        {
            "pdf_bytes": pdf_bytes,
            "filename": filename,
            "card_type_override": card_type_override,
        }
    )
    index = result["local_index"]
    save_index(index, Path(index_dir) / f"{index.namespace}.pkl")
    summary = result["summary"]
    return index, IngestionSummary(
        filename=summary["filename"],
        namespace=summary["namespace"],
        chunks_indexed=summary["chunks_indexed"],
        transaction_rows=summary["transaction_rows"],
        summary_chunks=summary["summary_chunks"],
        rollup_chunks=summary["rollup_chunks"],
        embedding_model=summary["embedding_model"],
        pinecone_upserts=summary["pinecone_upserts"],
        pinecone_namespace_cleared=bool(summary.get("pinecone_namespace_cleared", False)),
        errors=tuple(summary.get("errors", [])),
    )


def combine_indexes(indexes: list[LocalIndex], namespace: str = "active_session") -> LocalIndex:
    chunks = [chunk for index in indexes for chunk in index.chunks]
    return build_index(chunks, namespace)


def answer_question(index: LocalIndex, question: str, top_k: int = 6) -> dict[str, object]:
    aggregate_answer = answer_aggregate_question(index, question)
    if aggregate_answer:
        return aggregate_answer

    query_plan = _plan_spending_question(index, question)
    aggregate_answer = answer_aggregate_question(index, question, query_plan=query_plan)
    if aggregate_answer:
        return aggregate_answer

    retrieval = build_retrieval_graph().invoke(
        {
            "query": question,
            "local_index": index,
            "namespace": index.namespace,
        }
    )
    diagnostics = {
        "pinecone_used": bool(retrieval.get("pinecone_used", False)),
        "pinecone_namespaces": list(retrieval.get("pinecone_namespaces", [])),
        "vector_error": str(retrieval.get("vector_error", "")),
        "vector_warning": str(retrieval.get("vector_warning", "")),
    }
    final_context = list(retrieval.get("final_context", []))[:top_k]
    if not final_context:
        message = str(retrieval.get("response_message") or "No data found for that question in the indexed statements.")
        return {
            "answer": message,
            "model_provider": "retrieval",
            "confidence": float(retrieval.get("confidence_score", 0.0)),
            "faithfulness": 0.0,
            "rerank_used": bool(retrieval.get("rerank_used", False)),
            "matches": [],
            "retrieval_diagnostics": diagnostics,
        }

    retrieved_total = _answer_retrieved_transaction_total(question, final_context, diagnostics)
    if retrieved_total:
        return retrieved_total

    evidence = [str(item["text"]) for item in final_context]
    answer, model_provider = answer_with_configured_model(question, evidence)
    if not answer:
        answer = "No local or cloud model response is available. Here is the retrieved statement evidence to review."
    confidence = float(retrieval.get("confidence_score", 0.0))
    faithfulness = 0.95 if evidence else 0.0
    return {
        "answer": answer,
        "model_provider": model_provider,
        "confidence": confidence,
        "faithfulness": faithfulness,
        "rerank_used": bool(retrieval.get("rerank_used", False)),
        "matches": [
            {"text": item["text"], "score": item.get("score", 0.0), "metadata": item.get("metadata", {})}
            for item in final_context
        ],
        "retrieval_diagnostics": diagnostics,
    }


def _plan_spending_question(index: LocalIndex, question: str) -> dict[str, object] | None:
    lowered = question.lower()
    planning_terms = ("compare", " vs ", " versus ", "which", "highest", "higher", "more", "spending")
    if not any(term in lowered for term in planning_terms):
        return None

    df = transaction_frame(index)
    if df.empty:
        return None
    categories = sorted(df["category"].dropna().astype(str).unique().tolist())
    return plan_spending_query_with_ollama(question, categories)


def _answer_retrieved_transaction_total(
    question: str,
    final_context: list[dict[str, Any]],
    diagnostics: dict[str, object],
) -> dict[str, object] | None:
    lowered = question.lower()
    if not any(term in lowered for term in ("total", "how much", "sum", "amount")):
        return None
    if not any(term in lowered for term in ("charge", "charges", "transaction", "transactions", "spend", "spent")):
        return None

    transaction_context = [
        item
        for item in final_context
        if item.get("metadata", {}).get("chunk_type") == "transaction"
        and item.get("metadata", {}).get("amount") is not None
    ]
    if not transaction_context:
        return None

    total = sum(float(item["metadata"]["amount"]) for item in transaction_context)
    lines = [
        f"- {item['text']}"
        for item in sorted(
            transaction_context,
            key=lambda item: (
                str(item.get("metadata", {}).get("date", "")),
                str(item.get("metadata", {}).get("merchant", "")),
                float(item.get("metadata", {}).get("amount", 0.0)),
            ),
        )
    ]
    answer = (
        "Matching charges:\n"
        + "\n".join(lines)
        + f"\n\nTotal: ${total:,.2f} across {len(transaction_context)} transaction(s)."
    )
    return {
        "answer": answer,
        "model_provider": "deterministic_retrieval_math",
        "confidence": 0.95,
        "faithfulness": 1.0,
        "rerank_used": False,
        "matches": [
            {"text": item["text"], "score": item.get("score", 0.0), "metadata": item.get("metadata", {})}
            for item in transaction_context
        ],
        "retrieval_diagnostics": diagnostics,
    }
