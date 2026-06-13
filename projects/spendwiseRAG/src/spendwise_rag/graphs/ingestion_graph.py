from __future__ import annotations

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from spendwise_rag.core.state import IngestionState
from spendwise_rag.processing.chunking import chunk_node
from spendwise_rag.processing.ingestion import parse_node, upload_node
from spendwise_rag.retrieval import build_index, save_index
from spendwise_rag.retrieval.bm25_store import persist_bm25
from spendwise_rag.services.vector_store import chunks_to_pinecone_records, upsert_records_to_pinecone


def upload_graph_node(state: IngestionState) -> IngestionState:
    upload = upload_node(
        state["pdf_bytes"],
        state["filename"],
        card_type_override=state.get("card_type_override"),
    )
    return {
        **state,
        "card_type": str(upload["card_type"]),
        "statement_month": str(upload["statement_month"]),
        "statement_year": int(upload["statement_year"]),
        "card_namespace": str(upload["card_namespace"]),
        "bank_config": upload["bank_config"],
    }


def parse_graph_node(state: IngestionState) -> IngestionState:
    parsed = parse_node(
        state["pdf_bytes"],
        state["filename"],
        card_type_override=state.get("card_type_override"),
    )
    return {
        **state,
        "parsed_statement": parsed,
        "raw_tables": parsed.raw_tables,
        "raw_text": parsed.raw_text,
        "image_extractions": parsed.image_extractions,
        "excluded_regions": parsed.excluded_regions,
    }


def chunk_graph_node(state: IngestionState) -> IngestionState:
    chunks = chunk_node(state["parsed_statement"])
    documents = [Document(page_content=chunk.text, metadata=chunk.metadata) for chunk in chunks]
    return {**state, "chunks": chunks, "documents": documents}


def embed_graph_node(state: IngestionState) -> IngestionState:
    namespace = state["card_namespace"]
    records = chunks_to_pinecone_records(state.get("chunks", []), namespace)
    return {
        **state,
        "pinecone_records": records,
        "embedding_model": "pinecone_integrated_embedding",
    }


def upsert_graph_node(state: IngestionState) -> IngestionState:
    namespace = state["card_namespace"]
    chunks = state.get("chunks", [])
    records = state.get("pinecone_records", [])
    errors = list(state.get("errors", []))
    local_index = build_index(chunks, namespace)
    save_index(local_index, f"data/indexes/{namespace}.pkl")
    bm25_path = persist_bm25(chunks, namespace)
    upsert_result = upsert_records_to_pinecone(records, namespace) if records else None
    upsert_count = upsert_result.count if upsert_result else 0
    if upsert_result and upsert_result.error:
        errors.append(upsert_result.error)
    elif records and not upsert_count:
        errors.append("Pinecone integrated upsert skipped. Check PINECONE_API_KEY and PINECONE_INDEX.")
    chunk_types = [chunk.metadata.get("chunk_type") for chunk in chunks]
    summary = {
        "filename": state["filename"],
        "namespace": namespace,
        "chunks_indexed": len(chunks),
        "transaction_rows": chunk_types.count("transaction"),
        "summary_chunks": chunk_types.count("summary"),
        "rollup_chunks": chunk_types.count("rollup"),
        "embedding_model": state.get("embedding_model", ""),
        "pinecone_upserts": upsert_count,
        "pinecone_namespace_cleared": bool(upsert_result.namespace_cleared) if upsert_result else False,
        "bm25_path": bm25_path,
        "errors": errors,
    }
    return {
        **state,
        "local_index": local_index,
        "bm25_path": bm25_path,
        "upsert_count": upsert_count,
        "summary": summary,
        "errors": errors,
    }


def build_ingestion_graph():
    graph = StateGraph(IngestionState)
    graph.add_node("upload_node", upload_graph_node)
    graph.add_node("parse_node", parse_graph_node)
    graph.add_node("chunk_node", chunk_graph_node)
    graph.add_node("embed_node", embed_graph_node)
    graph.add_node("upsert_node", upsert_graph_node)
    graph.add_edge(START, "upload_node")
    graph.add_edge("upload_node", "parse_node")
    graph.add_edge("parse_node", "chunk_node")
    graph.add_edge("chunk_node", "embed_node")
    graph.add_edge("embed_node", "upsert_node")
    graph.add_edge("upsert_node", END)
    return graph.compile()
