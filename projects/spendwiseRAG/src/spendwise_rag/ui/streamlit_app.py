from __future__ import annotations

import os

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from spendwise_rag.processing.analytics import monthly_spend_trend, spending_by_category, top_merchants
from spendwise_rag.processing.ingestion import CARD_PATTERNS, detect_card_type
from spendwise_rag.services.pipeline import answer_question, build_local_index, combine_indexes


load_dotenv()

st.set_page_config(page_title="SpendWise RAG", page_icon="$", layout="wide")


def init_state() -> None:
    defaults = {
        "rag_index": None,
        "index_summaries": [],
        "chat_messages": [],
        "last_sources": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def configure_sidebar() -> str:
    with st.sidebar:
        st.title("SpendWise RAG")
        page = st.radio("Page", ["Upload", "Chat", "Analytics Dashboard"], label_visibility="collapsed")
        st.divider()
        st.subheader("Configuration")
        ai_provider = "ollama"
        st.text_input("AI provider", value="ollama", disabled=True)
        ollama_model = st.text_input("Ollama chat model", value=os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
        ollama_base_url = st.text_input("Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        pinecone_index = st.text_input("Pinecone index", value=os.getenv("PINECONE_INDEX", "smartspend"))
        os.environ["AI_PROVIDER"] = ai_provider
        if ollama_model:
            os.environ["OLLAMA_MODEL"] = ollama_model
        if ollama_base_url:
            os.environ["OLLAMA_BASE_URL"] = ollama_base_url
        if pinecone_index:
            os.environ["PINECONE_INDEX"] = pinecone_index
        st.caption(f"Answer model: {ai_provider}")
        st.caption("Embeddings: Pinecone integrated embedding")
        st.caption(f"Pinecone: {'configured' if os.getenv('PINECONE_API_KEY') else 'not configured'}")
        if st.session_state.rag_index:
            st.success(f"Active index: {len(st.session_state.rag_index.chunks)} chunks")
        return page


def card_options() -> list[str]:
    return ["Auto-detect"] + sorted(CARD_PATTERNS) + ["generic_card"]


def preview_detected_card(uploaded_files) -> str:
    if not uploaded_files:
        return "Auto-detect"
    first = uploaded_files[0]
    name = first.name.lower()
    for card_type in sorted(CARD_PATTERNS):
        if card_type.replace("_", " ") in name or card_type in name:
            return card_type
    return "Auto-detect"


def upload_page() -> None:
    st.header("Upload")
    st.caption("Upload statement PDFs once. The resulting index is reused by Chat and Analytics.")

    uploaded_files = st.file_uploader("PDF statements", type=["pdf"], accept_multiple_files=True)
    default_card = preview_detected_card(uploaded_files)
    selected_card = st.selectbox("Card type", card_options(), index=card_options().index(default_card))
    override = None if selected_card == "Auto-detect" else selected_card

    if uploaded_files:
        detected = ", ".join(detect_card_type(file.name, "") for file in uploaded_files)
        st.caption(f"Filename-based detection: {detected}")

    if uploaded_files and st.button("Run ingestion graph", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        summaries = []
        indexes = []

        for file_number, uploaded in enumerate(uploaded_files, start=1):
            file_label = f"{uploaded.name} ({file_number}/{len(uploaded_files)})"
            status.write(f"Parsing - {file_label}")
            progress.progress(10)
            status.write(f"LangGraph ingestion - {file_label}")
            progress.progress(35)
            index, summary = build_local_index(uploaded.getvalue(), uploaded.name, card_type_override=override)

            status.write(f"Preparing Pinecone integrated embedding records - {file_label}")
            progress.progress(60)

            status.write(f"Indexing BM25 and Pinecone integrated embeddings - {file_label}")
            progress.progress(82)
            summaries.append(summary)
            indexes.append(index)

        st.session_state.rag_index = combine_indexes(indexes)
        st.session_state.index_summaries = summaries
        progress.progress(100)
        status.success("Ingestion complete.")

    if st.session_state.index_summaries:
        st.subheader("Upsert Summary")
        total_chunks = sum(item.chunks_indexed for item in st.session_state.index_summaries)
        total_transactions = sum(item.transaction_rows for item in st.session_state.index_summaries)
        total_summaries = sum(item.summary_chunks for item in st.session_state.index_summaries)
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks indexed", f"{total_chunks:,}")
        c2.metric("Transaction rows", f"{total_transactions:,}")
        c3.metric("Summary chunks", f"{total_summaries:,}")

        for summary in st.session_state.index_summaries:
            with st.expander(f"{summary.filename} - {summary.namespace}", expanded=False):
                st.write(f"Chunks indexed: **{summary.chunks_indexed}**")
                st.write(f"Transaction rows: **{summary.transaction_rows}**")
                st.write(f"Summary chunks: **{summary.summary_chunks}**")
                st.write(f"Rollup chunks: **{summary.rollup_chunks}**")
                st.write(f"Embedding/indexing: **{summary.embedding_model}**")
                st.write(f"Pinecone upserts: **{summary.pinecone_upserts}**")
                st.write(f"Pinecone namespace cleared first: **{'Yes' if summary.pinecone_namespace_cleared else 'No'}**")
                for error in summary.errors:
                    st.warning(error)


def chat_page() -> None:
    st.header("Chat")
    if not st.session_state.rag_index:
        st.info("Upload and index statement PDFs before chatting.")
        return

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                c1, c2, c3 = st.columns(3)
                c1.caption(f"Confidence: {message.get('confidence', 0):.2f}")
                c2.caption(f"Faithfulness: {message.get('faithfulness', 0):.2f}")
                c3.caption(f"Rerank used: {'Y' if message.get('rerank_used') else 'N'}")
                st.caption(f"Model provider: {message.get('model_provider', 'unknown')}")
                diagnostics = message.get("retrieval_diagnostics", {})
                if diagnostics.get("vector_error"):
                    st.warning(diagnostics["vector_error"])
                elif diagnostics.get("vector_warning"):
                    st.warning(diagnostics["vector_warning"])
                elif diagnostics:
                    namespaces = ", ".join(diagnostics.get("pinecone_namespaces", [])) or "none"
                    st.caption(f"Pinecone vector search: {'used' if diagnostics.get('pinecone_used') else 'not used'} · Namespaces: {namespaces}")
                with st.expander("Sources"):
                    for source in message.get("sources", []):
                        st.write(source["text"])
                        st.caption(source["metadata"])

    question = st.chat_input("Ask about your indexed statements")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.status("Thinking...", expanded=True) as status:
                st.write("Retrieving relevant statement chunks...")
                st.write("Checking Pinecone vector results and BM25 matches...")
                st.write("Preparing grounded answer...")
                try:
                    result = answer_question(st.session_state.rag_index, question)
                except Exception as exc:
                    status.update(label="Something went wrong", state="error")
                    error_message = f"I hit an error while answering: {exc}"
                    st.error(error_message)
                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "confidence": 0.0,
                            "faithfulness": 0.0,
                            "rerank_used": False,
                            "model_provider": "error",
                            "sources": [],
                            "retrieval_diagnostics": {},
                        }
                    )
                    st.rerun()
                status.update(label="Answer ready", state="complete")
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "confidence": result["confidence"],
                "faithfulness": result["faithfulness"],
                "rerank_used": result["rerank_used"],
                "model_provider": result["model_provider"],
                "sources": result["matches"],
                "retrieval_diagnostics": result.get("retrieval_diagnostics", {}),
            }
        )
        st.rerun()


def analytics_page() -> None:
    st.header("Analytics Dashboard")
    st.caption("Charts are derived from transaction chunks in the active RAG index.")
    if not st.session_state.rag_index:
        st.info("Upload and index statement PDFs before viewing analytics.")
        return

    category_df = spending_by_category(st.session_state.rag_index)
    trend_df = monthly_spend_trend(st.session_state.rag_index)
    merchant_df = top_merchants(st.session_state.rag_index)

    if category_df.empty:
        st.warning("No transaction chunks were found in the active index.")
        return

    st.subheader("Spending by Category")
    st.plotly_chart(
        px.bar(category_df, x="category", y="amount", text_auto=".2s"),
        use_container_width=True,
    )

    st.subheader("Monthly Spend Trend")
    st.plotly_chart(
        px.line(trend_df, x="month", y="amount", markers=True),
        use_container_width=True,
    )

    st.subheader("Top Merchants by Spend")
    st.plotly_chart(
        px.bar(merchant_df.sort_values("amount"), x="amount", y="merchant", orientation="h", text_auto=".2s"),
        use_container_width=True,
    )


def main() -> None:
    init_state()
    page = configure_sidebar()

    if page == "Upload":
        upload_page()
    elif page == "Chat":
        chat_page()
    else:
        analytics_page()


if __name__ == "__main__":
    main()
