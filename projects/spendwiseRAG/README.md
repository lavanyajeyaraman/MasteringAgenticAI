# SpendWise RAG

SpendWise RAG is a standalone personal-finance RAG app for bank and credit-card statements. It parses uploaded statement PDFs, turns transactions and statement text into financial chunks, indexes them locally, and answers questions using only retrieved evidence.

This app is intentionally separate from `projects/spendwise-agent`.

## Features

- Upload PDF statements in Streamlit.
- Detect likely bank/card type, statement year, and namespace.
- Extract table rows and narrative text with `pdfplumber`.
- Build transaction, summary, and rollup chunks with metadata.
- Search chunks with local BM25-style retrieval.
- Upsert text chunks to Pinecone integrated embedding indexes.
- Generate grounded answers with Ollama from retrieved evidence.
- Analyze spending from transaction chunks in the active RAG index.

## UI Pages

- **Upload**: multiple PDF uploader, auto-detected but overridable card type, ingestion progress, and chunk/upsert summaries.
- **Chat**: bottom chat input, conversation history, answer markdown, source chunks, confidence, faithfulness, and rerank status.
- **Analytics Dashboard**: Plotly category, monthly trend, and top merchant charts powered by indexed transaction chunks.

## Setup

```bash
cd projects/spendwiseRAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

The app still supports `streamlit run app.py`, but the production UI entrypoint now lives at:

```text
src/spendwise_rag/ui/streamlit_app.py
```

## Environment

Ollama powers local chat. Pinecone integrated embeddings store the vector index, while BM25 is persisted locally for the planned hybrid retrieval layer.

```bash
AI_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434

PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=smartspend
```

Run a local model with Ollama:

```bash
ollama pull llama3.1:8b
ollama serve
```

## Tests

```bash
pytest -q
```

You can also use the project helpers:

```bash
make run
make test
```

## Project Structure

See `docs/project_structure.md` for the production folder layout.
