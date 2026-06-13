# SpendWise RAG Production Structure

```text
spendwiseRAG/
├── app.py                         # Backward-compatible Streamlit launcher
├── pyproject.toml                 # Test/package configuration
├── requirements.txt               # Runtime and test dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Local/generated artifact exclusions
├── Makefile                       # Common developer commands
├── config/
│   └── settings.example.toml      # Non-secret runtime defaults
├── data/
│   ├── raw/                       # Local statement PDFs; ignored by Git
│   ├── processed/                 # Generated extracts/exports; ignored by Git
│   └── indexes/                   # Local BM25/vector artifacts; ignored by Git
├── deploy/
│   └── docker/
│       ├── Dockerfile             # Container image for Streamlit app
│       └── docker-compose.yml     # Local container orchestration
├── docs/
│   ├── operations/                # Runbooks and deployment notes
│   └── project_structure.md       # This file
├── logs/                          # Local logs; ignored by Git
├── scripts/
│   ├── run_streamlit.sh           # Local app runner
│   └── run_tests.sh               # Local test runner
├── src/
│   └── spendwise_rag/
│       ├── core/
│       │   ├── config.py          # Bank/card config
│       │   ├── models.py          # Dataclasses and shared domain models
│       │   └── state.py           # LangGraph state types
│       ├── graphs/
│       │   └── ingestion_graph.py # LangGraph ingestion workflow
│       ├── processing/
│       │   ├── analytics.py       # Analytics derived from indexed chunks
│       │   ├── chunking.py        # Transaction, summary, rollup chunk builders
│       │   ├── ingestion.py       # PDF extraction and statement parsing nodes
│       │   └── transaction_parser.py
│       ├── retrieval/
│       │   ├── __init__.py        # Local retrieval index and search
│       │   └── bm25_store.py      # BM25 persistence
│       ├── services/
│       │   ├── llm.py             # Ollama/model provider integration
│       │   ├── pipeline.py        # Application orchestration
│       │   └── vector_store.py    # Pinecone records/upsert integration
│       └── ui/
│           └── streamlit_app.py   # Main Streamlit UI
└── tests/                         # Unit tests
```

## Production Notes

- Keep source code inside `src/spendwise_rag/`.
- Keep Streamlit page code inside `src/spendwise_rag/ui/`.
- Keep domain models/config/state in `src/spendwise_rag/core/`.
- Keep parsing, ingestion, chunking, and analytics logic in `src/spendwise_rag/processing/`.
- Keep provider integrations and orchestration in `src/spendwise_rag/services/`.
- Keep local PDFs, extracted files, generated CSVs, indexes, logs, `.env`, and `.venv` out of Git.
- Keep deploy-specific files under `deploy/`.
- Keep repeatable developer operations under `scripts/` and `Makefile`.
