# semanticATSRecruit

semanticATSRecruit is an evidence-based semantic recruiting app for finding high-potential candidates missed by keyword-only ATS systems.

It combines BM25 retrieval, vector search, Reciprocal Rank Fusion, skill graph expansion, cross-encoder reranking, LangGraph orchestration, HITL checkpoints, HOTL monitoring, FastAPI, and Streamlit.

## Quickstart

```bash
cd projects/semanticats
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
streamlit run app.py
```

For the API:

```bash
uvicorn main:app --reload
```

Copy `.env.example` to `.env` to configure Nebius and Pinecone. Without API keys, semanticATSRecruit uses deterministic local fallbacks so the demo and tests still run. Set `SEMANTICATS_ENABLE_LOCAL_MODELS=true` when you want to download and run the sentence-transformer embedder and cross-encoder reranker locally.

## Required Scenarios

The included tests demonstrate:

- Selenium Engineer vs Playwright JD
- CrewAI Engineer vs LangGraph JD
- ECS Engineer vs Kubernetes JD

Each scenario must produce semantic matches with evidence, taxonomy audit records, and faithfulness verification.

## Documentation

- [Architecture](docs/architecture.md)
- [Project Explanation](docs/project_explanation.md)
- [Week 3 Project Documentation](docs/week3_project_documentation.md)
