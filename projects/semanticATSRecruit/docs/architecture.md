# semanticATSRecruit Architecture

semanticATSRecruit uses a LangGraph orchestrator to coordinate four agents:

1. Ingestion Agent parses resumes and job descriptions, chunks candidate evidence, embeds chunks, and stores metadata.
2. Search Agent runs hybrid retrieval with BM25 and vector search, merges rankings with Reciprocal Rank Fusion, applies skill expansion, and reranks candidates.
3. Reasoning Agent creates evidence-backed candidate reports with direct matches, semantic matches, transferable skills, gaps, ramp-up estimates, ATS rejection reasons, and hiring recommendations.
4. Verification Agent independently checks whether each report claim is supported by cited resume evidence and flags low-confidence claims.

Human-in-the-loop checkpoints are represented at JD review, shortlist review, and report gate. In interactive Streamlit mode, checkpoint payloads are stored in state under `interrupts`, `paused_at` marks the active checkpoint, and graph routing stops execution until the recruiter approves the step. Automated tests and API-style runs can skip the pause by leaving `interactive_hitl` disabled.

Human-on-the-loop monitors include taxonomy audit rows for every semantic match, faithfulness flags, and session override filters parsed from recruiter commands.
