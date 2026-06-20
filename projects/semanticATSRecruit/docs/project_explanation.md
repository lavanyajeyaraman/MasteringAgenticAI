# semanticATSRecruit Project Explanation

## Project Overview

semanticATSRecruit is an AI recruiting assistant that helps recruiters find strong candidates who may be missed by a traditional keyword-based ATS.

Most ATS systems depend heavily on exact keyword matches. For example, if a job description asks for Playwright, a candidate with strong Selenium experience may be rejected even though the skills are transferable. semanticATSRecruit solves this by combining keyword search, vector search, semantic skill matching, evidence-backed reports, and human review checkpoints.

The goal of the project is not to replace the recruiter. The goal is to help the recruiter make faster and more informed screening decisions while keeping the reasoning transparent.

## Problem Statement

Traditional resume screening has three common problems:

1. It can miss good candidates because their resumes use different wording than the job description.
2. It often gives rankings without explaining the evidence behind them.
3. It can generate recommendations that are hard to trust if there is no verification step.

semanticATSRecruit addresses these problems by looking for both direct matches and semantic matches, showing evidence from the resume, and flagging weak or unsupported claims.

## High-Level Solution

semanticATSRecruit uses a LangGraph-based agentic workflow. The workflow takes a job description and candidate resumes, then moves through these stages:

1. Extract the job requirements.
2. Parse candidate resumes into structured profiles.
3. Chunk and embed resume evidence.
4. Search candidates using both BM25 and vector search.
5. Expand skills through a skill graph to find transferable matches.
6. Rerank candidates.
7. Generate candidate reports.
8. Verify report claims against resume evidence.
9. Show audit signals and recruiter review checkpoints.

## Where The Agent Is In The Code

semanticATSRecruit does not use one single `AgentExecutor` or `Agent(...)` object. The agent is implemented as a LangGraph workflow.

The main workflow is defined in:

```text
projects/semanticats/semanticats/graph/graph.py
```

The `build_graph()` function creates a `StateGraph`, adds each node, connects the nodes in order, and compiles the graph:

```python
graph = StateGraph(RecruitingState)
graph.add_node("ingest_jd", ingest_jd)
graph.add_node("search_candidates", search_candidates)
graph.add_node("generate_reports", generate_reports)
graph.add_node("verify_reports", verify_reports)
return graph.compile(checkpointer=MemorySaver())
```

The graph is called from the Streamlit UI here:

```text
projects/semanticats/semanticats/ui/streamlit_app.py
```

The key call is:

```python
build_graph().invoke(state, config={"configurable": {"thread_id": "streamlit"}})
```

It is also called from the FastAPI endpoint in:

```text
projects/semanticats/main.py
```

The key call is:

```python
build_graph().invoke(state, config={"configurable": {"thread_id": "api"}})
```

## Are The Nodes Agents?

Each node is not a separate agent object. Each node is a step in the workflow.

However, related nodes behave like specialized agent roles:

1. Ingestion Agent
   The ingestion nodes parse the job description and resumes, extract structured data, chunk candidate evidence, and create embeddings.

2. Search Agent
   The search nodes run BM25 keyword search, vector search, reciprocal rank fusion, skill expansion, filtering, and reranking.

3. Reasoning Agent
   The reasoning node creates candidate reports with direct matches, semantic matches, gaps, ramp-up estimates, ATS rejection reasons, and hiring recommendations.

4. Verification Agent
   The verification node checks whether the generated report claims are supported by resume evidence and creates faithfulness flags.

So, in demo language: the full graph is the agentic system, and the groups of nodes represent specialized agent roles.

## Workflow Explanation

### 1. Job Description Ingestion

The workflow starts with `ingest_jd`.

This node receives the raw job description text, extracts structured requirements, and stores them in the shared graph state as `jd_structured`.

The structured job description can include:

- Required skills
- Preferred skills
- Responsibilities
- Seniority
- Implicit requirements

This matters because the rest of the workflow uses the structured job description as the search and reasoning target.

### 2. Resume Ingestion

The `ingest_resumes` node reads candidate resumes and extracts structured candidate profiles.

Each candidate profile can include:

- Candidate name
- Candidate ID
- Skills
- Experience
- Resume evidence

The system supports PDF and text inputs. In demo mode, it can also use built-in sample resumes.

### 3. Candidate Indexing

The `index_candidates` node chunks each candidate profile into searchable resume evidence.

Then it creates embeddings for the chunks. These chunks become the searchable evidence base used by retrieval and reporting.

### 4. Hybrid Search

The `search_candidates` node performs the main candidate search.

It uses multiple retrieval techniques:

- BM25 for keyword-based search
- Vector search for semantic similarity
- Reciprocal Rank Fusion to merge search results
- Skill graph expansion to find related skills
- Cross-encoder reranking to improve final ranking

This is important because the system does not depend only on exact keywords. It can understand that related tools or adjacent experience may still be relevant.

Example:

```text
Job asks for: Playwright
Resume has: Selenium
semanticATSRecruit can identify this as a transferable testing automation match.
```

### 5. Skill Graph And Semantic Matching

The skill graph is used to connect related skills.

For example:

- Selenium can be related to Playwright.
- CrewAI can be related to LangGraph or agentic workflows.
- ECS can be related to Kubernetes or container orchestration.

This is how semanticATSRecruit catches candidates that a keyword-only system may miss.

The system also creates a `taxonomy_audit`, which shows semantic matches that should be visible to the recruiter.

### 6. Candidate Ranking

After retrieval and reranking, semanticATSRecruit aggregates evidence by candidate and produces a ranked candidate list.

Each candidate can have:

- Score
- Matched skills
- Semantic matches
- Resume evidence
- Years of experience

The recruiter can also apply session filters, such as only showing candidates with a minimum number of years of experience.

### 7. Report Generation

The `generate_reports` node creates a candidate report for approved shortlisted candidates.

Each report includes:

- Direct matches
- Semantic matches
- Transferable skills
- Transferability score
- Skill gaps
- Ramp-up estimate
- Possible ATS rejection reason
- Hiring recommendation
- Recruiter summary

The report is designed to explain why a candidate is recommended, not just assign a score.

### 8. Verification

The `verify_reports` node checks whether report claims are supported by resume evidence.

For example, if the report says the candidate has a specific skill, the verifier checks whether that skill appears in the cited evidence.

If a claim is unsupported or low confidence, the system adds it to `faithfulness_flags`.

This gives the recruiter a trust layer. Instead of blindly accepting the generated report, the recruiter can see which claims need review.

## HITL: Human In The Loop

HITL means the human is directly inside the workflow at important decision points.

In semanticATSRecruit, HITL checkpoints are defined in:

```text
projects/semanticats/semanticats/graph/nodes/hitl.py
```

The main HITL checkpoints are:

1. `hitl_jd_review`
   The recruiter reviews the extracted job description requirements.

2. `hitl_shortlist`
   The recruiter reviews the ranked candidates and approves a subset.

3. `hitl_report_gate`
   The recruiter selects which candidate reports should be rendered.

For the Streamlit demo, each HITL node writes a visible checkpoint object into the shared graph state and sets `paused_at` to the current checkpoint:

```python
{
    "checkpoint": checkpoint,
    "message": message,
    "payload": payload,
    "status": "review_required",
}
```

The graph checks `paused_at` after each HITL node. If the workflow is paused, routing stops at that checkpoint instead of continuing. The UI reads checkpoint objects from `state["interrupts"]`, displays them as recruiter review panels, and resumes the graph only after the recruiter clicks the approval button.

Demo explanation:

```text
Human-in-the-loop is used when the system reaches a decision point that should not be fully automated. For example, before trusting the extracted job description or final shortlist, the recruiter can review and approve the data.
```

## HOTL: Human On The Loop

HOTL means the workflow continues automatically, but the human gets monitoring signals and can intervene if needed.

In semanticATSRecruit, HOTL is represented by:

- Taxonomy audit rows
- Faithfulness flags
- Session override filters

The HOTL hooks are:

```text
hotl_taxonomy
hotl_faithfulness
```

These hooks currently pass the state through, while the monitoring data is created by other nodes and shown in the UI.

Examples:

- `taxonomy_audit` shows semantic matches like related or transferable skills.
- `faithfulness_flags` shows claims that may not be supported by evidence.
- Session overrides let the recruiter adjust filtering with natural language commands.

Demo explanation:

```text
Human-on-the-loop is used for oversight. The system keeps running, but it surfaces audit signals such as semantic match explanations and faithfulness warnings so the recruiter can monitor the automation.
```

## MemorySaver Checkpointer

The graph is compiled with:

```python
checkpointer=MemorySaver()
```

This lets LangGraph keep track of graph state by thread ID while the app is running.

The invocation includes a thread ID:

```python
config={"configurable": {"thread_id": "streamlit"}}
```

This is useful for stateful execution, interrupts, and human review checkpoints.

`MemorySaver` stores the checkpoints in memory, so the state is not permanent. If the app restarts, the in-memory checkpoints are lost.

## User Interface

The project has a Streamlit UI.

The user can:

- Enter a job description
- Upload resumes
- Include demo resumes
- Apply session overrides
- Run candidate search
- View ranked candidates
- Review taxonomy audits
- View generated reports
- See faithfulness warnings
- Export candidate reports

The UI calls the LangGraph workflow and then renders the final graph state.

## API

The project also has a FastAPI entry point.

The API can receive a job description and resume files, create the initial recruiting state, invoke the graph, and return the final result as JSON.

This shows that the core agentic workflow is independent of the UI. Streamlit is one interface, and FastAPI is another interface.

## Technology Stack

semanticATSRecruit uses:

- Python for the backend logic
- LangGraph for workflow orchestration
- Streamlit for the demo UI
- FastAPI for API access
- BM25 for keyword retrieval
- Embeddings and vector search for semantic retrieval
- Pinecone-compatible vector client logic
- Reciprocal Rank Fusion for merging rankings
- Cross-encoder reranking
- Skill graph expansion for transferable skills
- Pydantic models for structured data
- Pytest for validation scenarios

## Important Demo Scenarios

The tests include scenarios that show why semantic matching is useful:

1. Selenium Engineer vs Playwright job description
   A keyword ATS might miss the candidate, but semanticATSRecruit can detect transferable test automation experience.

2. CrewAI Engineer vs LangGraph job description
   A keyword ATS may not connect the tools, but semanticATSRecruit can recognize related agentic workflow experience.

3. ECS Engineer vs Kubernetes job description
   A keyword ATS may reject the candidate, but semanticATSRecruit can identify adjacent container orchestration experience.

These scenarios demonstrate the main value of the project: finding strong candidates even when their resume does not use the exact same words as the job description.

## Demo Talk Track

Use this explanation during the demo:

```text
semanticATSRecruit is an evidence-based recruiting assistant. The problem it solves is that traditional ATS systems are usually keyword-heavy, so they can miss strong candidates who describe the same skill in a different way.

In this project, I built the agent as a LangGraph workflow. There is not one single AgentExecutor object. Instead, the full graph is the agentic system. Each node is a workflow step, and groups of nodes behave like specialized agents: ingestion, search, reasoning, and verification.

The workflow starts by extracting structured requirements from the job description. Then it parses resumes, chunks candidate evidence, creates embeddings, and searches candidates using both BM25 keyword search and vector search. The results are merged with Reciprocal Rank Fusion and reranked.

The key feature is semantic matching. For example, if a job asks for Playwright and a resume says Selenium, a keyword ATS might miss that candidate. semanticATSRecruit uses a skill graph to identify that the experience is transferable, then shows the evidence behind the match.

After ranking candidates, the reasoning step generates candidate reports. These reports include direct matches, semantic matches, gaps, ramp-up estimates, ATS rejection reasons, and hiring recommendations.

Then the verification step checks whether the report claims are actually supported by resume evidence. If something is weak or unsupported, it is shown as a faithfulness flag.

The project also includes HITL and HOTL. Human-in-the-loop is used for decision checkpoints like reviewing the job description, approving a shortlist, and selecting reports. Human-on-the-loop is used for monitoring, such as taxonomy audit rows and faithfulness warnings.

So the main idea is not just automation. It is transparent automation. The system helps recruiters find better candidates while still showing the evidence, the semantic reasoning, and the places where a human should review the result.
```

## One-Minute Summary

semanticATSRecruit is a LangGraph-based recruiting assistant that improves candidate screening by combining keyword retrieval, semantic retrieval, skill graph expansion, reranking, evidence-backed reports, HITL checkpoints, and HOTL monitoring.

Its main value is that it can identify qualified candidates who may be rejected by keyword-only ATS systems, while still keeping the recruiter in control through review checkpoints and verification flags.
