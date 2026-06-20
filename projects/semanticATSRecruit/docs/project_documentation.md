# semanticATSRecruit: Evidence-Based Semantic Recruiting Agent

## 1. Project Overview

semanticATSRecruit is an agentic recruiting system that helps recruiters find qualified candidates who may be missed by traditional keyword-based ATS tools. Instead of only checking whether a resume contains the exact words from a job description, semanticATSRecruit looks for direct skills, transferable skills, semantic matches, and evidence from the resume.

The project is built as a LangGraph workflow in Python. The graph coordinates multiple specialized agent roles: an Ingestion Agent, a Search Agent, a Reasoning Agent, and a Verification Agent. The system takes a job description and resumes as input, extracts structured information, searches across candidate evidence, ranks candidates, generates evidence-backed reports, verifies the claims in those reports, and pauses for human review at important checkpoints.

The main goal is not to replace the recruiter. The goal is to make resume screening more explainable, evidence-based, and human-controlled.

## 2. One-Liner

My agent helps recruiters screen candidates in a Streamlit web app, replacing the manual workflow of reading resumes, matching keywords, and writing shortlist notes. It extracts job and resume data, searches candidates, finds transferable skills, generates reports, verifies evidence, and hands off to a human at JD review, shortlist review, and report approval. I will know it works when a recruiter can produce an evidence-backed candidate shortlist and report in under 5 minutes.

## 3. Problem Statement

Traditional ATS systems often rely heavily on exact keyword matching. This creates a problem when a strong candidate describes equivalent or adjacent experience using different wording.

Example:

```text
Job description asks for: Playwright
Resume says: Selenium
```

A keyword-only ATS may reject the candidate because the word "Playwright" is missing. semanticATSRecruit identifies that Selenium is transferable browser automation experience and shows the recruiter the supporting evidence.

The project addresses three problems:

1. Qualified candidates can be missed because of wording differences.
2. Candidate rankings often lack clear resume evidence.
3. AI-generated recommendations can be hard to trust without verification.

semanticATSRecruit solves this by combining hybrid search, semantic skill matching, evidence-backed reporting, verification, and human-in-the-loop checkpoints.

## 4. Build Track

I chose the code-heavy build track using Python and LangGraph.

This track was appropriate because the project needs:

- Stateful multi-step workflow orchestration
- Multiple specialized agent roles
- Human-in-the-loop checkpoints
- Candidate state passed across graph nodes
- Custom ranking and verification logic
- Testable behavior through Pytest

## 5. Application Surface

semanticATSRecruit is available through:

1. Streamlit UI
   Recruiters can enter a job description, upload resumes, run the agentic workflow, review checkpoints, view candidate rankings, inspect reports, and ask chat questions.

2. FastAPI endpoint
   The same graph can be invoked through an API with a job description and resume files.

The core agentic workflow is independent of the UI. Streamlit and FastAPI are only different entry points into the same graph.

## 6. Architecture Summary

The main graph is defined in:

```text
projects/semanticats/semanticats/graph/graph.py
```

The Streamlit UI invokes the graph from:

```text
projects/semanticats/semanticats/ui/streamlit_app.py
```

The FastAPI endpoint invokes the graph from:

```text
projects/semanticats/main.py
```

The high-level flow is:

```text
Job Description + Resumes
        |
        v
Ingestion Agent
        |
        v
Search Agent
        |
        v
Reasoning Agent
        |
        v
Verification Agent
        |
        v
Recruiter Review + Reports
```

## 7. What Each Agent Does

### 7.1 Ingestion Agent

The Ingestion Agent prepares raw inputs for the rest of the workflow.

Nodes:

```text
ingest_jd
hitl_jd_review
ingest_resumes
index_candidates
```

Responsibilities:

- Read the job description.
- Extract structured job requirements.
- Read uploaded resumes.
- Extract structured candidate profiles.
- Chunk resumes into searchable evidence.
- Generate embeddings for resume chunks.

The structured job description includes:

- Required skills
- Preferred skills
- Seniority
- Responsibilities
- Implicit requirements

The structured candidate profile includes:

- Candidate name
- Candidate ID
- Skills
- Years of experience
- Companies
- Projects
- Certifications
- Source resume text

The Ingestion Agent answers:

```text
What does the job need, and what evidence do we have from each resume?
```

### 7.2 Search Agent

The Search Agent finds and ranks candidates.

Nodes:

```text
search_candidates
hotl_taxonomy
hitl_shortlist
```

Responsibilities:

- Expand the job description query using the skill graph.
- Run BM25 keyword retrieval.
- Run vector search over resume chunks.
- Merge search results using Reciprocal Rank Fusion.
- Rerank candidate evidence.
- Aggregate chunk-level results into candidate-level rankings.
- Apply recruiter session overrides.
- Produce a taxonomy audit of semantic matches.

The Search Agent answers:

```text
Which candidates are relevant, even when their resumes use different wording?
```

Example:

```text
Playwright requirement -> Selenium resume evidence
LangGraph requirement -> CrewAI resume evidence
Kubernetes requirement -> ECS resume evidence
```

### 7.3 Reasoning Agent

The Reasoning Agent turns ranked candidates into recruiter-friendly reports.

Node:

```text
generate_reports
```

Responsibilities:

- Compare candidate skills against job requirements.
- Separate direct matches from semantic matches.
- Calculate a transferability score.
- Identify gaps.
- Estimate ramp-up effort.
- Explain why a keyword ATS might reject the candidate.
- Produce a hiring recommendation.
- Write a recruiter summary.

The Reasoning Agent answers:

```text
Why should the recruiter consider or reject this candidate?
```

The report includes:

- Direct matches
- Semantic matches
- Transferable skills
- Transferability score
- Skill gaps
- Ramp-up estimate
- ATS rejection reason
- Hiring recommendation
- Recruiter summary

### 7.4 Verification Agent

The Verification Agent checks whether report claims are supported by resume evidence.

Nodes:

```text
verify_reports
hotl_faithfulness
hitl_report_gate
```

Responsibilities:

- Review direct match claims.
- Review semantic match claims.
- Check whether cited evidence supports each claim.
- Assign confidence scores.
- Create faithfulness flags for weak or unsupported claims.
- Pause before final report rendering.

The Verification Agent answers:

```text
Can we trust the claims in this report?
```

If the report says a candidate has Python experience, the verifier checks whether the cited evidence actually contains Python. If the evidence does not support the claim, the UI shows a faithfulness warning.

## 8. Human-In-The-Loop Design

semanticATSRecruit uses human-in-the-loop checkpoints because hiring-related decisions should not be fully automated.

The workflow pauses at:

1. JD Review
   The recruiter reviews and edits extracted job requirements before search begins.

2. Shortlist Review
   The recruiter reviews ranked candidates before reports are generated.

3. Report Gate
   The recruiter approves final reports before they are rendered.

In interactive Streamlit mode, each checkpoint sets:

```text
paused_at
interrupts
```

The graph stops execution until the recruiter clicks an approval button. Reject buttons are also available, and rejection stops the workflow.

This means the system cannot move from JD review to candidate search, or from candidate ranking to report generation, unless the recruiter approves the current checkpoint.

## 9. Human-On-The-Loop Monitoring

semanticATSRecruit also uses human-on-the-loop monitoring. In this pattern, the system continues automatically, but it surfaces audit signals for the recruiter.

HOTL signals include:

- Taxonomy audit rows
- Faithfulness warnings
- Session overrides

The taxonomy audit shows semantic matches such as:

```text
Selenium -> Playwright
CrewAI -> LangGraph
ECS -> Kubernetes
```

Faithfulness warnings appear when the verification step finds a claim that is unsupported or below the confidence threshold.

Session overrides let the recruiter steer the current search, for example:

```text
Only show candidates with 5+ years
Require Kubernetes
Boost Maya
```

## 10. State Management

The graph uses a shared state object called `RecruitingState`.

Important fields include:

```text
jd_text
jd_structured
jd_approved
candidates_raw
indexed_candidates
ranked_candidates
shortlist_approved
reports
selected_reports
taxonomy_audit
faithfulness_flags
interrupts
paused_at
rejected_at
rejection_reason
conversation_history
```

Each node receives the current state, updates part of it, and passes it to the next node.

The graph is compiled with a MemorySaver checkpointer. This allows LangGraph to track workflow state by thread ID during the app session. Since MemorySaver is in-memory, the state is not permanently stored after app restart.

## 11. Resume Chunking And Vector Storage

Chunking is done per candidate resume.

For each candidate, the system creates:

1. Skill chunks
   One chunk per extracted skill.

2. Summary chunk
   One compact summary of the candidate.

3. Narrative chunks
   Sliding windows over the resume text.

The sliding window uses approximately 90 words with 25 words of overlap. This preserves context while keeping chunks small enough for precise retrieval.

Each chunk stores metadata:

```text
candidate_id
candidate_name
years_experience
skills
chunk_type
source_text
embedding
```

All chunks from all resumes go into the same searchable collection, but every chunk keeps the candidate ID. After retrieval, the system groups matching chunks back into candidate-level rankings.

If Pinecone credentials are configured, chunks are stored in Pinecone. If not, the app uses an in-memory vector fallback so local demos still work.

## 12. LLM Usage

semanticATSRecruit uses an LLM for extraction when a Nebius API key is configured.

The LLM is used to convert unstructured text into structured JSON:

- Job description extraction
- Resume extraction

The final ranking and recommendation workflow is not a black-box LLM decision. It uses:

- BM25 retrieval
- Vector search
- Skill graph matching
- Reciprocal Rank Fusion
- Reranking
- Rule-based recommendation thresholds
- Evidence verification

The chat assistant is currently deterministic and grounded in current session results. It answers questions about ranking, candidate fit, evidence, gaps, and comparisons. It can be extended to use an LLM as a grounded explanation layer, but the safest design is to have the LLM explain only the already-generated evidence-backed outputs.

## 13. Tools And Technologies Used

- Python
- LangGraph
- Streamlit
- FastAPI
- Pydantic
- PyMuPDF
- BM25
- Embeddings
- Pinecone-compatible vector storage
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Skill graph matching
- Pytest
- Nebius LLM API for optional extraction

## 14. Datasets Used

The project uses sample resumes embedded in the Streamlit application for demo and testing:

1. Alex Rivera
   Senior QA Automation Engineer with Selenium, Python, API Testing, Docker, and test automation experience.

2. Maya Chen
   Agentic AI Engineer with CrewAI, LangChain, Python, FastAPI, LlamaIndex, and multi-agent workflow experience.

3. Jordan Patel
   Cloud Platform Engineer with ECS, Docker, Terraform, AWS, and container deployment experience.

The app also supports uploaded PDF and TXT resumes. Text-based PDFs are parsed using PyMuPDF. Scanned/image-only PDFs are rejected with a clear message because OCR is not currently implemented.

The default demo job description asks for:

```text
Senior engineer required with Playwright, LangGraph, Kubernetes, and Python experience for production AI systems.
```

This demo intentionally tests semantic matching:

- Selenium can transfer to Playwright.
- CrewAI can transfer to LangGraph.
- ECS can transfer to Kubernetes.

## 15. Prompts Used During Vibe Coding

During development, I used AI coding assistance to inspect the codebase, explain the architecture, and improve the implementation. Example prompts included:

```text
Where is the agent call in semanticATSRecruit?
```

```text
What is checkpointer=MemorySaver() for?
```

```text
Is each node an agent?
```

```text
Give me an explanation to talk through in the demo.
```

```text
How do HITL and HOTL work?
```

```text
How is chunking done? How is the resume stored in the vector DB?
```

```text
I am not seeing anything interrupting in each step in the UI.
```

```text
How can I edit the JD review section?
```

```text
Why are all reports going to Reject?
```

```text
Can we use an LLM here?
```

These prompts helped clarify both the technical architecture and the demo story. They also led to implementation improvements such as true blocking HITL checkpoints, editable JD review, reject buttons, PDF upload handling, improved chat responses, and better recommendation logic.

## 16. Iterations Tried

### Iteration 1: Basic LangGraph Workflow

The first version used a LangGraph state graph with nodes for ingestion, search, reasoning, verification, and display. This established the basic agentic flow.

### Iteration 2: Semantic Matching

I added a skill graph so the system could identify transferable skills such as Selenium to Playwright, CrewAI to LangGraph, and ECS to Kubernetes.

### Iteration 3: Hybrid Search

I added BM25, vector search, reciprocal rank fusion, and reranking so the search was not dependent on only one retrieval method.

### Iteration 4: HITL Visibility

At first, HITL checkpoints were only displayed as review messages, but the graph kept moving forward. This was not true HITL.

I updated the workflow so Streamlit interactive mode sets `paused_at`, stores checkpoint payloads in `interrupts`, and stops graph execution until the recruiter approves or rejects the checkpoint.

### Iteration 5: Editable JD Review

I added editable fields for required skills, preferred skills, seniority, responsibilities, and implicit requirements. The edited JD is passed back into the graph before search continues.

### Iteration 6: Reject Path

The first UI only had approval buttons. I added reject buttons so the recruiter can stop the workflow at JD review, shortlist review, or report gate.

### Iteration 7: PDF Upload Fix

PDF uploads were improved by consolidating the parsing logic into a shared parser that handles uploaded PDF bytes. The parser also gives a clear error for scanned PDFs without selectable text.

### Iteration 8: Recommendation Tuning

The first recommendation logic was too strict for multi-skill JDs. Candidates with meaningful transferable evidence were sometimes marked Reject because they had several gaps. I changed the logic so partial but meaningful matches become Consider, while Reject is reserved for candidates with little or no meaningful support.

### Iteration 9: Chat Improvements

The first chat assistant only responded to narrow keywords. I improved it so it can answer natural questions such as:

```text
Is Alex right fit for QA role?
Who is ranked higher?
```

## 17. Error Handling

The project handles several failure cases:

1. Missing job description
   The graph adds an error when the job description is empty.

2. Missing LLM API key
   The system falls back to deterministic extraction.

3. Missing Pinecone API key
   The system falls back to in-memory vector storage.

4. PDF parsing failure
   The UI shows an error if the uploaded PDF cannot be parsed.

5. Scanned PDF with no selectable text
   The parser raises a clear message asking for a text-based PDF or TXT file.

6. Unsupported report claims
   The Verification Agent creates faithfulness flags for recruiter review.

7. Human rejection
   The workflow records where the rejection happened and stops.

## 18. Evaluation And Tests

The project includes Pytest tests for:

- Skill graph transferability
- Hybrid ranking fusion
- Semantic matching scenarios
- Verification flags
- PDF parsing
- Chat answer behavior
- Multi-skill demo recommendation behavior

Key semantic scenarios:

1. Selenium Engineer vs Playwright JD
   The system should identify Selenium as transferable browser automation experience.

2. CrewAI Engineer vs LangGraph JD
   The system should identify CrewAI as related agentic workflow experience.

3. ECS Engineer vs Kubernetes JD
   The system should identify ECS as transferable container orchestration experience.

The current test suite passes locally.

## 19. Demo Walkthrough

1. Open the Streamlit app.
2. Enter or use the default job description.
3. Upload resumes or include demo resumes.
4. Click Search Candidates.
5. The workflow pauses at JD Review.
6. Edit extracted job requirements if needed.
7. Approve JD and continue.
8. Review the ranked candidate shortlist.
9. Approve the shortlist or reject it.
10. Review faithfulness warnings if any appear.
11. Approve final reports.
12. Open the Reports tab to inspect candidate reports.
13. Use the Chat tab to ask questions such as:

```text
Who is ranked higher?
Is Alex right fit for QA role?
What gaps does Maya have?
```

## 20. What The Agent Should Never Do

semanticATSRecruit should never:

- Make a final hiring decision without human review.
- Hide unsupported claims from the recruiter.
- Treat semantic matches as guaranteed skill equivalence without showing evidence.
- Process scanned PDFs silently without text extraction.
- Modify or delete source resumes.
- Send reports externally without approval.
- Use an LLM as an ungrounded black-box decision maker.

## 21. Success Criteria

The project is successful if:

1. A recruiter can run a full candidate screening workflow in under 5 minutes.
2. The system finds transferable candidates missed by keyword-only matching.
3. Each recommendation includes evidence.
4. HITL checkpoints actually pause the workflow.
5. Unsupported claims are flagged.
6. The recruiter can understand why each candidate was ranked.

## 22. Learnings And Observations

The biggest learning was that agentic systems are mostly about state and control flow, not only prompts. The hard part was deciding when the system should move autonomously and when it should stop for human review.

Another learning was that HITL must actually block the workflow. A review panel is not enough if the graph continues running in the background. Real HITL requires the graph to pause and wait for approval.

I also learned that semantic matching needs careful recommendation logic. If a job description contains many required skills, a specialized candidate may have strong transferable evidence for one area but still score low overall. The recommendation logic must separate "not a complete fit" from "no meaningful evidence."

Finally, verification is important because AI-generated reports can sound confident. The faithfulness step makes the system more trustworthy by checking whether claims are supported by resume evidence.

## 23. Future Improvements

Future improvements could include:

- OCR support for scanned resumes.
- LLM-powered grounded chat over current session results.
- More advanced skill ontology.
- Persistent vector database per recruiting session.
- Recruiter-selectable shortlist instead of approving only top 5.
- Better report editing before export.
- LangSmith tracing and evaluation.
- Authentication and candidate privacy controls.
- More realistic resume datasets.

## 24. Final Summary

semanticATSRecruit is a LangGraph-based agentic recruiting system that combines semantic search, skill graph reasoning, evidence-backed reporting, verification, and human review. It demonstrates the key pieces of an agentic AI system: tool use, stateful workflow, multi-step control flow, error handling, human-in-the-loop approval, human-on-the-loop monitoring, and grounded outputs.

The main value of the project is that it helps recruiters find candidates who may be missed by keyword-only ATS systems while keeping the recruiter in control of the final decision.

