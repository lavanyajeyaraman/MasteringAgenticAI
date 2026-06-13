# SpendWise RAG Architecture

SpendWise RAG is a local-first Streamlit application for asking grounded questions over personal bank and credit-card statements. It ingests uploaded PDF statements, parses transaction and summary text, creates searchable chunks, stores local and vector indexes, and answers questions only from retrieved statement evidence.

## System Context

```mermaid
flowchart LR
    User[User] -->|Uploads PDFs<br/>asks questions| UI[Streamlit App]

    UI -->|PDF bytes| Ingestion[LangGraph<br/>Ingestion Graph]
    UI -->|question + active index| Chat[Answer Pipeline]
    UI -->|active transaction chunks| Analytics[Analytics Dashboard]

    Ingestion --> LocalFiles[(data/indexes<br/>LocalIndex + BM25)]
    Ingestion --> Pinecone[(Pinecone<br/>Integrated Embedding Index)]

    Chat --> LocalFiles
    Chat --> Pinecone
    Chat --> Ollama[Ollama<br/>Local Chat Model]

    Analytics --> LocalFiles
```

## End-to-End Runtime Flow

```mermaid
flowchart TD
    A[Upload statement PDFs] --> B[Detect card type,<br/>month, year, namespace]
    B --> C[Extract tables and text<br/>with pdfplumber]
    C --> D[Build chunks]
    D --> D1[Transaction chunks]
    D --> D2[Rollup chunks]
    D --> D3[Summary chunks]
    D --> D4[Image extraction chunks]

    D1 --> E[Build active LocalIndex]
    D2 --> E
    D3 --> E
    D4 --> E

    E --> F[(Persist local pickle index)]
    E --> G[(Persist BM25 index)]
    D --> H[Create Pinecone records]
    H --> I[(Upsert to Pinecone namespace)]

    E --> J[Chat and Analytics]
    G --> J
    I --> J
```

## Ingestion Graph

Implemented in `src/spendwise_rag/graphs/ingestion_graph.py`.

```mermaid
flowchart LR
    START((START)) --> Upload[upload_node]
    Upload --> Parse[parse_node]
    Parse --> Chunk[chunk_node]
    Chunk --> Embed[embed_node]
    Embed --> Upsert[upsert_node]
    Upsert --> END((END))

    Upload -.-> UploadOut[card_type<br/>statement_month<br/>statement_year<br/>namespace<br/>bank_config]
    Parse -.-> ParseOut[ParsedStatement<br/>raw_tables<br/>raw_text]
    Chunk -.-> ChunkOut[Chunk list<br/>LangChain Documents]
    Embed -.-> EmbedOut[Pinecone records<br/>integrated embedding marker]
    Upsert -.-> UpsertOut[LocalIndex<br/>BM25 path<br/>Pinecone upsert count<br/>summary/errors]
```

## Retrieval And Answering Flow

Implemented across `services/pipeline.py`, `graphs/retrieval_graph.py`, `services/aggregate.py`, and `services/llm.py`.

```mermaid
flowchart TD
    Q[User question] --> P[answer_question]

    P --> A{Can direct deterministic<br/>analytics answer it?}
    A -->|Yes| DA[Compute totals,<br/>top merchants, comparisons]
    DA --> R[Return answer + sources]

    A -->|No| LP{Likely spending<br/>comparison?}
    LP -->|Yes| PLAN[Ollama query planner<br/>JSON intent, operation,<br/>categories, month]
    PLAN --> PA{Can planned<br/>analytics answer it?}
    PA -->|Yes| PDA[Compute exact category totals<br/>from transaction chunks]
    PDA --> R
    PA -->|No| RG[LangGraph Retrieval Graph]
    LP -->|No| RG

    RG --> QN[query_node<br/>expand financial query<br/>infer intent + filters]
    QN --> HS[hybrid_search_node]

    HS --> BM25[Local BM25 search]
    HS --> VEC[Pinecone vector search]
    BM25 --> RRF[RRF merge]
    VEC --> RRF

    RRF --> C{confidence >= 0.70?}
    C -->|Yes| GEN[generate_node]
    C -->|No| RR[rerank_node<br/>cross-encoder if available<br/>lexical fallback]

    GEN --> CV[context_validation_node]
    RR --> CV
    CV --> FC[Final evidence context]

    FC --> M{Direct retrieved<br/>transaction math?}
    M -->|Yes| RT[Sum retrieved matching transactions]
    M -->|No| LLM[Ollama grounded answer]

    RT --> R
    LLM --> R
```

## Component Architecture

```mermaid
flowchart TB
    subgraph UI["UI Layer"]
        App[app.py<br/>compatibility launcher]
        Streamlit[src/spendwise_rag/ui/streamlit_app.py]
        UploadPage[Upload Page]
        ChatPage[Chat Page]
        AnalyticsPage[Analytics Dashboard]
    end

    subgraph Core["Core Domain"]
        Models[core/models.py<br/>BankConfig, ParsedStatement,<br/>Chunk, LocalIndex]
        Config[core/config.py<br/>bank/card parsing settings]
        State[core/state.py<br/>LangGraph state types]
    end

    subgraph Processing["Statement Processing"]
        Ingestion[processing/ingestion.py<br/>card detection + PDF extraction]
        Parser[processing/transaction_parser.py<br/>dates, amounts, merchants,<br/>categories]
        Chunking[processing/chunking.py<br/>transaction, rollup, summary chunks]
        AnalyticsProc[processing/analytics.py<br/>charts from transaction chunks]
    end

    subgraph Graphs["LangGraph Workflows"]
        IngestionGraph[graphs/ingestion_graph.py]
        RetrievalGraph[graphs/retrieval_graph.py]
    end

    subgraph Retrieval["Retrieval"]
        LocalRetrieval[retrieval/__init__.py<br/>LocalIndex search]
        BM25[retrieval/bm25_store.py<br/>BM25 persistence + search]
        VectorStore[services/vector_store.py<br/>Pinecone upsert/search]
    end

    subgraph Services["Service Orchestration"]
        Pipeline[services/pipeline.py<br/>build index, combine index,<br/>answer question]
        Aggregate[services/aggregate.py<br/>deterministic finance answers]
        LLM[services/llm.py<br/>Ollama grounded generation]
    end

    Streamlit --> UploadPage
    Streamlit --> ChatPage
    Streamlit --> AnalyticsPage
    App --> Streamlit

    UploadPage --> Pipeline
    ChatPage --> Pipeline
    AnalyticsPage --> AnalyticsProc

    Pipeline --> IngestionGraph
    Pipeline --> RetrievalGraph
    Pipeline --> Aggregate
    Pipeline --> LLM

    IngestionGraph --> Ingestion
    IngestionGraph --> Chunking
    IngestionGraph --> LocalRetrieval
    IngestionGraph --> BM25
    IngestionGraph --> VectorStore

    RetrievalGraph --> LocalRetrieval
    RetrievalGraph --> BM25
    RetrievalGraph --> VectorStore

    Ingestion --> Config
    Ingestion --> Models
    Chunking --> Parser
    Chunking --> Models
    AnalyticsProc --> Models
    LocalRetrieval --> Models
    Pipeline --> Models
```

## Data Stores And External Services

| Store or service | Purpose | Used by |
| --- | --- | --- |
| Streamlit session state | Holds the active in-memory `LocalIndex`, ingestion summaries, chat history, and last sources for the current app session. | `ui/streamlit_app.py` |
| `data/indexes/{namespace}.pkl` | Serialized local index containing chunks and tokenized chunk text. | Ingestion graph, pipeline |
| `data/indexes/{namespace}_bm25.pkl` | Persisted BM25 payload for lexical retrieval. | Ingestion graph, retrieval graph |
| Pinecone index | Stores text records with integrated embeddings in a card/year namespace. | `services/vector_store.py` |
| Ollama | Generates grounded natural-language answers from retrieved evidence. | `services/llm.py` |
| `.env` / environment variables | Configures Ollama and Pinecone. | UI, vector store, LLM service |

## Important Runtime Namespaces

```mermaid
flowchart LR
    Filename[PDF filename] --> Detect[detect_card_type]
    FirstPage[First page text] --> Detect
    FirstPage --> Year[infer_statement_year]
    FirstPage --> Month[infer_statement_month]
    Detect --> Namespace[build_namespace]
    Year --> Namespace
    Namespace --> Example[chase_sapphire_2026]
```

Namespaces are built from the detected card type and statement year. They are used for local index filenames, BM25 filenames, chunk metadata, and Pinecone namespaces.

## Deployment View

```mermaid
flowchart TB
    Dev[Developer Machine] --> Make[Makefile<br/>make run / make test]
    Dev --> StreamlitRun[streamlit run app.py]
    Dev --> Docker[deploy/docker<br/>Dockerfile + compose]

    StreamlitRun --> App[SpendWise RAG App]
    Docker --> App

    App --> LocalDisk[(Local disk<br/>data/, logs/)]
    App --> Ollama[Ollama service<br/>http://localhost:11434]
    App --> PineconeCloud[(Pinecone Cloud)]

    Env[.env] --> App
```

## Request Paths

| User action | Primary path | Output |
| --- | --- | --- |
| Upload PDFs | Streamlit Upload page -> `build_local_index` -> ingestion graph -> local/BM25/Pinecone indexes | Active combined session index and upsert summary |
| Ask aggregate question | Chat page -> `answer_question` -> deterministic aggregate service | Exact totals, top merchants, or category comparisons with source transactions |
| Ask ambiguous comparison | Chat page -> Ollama query planner -> deterministic aggregate service | LLM maps natural language to categories; code computes exact winner |
| Ask semantic question | Chat page -> retrieval graph -> BM25 + Pinecone -> rerank/validate -> Ollama | Grounded answer with source chunks, confidence, faithfulness, and diagnostics |
| View dashboard | Analytics page -> transaction chunks -> pandas/Plotly analytics | Category, monthly trend, and merchant charts |

## Reranker Applied Scenario

The reranker is applied when the first hybrid retrieval pass finds possible matches, but the confidence score is below `0.70`. In that case, the retrieval graph routes from `hybrid_search_node` to `rerank_node` before validating the final context.

```mermaid
sequenceDiagram
    participant User
    participant Chat as Chat Page
    participant Pipeline as answer_question
    participant Retrieval as Retrieval Graph
    participant BM25 as Local BM25
    participant Pinecone as Pinecone Vector Search
    participant Reranker as rerank_node
    participant Ollama as Ollama

    User->>Chat: "Which travel charges look like airfare?"
    Chat->>Pipeline: answer_question(active_index, question)
    Pipeline->>Retrieval: Invoke retrieval graph
    Retrieval->>Retrieval: Expand query with airfare aliases
    Retrieval->>BM25: Search local transaction and summary chunks
    Retrieval->>Pinecone: Search vector records by namespace
    BM25-->>Retrieval: Mixed matches: travel, rideshare, parking, airlines
    Pinecone-->>Retrieval: Mixed semantic matches
    Retrieval->>Retrieval: Merge with reciprocal rank fusion
    Retrieval->>Retrieval: Confidence score = 0.64
    Retrieval->>Reranker: Route to rerank_node
    Reranker->>Reranker: Score top candidates against the original query
    Reranker-->>Retrieval: Top 5 reordered around airfare evidence
    Retrieval->>Ollama: Send validated evidence only
    Ollama-->>Chat: Grounded answer with sources
```

Example user question:

```text
Which travel charges look like airfare?
```

Why this can trigger reranking:

- The query is semantic rather than exact arithmetic, so deterministic analytics does not answer it first.
- `airfare` expands to aliases such as `Frontier Airlines`, `Delta Air Lines`, `American Airlines`, `Spirit Airlines`, `flight`, and `airline`.
- BM25 and Pinecone may return a mixed set of travel-related chunks, including rideshare, parking, hotel, and airline transactions.
- If the top merged result does not overlap strongly enough with the query terms, `_confidence_from_results` can score below `0.70`.
- The graph then applies `rerank_node`, using `cross-encoder/ms-marco-MiniLM-L-6-v2` when available and lexical reranking as a fallback.

Expected UI diagnostics after this path:

| Field | Expected value |
| --- | --- |
| `rerank_used` | `Y` / `true` |
| `confidence` | Often below `0.70` before rerank routing |
| `model_provider` | Usually `ollama`, unless retrieved transaction math handles the answer |
| Sources | Reordered to prioritize the chunks most relevant to airfare |

## Leadership Demo Talk Track

### Slide 1: What SpendWiseRAG Does

SpendWiseRAG turns bank and credit-card statements into a grounded financial assistant.

Core message:

> This is not just a chatbot over PDFs. It is a controlled financial reasoning pipeline that uses exact code for math, search for evidence, and an LLM only where language understanding is useful.

The system supports:

- Statement upload and parsing.
- Transaction-level search.
- Category totals and comparisons.
- Merchant ranking.
- Semantic questions such as `airfare`, `food delivery`, or `streaming`.
- Source citations and retrieval diagnostics for every answer.

### Slide 2: Main Libraries And Their Roles

| Library or service | Role in the system | Demo explanation |
| --- | --- | --- |
| Streamlit | Web UI for upload, chat, diagnostics, and sources. | Gives users a simple app experience without building a custom frontend. |
| LangGraph | Orchestrates ingestion and retrieval as explicit node-based workflows. | Makes the pipeline observable, testable, and easy to explain. |
| pdfplumber | Extracts statement text and tables from PDFs. | Converts raw PDF statements into machine-readable transaction data. |
| pandas | Performs deterministic financial analytics. | Exact totals, comparisons, and rankings are computed in code, not guessed by an LLM. |
| rank-bm25 | Local keyword search over chunks. | Finds exact merchant names, dates, amounts, and category words. |
| Pinecone | Vector database for semantic retrieval. | Finds meaning-based matches, such as `airfare` matching airline charges. |
| Ollama / LangChain Ollama | Local LLM for query planning and grounded answer generation. | Used for language interpretation, not unchecked financial math. |
| sentence-transformers CrossEncoder | Optional reranker for low-confidence retrieval. | Reorders candidate evidence when first-pass retrieval is uncertain. |

### Slide 3: Why The Pipeline Starts With Deterministic Analytics

The first check asks:

```text
Can code answer this exactly?
```

Examples:

```text
How much did I spend on dining?
What are my top 5 merchants by total spend?
Compare groceries vs dining.
```

These questions are answered from parsed transaction chunks using deterministic analytics.

Leadership message:

> For financial math, exact computation is safer than generation. The LLM does not calculate totals when the transaction table can do it exactly.

Typical diagnostics:

```text
Confidence: 0.99
Faithfulness: 1.00
Rerank used: N
Model provider: deterministic_analytics
```

### Slide 4: How The LLM Planner Is Used Safely

Some questions are phrased naturally but still map to structured analytics.

Example:

```text
Which transactions look like transportation expenses?
```

The planner may map this to categories such as:

```text
Ride Share vs Auto
```

Then code computes the exact category totals.

Leadership message:

> The LLM is used as a planner. It converts messy language into structured intent, then deterministic code performs the calculation.

This is why some natural questions still show:

```text
Rerank used: N
Model provider: llm_planned_deterministic_analytics
```

### Slide 5: Retrieval For Semantic Questions

If deterministic analytics cannot answer, the question enters the LangGraph retrieval pipeline.

The retrieval graph performs:

1. Query expansion.
2. BM25 keyword search.
3. Pinecone vector search.
4. Reciprocal rank fusion.
5. Confidence-based reranking.
6. Context validation.

Example:

```text
Show me all airfare charges and the total amount.
```

`airfare` may not literally appear in the statement. The query node expands it to known airline terms and merchants:

```text
Frontier Airlines, Delta Air Lines, American Airlines, Spirit Airlines, flight, airline
```

Leadership message:

> BM25 gives precision. Pinecone gives semantic recall. RRF combines them so the final evidence is stronger than either search path alone.

### Slide 6: What RRF Does

RRF means Reciprocal Rank Fusion.

It combines the BM25 result list and the Pinecone result list.

Simple explanation:

> A chunk is trusted more when it ranks highly in either search system, and especially when both systems agree on it.

Why it matters:

- BM25 is strong for exact strings like `Publix`, `$52.75`, or `2026-04-10`.
- Pinecone is strong for meaning-based terms like `airfare`, `streaming`, or `food delivery`.
- RRF gives a balanced ranking without needing either search method to be perfect.

### Slide 7: When Reranking Happens

The reranker is not the default path.

The graph checks:

```text
confidence >= 0.70?
```

If yes:

```text
Rerank used: N
```

If no:

```text
Rerank used: Y
```

Leadership message:

> Reranking is a low-confidence quality-control step. We use it when the first-pass retrieval is uncertain, which helps control latency and cost.

Good reranker demo question:

```text
Find ambiguous merchant names and explain what they might be.
```

### Slide 8: Context Validation Before The LLM

Before the LLM sees anything, the pipeline validates the retrieved context.

It checks:

- Did we retrieve any chunks?
- Do chunks overlap with the original or expanded query?
- Are chunks sourceable with citation numbers?
- Are off-topic chunks blocked?

Leadership message:

> The LLM only receives validated evidence. If the evidence is weak or unrelated, the system returns a no-data response instead of inventing an answer.

### Slide 9: Deterministic Math After Semantic Retrieval

Semantic retrieval finds the relevant transaction rows. Code still performs the arithmetic.

Example:

```text
Show me all airfare charges and the total amount.
```

Expected answer:

```text
Matching charges:
- 2026-04-25 | Frontier Airlines | $68.98 | Travel
- 2026-04-26 | Frontier Airlines | $143.98 | Travel
- 2026-05-08 | Frontier Airlines | $11.20 | Travel
- 2026-05-11 | Frontier Airlines | $134.98 | Travel

Total: $359.14 across 4 transaction(s).
```

Leadership message:

> The vector database helps find the right evidence, but the final total is calculated deterministically from the retrieved transactions.

### Slide 10: What The Diagnostics Mean

| Diagnostic | Meaning | How to explain it |
| --- | --- | --- |
| Confidence | Retrieval or deterministic answer confidence. | Higher means the system found stronger evidence or used exact analytics. |
| Faithfulness | Whether the answer is grounded in available evidence. | `1.00` means the answer is directly supported by transaction data or retrieved chunks. |
| Rerank used | Whether the low-confidence reranker path was used. | `Y` means retrieval needed extra evidence ordering; `N` means the faster path was sufficient. |
| Model provider | Which path produced the answer. | Shows whether the answer came from deterministic analytics, planned analytics, retrieval math, or Ollama. |
| Pinecone vector search | Whether semantic vector retrieval participated. | Confirms if meaning-based search was used and which namespace was queried. |

### Slide 11: Recommended Demo Script

Question 1:

```text
How much did I spend on dining?
```

Say:

> This is answered by deterministic analytics. We avoid LLM math and compute the exact total from parsed transaction chunks.

Question 2:

```text
Show me all airfare charges and the total amount.
```

Say:

> `Airfare` is semantic. Pinecone helps map that concept to airline transactions, then deterministic code calculates the total.

Question 3:

```text
Find ambiguous merchant names and explain what they might be.
```

Say:

> This is less exact, so the system may use reranking before giving the grounded answer. This shows the quality-control path.

Question 4:

```text
What are my top 5 merchants by total spend?
```

Say:

> This is a structured analytics question. The system ranks merchants directly from transactions, which is faster and more reliable than asking an LLM to infer it.

### Slide 12: Executive Summary

SpendWiseRAG uses the right tool for each part of the problem:

- Exact code for financial math.
- BM25 for precise keyword matching.
- Pinecone for semantic retrieval.
- RRF to combine evidence.
- Reranking when confidence is low.
- Context validation before generation.
- Ollama only for grounded language responses or safe query planning.

Final leadership message:

> The architecture is designed for trust. Every answer is either computed exactly from transaction data or generated from validated source evidence.

## Design Notes

- The UI keeps uploaded statement data in the current Streamlit session and combines multiple statement indexes into one active `LocalIndex`.
- Ambiguous spending comparisons can use Ollama as a planner. The planner returns structured intent and category names, but does not answer or calculate totals.
- Deterministic analytics runs before LLM retrieval for totals, comparisons, and top merchant queries so arithmetic stays exact.
- Retrieval is hybrid: local BM25 provides lexical matching, Pinecone provides vector matching, and reciprocal rank fusion merges both result sets.
- If retrieval confidence is low, the graph reranks candidates with a cross-encoder when available, falling back to lexical reranking.
- Ollama only receives selected evidence strings, and the prompt tells the model to answer strictly from that evidence.
- Pinecone is mandatory for vector search. If Pinecone is unavailable or returns no matches, the UI surfaces a visible diagnostic and retrieval continues in degraded BM25-only mode.
