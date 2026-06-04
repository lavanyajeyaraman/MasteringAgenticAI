# SpendWise Agent Architecture

## High-Level Flow

```mermaid
flowchart LR
    User[User] --> UI[Streamlit UI]

    UI --> Import[Import & Review]
    Import --> Upload[PDF / CSV Upload]
    Upload --> Extract[PDF Text + Table Extraction]
    Extract --> Cleanse[Cleanse Merchants<br/>Normalize Dates + Amounts]
    Cleanse --> Rules[Category Rule Engine]

    Rules --> BuiltIn[Built-in Rules]
    Rules --> Learned[data/category_rules.csv<br/>Learned User Corrections]
    BuiltIn --> Standardized[Standardized Transactions]
    Learned --> Standardized

    Standardized --> Review[Editable Review Table]
    Review --> SaveRules[Save Category Rules]
    SaveRules --> Learned

    Review --> ActiveData[Session Transaction Data]
    ActiveData --> Dashboard[Dashboard]
    ActiveData --> Insights[Insights]
    ActiveData --> Transactions[Transactions Table]
    ActiveData --> ChatContext[AI Context Builder]

    ChatContext --> LLM[LangChain Provider Registry]
    LLM --> Groq[Groq]
    LLM --> Gemini[Gemini]
    LLM --> OpenAI[OpenAI]
    LLM --> Ollama[Ollama Local Model]

    Groq --> Chat[SpendWise Agent Chat]
    Gemini --> Chat
    OpenAI --> Chat
    Ollama --> Chat
```

## Component View

```mermaid
flowchart TB
    subgraph UI["Streamlit App"]
        ImportTab[Import Tab]
        DashboardTab[Dashboard Tab]
        InsightsTab[Insights Tab]
        TransactionsTab[Transactions Tab]
        FloatingChat[Floating Chat]
    end

    subgraph Ingestion["Ingestion Layer"]
        PDFParser[pdfplumber Parser]
        CSVParser[CSV Normalizer]
        MerchantCleaner[Merchant Cleaner]
        CategoryEngine[Category Engine]
    end

    subgraph Memory["Learning Memory"]
        RuleFile[data/category_rules.csv]
        ReviewLoop[User Review Corrections]
    end

    subgraph Analytics["Analytics Layer"]
        Metrics[Summary Metrics]
        Charts[Category + Daily Charts]
        Alerts[Overspending Alerts]
        Subs[Subscription Detection]
        Patterns[Pattern Observations]
    end

    subgraph AI["AI Orchestration"]
        Context[Financial Context Builder]
        LangChain[LangChain Registry]
        Providers[Groq / Gemini / OpenAI / Ollama]
    end

    ImportTab --> PDFParser
    ImportTab --> CSVParser
    PDFParser --> MerchantCleaner
    CSVParser --> MerchantCleaner
    MerchantCleaner --> CategoryEngine
    RuleFile --> CategoryEngine
    CategoryEngine --> ImportTab
    ImportTab --> ReviewLoop
    ReviewLoop --> RuleFile

    ImportTab --> Metrics
    Metrics --> DashboardTab
    Charts --> DashboardTab
    Alerts --> InsightsTab
    Subs --> InsightsTab
    Patterns --> InsightsTab
    ImportTab --> TransactionsTab

    Metrics --> Context
    Alerts --> Context
    Subs --> Context
    Patterns --> Context
    Context --> LangChain
    LangChain --> Providers
    Providers --> FloatingChat
```

## Demo Talk Track

SpendWise Agent starts with a bank statement PDF or CSV. The import layer extracts transactions, cleans noisy merchant names, standardizes dates and amounts, and applies categorization rules.

The key design decision is the learning loop. When a user corrects categories in the review table, those corrections are saved to `data/category_rules.csv`. Future statements use those learned rules before falling back to built-in rules, so the agent improves over time.

Once transactions are standardized, the same clean dataset powers the dashboard, insights, transaction table, and AI chat context. The chatbot is model-agnostic through LangChain and can run on Groq, Gemini, OpenAI, or a local Ollama model.

## Why This Architecture Works

- Real bank PDFs are inconsistent, so the app separates extraction from review.
- The dashboard only uses standardized transactions.
- User corrections become persistent rules.
- LLM providers are swappable through one registry.
- Local Ollama support avoids cloud API quota limits.
