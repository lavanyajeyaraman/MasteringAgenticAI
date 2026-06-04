# Mastering Agentic AI

A portfolio of practical AI agent projects built with Python, LangChain, Streamlit, and multiple LLM providers.

## Featured Projects

| Project | Description | Stack |
|---|---|---|
| [SpendWise Agent](projects/spendwise-agent) | AI expense tracker that turns bank statement PDFs/CSVs into clean transactions, dashboards, learned categorization rules, and chat-based insights. | Streamlit, LangChain, pdfplumber, Groq, Gemini, OpenAI, Ollama |

## Repository Layout

```text
.
├── README.md
├── projects/
│   └── spendwise-agent/
│       ├── README.md
│       ├── app.py
│       ├── requirements.txt
│       ├── .env.example
│       ├── src/
│       ├── tests/
│       ├── docs/
│       └── data/
└── .gitignore
```

## Privacy

This repository is configured to ignore private files such as `.env`, virtual environments, real bank statement PDFs, extracted statement text, and generated standardized CSVs.

Before committing, run:

```bash
git status --short
```
