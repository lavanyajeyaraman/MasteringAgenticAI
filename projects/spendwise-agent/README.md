# SpendWise Agent

SpendWise Agent is a Streamlit expense tracker that turns bank statement PDFs and CSVs into clean transactions, spending dashboards, insights, and an AI chat assistant.

## Features

- Upload bank statement PDFs or CSVs.
- Extract, cleanse, and standardize transaction data.
- Review and correct categories before dashboard analysis.
- Save learned merchant category rules in `data/category_rules.csv`.
- View spending dashboard, insights, subscriptions, alerts, and transactions.
- Chat with an AI assistant using Groq, Gemini, OpenAI, or local Ollama models.

## Project Structure

```text
.
├── app.py                    # Streamlit app entrypoint
├── src/                      # Core app package
│   ├── ingestion.py          # PDF/CSV parsing, cleansing, learned rules
│   ├── analytics.py          # Metrics, alerts, subscriptions, insights
│   ├── ai.py                 # Chat and AI helper functions
│   ├── llm.py                # LangChain provider registry
│   ├── sample_data.py        # Demo transactions
│   └── styles.py             # Streamlit styling
├── data/
│   └── category_rules.csv    # Learned merchant/category rules
├── docs/
│   ├── architecture.md       # Demo architecture diagrams
│   └── project_structure.md
├── tests/                    # Unit tests
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd projects/spendwise-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and choose one provider:

```bash
AI_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

For local open-weight models, use Ollama:

```bash
AI_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

Then run:

```bash
streamlit run app.py
```

## Usage Flow

1. Open the app.
2. Go to **Import**.
3. Upload a PDF or CSV statement.
4. Review extracted transactions.
5. Correct categories if needed.
6. Click **Save category rules** so future statements learn from corrections.
7. Click **Use reviewed rows**.
8. Explore Dashboard, Insights, Transactions, and Chat.

## Privacy Notes

- Do not commit `.env`.
- Do not commit real bank PDFs, extracted text files, or standardized CSVs.
- The root `.gitignore` excludes private financial statement files by default.

## Tests

```bash
source .venv/bin/activate
pytest -q
```
