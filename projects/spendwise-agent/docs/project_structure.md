# GitHub Project Structure

Use this layout when uploading SpendWise Agent to GitHub:

```text
SpendWise-Agent/
├── README.md
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── category_rules.csv
├── docs/
│   ├── architecture.md
│   └── project_structure.md
├── src/
│   ├── __init__.py
│   ├── ai.py
│   ├── analytics.py
│   ├── ingestion.py
│   ├── llm.py
│   ├── sample_data.py
│   └── styles.py
└── tests/
    ├── test_ai.py
    ├── test_analytics.py
    └── test_ingestion.py
```

## Commit These

- Source files under `src/`
- `app.py`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `README.md`
- `docs/`
- `tests/`
- `data/category_rules.csv`
- Optional demo-only files such as `sample_expenses.pdf`

## Do Not Commit These

- `.env`
- `.venv/`
- Real bank statement PDFs
- Extracted statement text
- Generated standardized CSVs from real statements
- `__pycache__/`
- `.pytest_cache/`

## Before Publishing

Run:

```bash
source .venv/bin/activate
pytest -q
git status --short
```

Review `git status` carefully and make sure no private financial or secret files are staged.
