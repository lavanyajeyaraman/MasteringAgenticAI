#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -x .venv/bin/streamlit ]]; then
  .venv/bin/streamlit run app.py
else
  streamlit run app.py
fi
