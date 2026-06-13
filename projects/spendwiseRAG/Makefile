.PHONY: install run test lint clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
STREAMLIT := $(VENV)/bin/streamlit
PYTEST := $(PYTHON) -m pytest

$(PYTHON):
	python3 -m venv $(VENV)

install: $(PYTHON)
	$(PIP) install -r requirements.txt

run: $(PYTHON)
	$(STREAMLIT) run app.py

test: $(PYTHON)
	$(PYTEST) -q

lint: $(PYTHON)
	$(PYTHON) -m compileall app.py src tests

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
