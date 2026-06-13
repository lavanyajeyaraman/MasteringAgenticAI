from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from langchain_ollama import ChatOllama


def _ollama_client() -> ChatOllama:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    return ChatOllama(model=model, base_url=base_url, temperature=0)


def _grounded_prompt(question: str, evidence: list[str]) -> str:
    context = "\n\n".join(f"[{idx + 1}] {item}" for idx, item in enumerate(evidence))
    return (
        "Answer the finance question strictly from the evidence. "
        "If the evidence is insufficient, say what is missing. Cite evidence numbers inline.\n\n"
        f"Question: {question}\n\nEvidence:\n{context}"
    )


def answer_with_ollama(question: str, evidence: list[str]) -> str:
    try:
        response = _ollama_client().invoke(_grounded_prompt(question, evidence))
    except Exception:
        return ""
    return str(response.content).strip()


def answer_with_configured_model(question: str, evidence: list[str]) -> tuple[str, str]:
    provider = os.getenv("AI_PROVIDER", "ollama").lower().strip()
    answer = answer_with_ollama(question, evidence)
    return answer, provider


def plan_spending_query_with_ollama(question: str, allowed_categories: list[str]) -> dict[str, object] | None:
    if not allowed_categories:
        return None
    prompt = (
        "You are a finance query planner. Convert the user's question into JSON only. "
        "Do not answer the question and do not calculate totals. "
        "Use only these category names exactly as written: "
        f"{json.dumps(allowed_categories)}.\n\n"
        "Schema:\n"
        "{"
        '"intent": "compare_categories|category_total|top_merchants|unknown", '
        '"operation": "highest_spending|total|ranking|unknown", '
        '"categories": ["category name"], '
        '"month": "Month YYYY or empty string"'
        "}\n\n"
        f"Question: {question}"
    )
    try:
        response = _ollama_client().invoke(prompt)
        raw = str(response.content).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        parsed = json.loads(raw[start : end + 1])
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None

    allowed_by_lower = {category.lower(): category for category in allowed_categories}
    categories = []
    for category in parsed.get("categories", []):
        canonical = allowed_by_lower.get(str(category).strip().lower())
        if canonical and canonical not in categories:
            categories.append(canonical)

    intent = str(parsed.get("intent", "unknown")).strip()
    operation = str(parsed.get("operation", "unknown")).strip()
    month = str(parsed.get("month", "")).strip()
    if intent not in {"compare_categories", "category_total", "top_merchants", "unknown"}:
        intent = "unknown"
    if operation not in {"highest_spending", "total", "ranking", "unknown"}:
        operation = "unknown"
    return {
        "intent": intent,
        "operation": operation,
        "categories": categories,
        "month": month,
        "planner": "ollama",
    }


def ollama_status() -> tuple[bool, str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, f"Ollama is not reachable at {base_url}: {exc}"

    models = [item.get("name", "") for item in payload.get("models", []) if isinstance(item, dict)]
    model_names = {name.split(":")[0] for name in models} | set(models)
    if model not in models and model.split(":")[0] not in model_names:
        return False, f"Ollama is running, but model `{model}` is not pulled. Available: {', '.join(models) or 'none'}"
    return True, f"Ollama is ready with `{model}`."


def suggest_categories_with_ollama(merchants: list[str], allowed_categories: list[str]) -> list[dict[str, str]]:
    if not merchants:
        return []
    prompt = (
        "You categorize personal finance merchants. Return only valid JSON as a list. "
        "Each item must have merchant, suggested_category, and reason. "
        "Use only these categories: "
        f"{', '.join(allowed_categories)}.\n\n"
        f"Merchants:\n{json.dumps(merchants, indent=2)}"
    )
    try:
        response = _ollama_client().invoke(prompt)
        raw = str(response.content).strip()
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []
        parsed = json.loads(raw[start : end + 1])
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []

    allowed_by_lower = {category.lower(): category for category in allowed_categories}
    merchants_by_lower = {merchant.lower(): merchant for merchant in merchants}
    suggestions = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        merchant = str(item.get("merchant", "")).strip()
        category = str(item.get("suggested_category", item.get("category", ""))).strip()
        reason = str(item.get("reason", "")).strip()
        canonical_merchant = merchants_by_lower.get(merchant.lower())
        canonical_category = allowed_by_lower.get(category.lower())
        if canonical_merchant and canonical_category:
            suggestions.append(
                {
                    "merchant": canonical_merchant,
                    "suggested_category": canonical_category,
                    "reason": reason,
                }
            )
    return suggestions
