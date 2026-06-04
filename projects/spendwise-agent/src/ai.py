from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.ingestion import normalize_transactions
from src.llm import build_chat_model, has_api_key, provider_label


def ask_openai(question: str, context: str, history: list[dict[str, str]] | None = None) -> str:
    if not has_api_key():
        return (
            f"Add {provider_label()} API settings to your environment or .env file to enable live AI answers. "
            "I can still show deterministic dashboard insights from your uploaded data."
        )
    messages = [
        SystemMessage(
            content=(
                "You are SpendWise Agent, a concise expense tracker assistant. "
                "Use the provided transaction context only. Give practical, specific answers. "
                "Avoid regulated financial advice and avoid inventing data."
            )
        ),
        HumanMessage(content=f"Current app context:\n{context}"),
    ]
    if history:
        for item in history[-8:]:
            if item.get("role") == "assistant":
                messages.append(AIMessage(content=item.get("content", "")))
            else:
                messages.append(HumanMessage(content=item.get("content", "")))
    messages.append(HumanMessage(content=question))
    try:
        response = build_chat_model().invoke(messages)
        return str(response.content) if response.content else "I could not produce an answer."
    except Exception as exc:  # pragma: no cover - depends on external API/network
        return f"{provider_label()} request failed: {exc}"


def generate_tips(alerts: list[dict[str, Any]], context: str) -> dict[str, str]:
    deterministic = {str(alert["category"]): str(alert["tip"]) for alert in alerts}
    if not alerts or not has_api_key():
        return deterministic
    prompt = (
        "Create one concise savings tip per overspending category. "
        "Return JSON only as {\"Category\": \"tip\"}.\n"
        f"Alerts: {alerts}\nContext: {context}"
    )
    answer = ask_openai(prompt, context, [])
    parsed = _extract_json(answer)
    if isinstance(parsed, dict):
        deterministic.update({str(key): str(value) for key, value in parsed.items()})
    return deterministic


def extract_transactions_from_pdf_text(text: str) -> pd.DataFrame:
    if not text.strip() or not has_api_key():
        return pd.DataFrame(columns=["date", "merchant", "category", "amount", "source", "category_source"])
    prompt = (
        "Extract expense transactions from this bank or card statement text. "
        "Return JSON only as an array of objects with date, merchant, category, amount. "
        "Use positive numeric amounts for spending. If unsure, omit the row.\n\n"
        f"{text[:12000]}"
    )
    answer = ask_openai(prompt, "PDF statement extraction task.", [])
    parsed = _extract_json(answer)
    if not isinstance(parsed, list):
        return pd.DataFrame(columns=["date", "merchant", "category", "amount", "source", "category_source"])
    result = normalize_transactions(pd.DataFrame(parsed), source="pdf-ai")
    return result.transactions


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    start = min([idx for idx in [text.find("{"), text.find("[")] if idx >= 0], default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
