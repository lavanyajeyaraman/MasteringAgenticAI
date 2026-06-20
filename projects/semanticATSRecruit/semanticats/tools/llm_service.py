from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class NebiusLLMService:
    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key
        self.base_url = "https://api.studio.nebius.com/v1/chat/completions"

    def complete_json(self, *, model: str, system: str, user: str) -> dict[str, Any] | None:
        if not self.api_key:
            return None
        payload = {
            "model": model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:  # pragma: no cover - requires external service
            response = httpx.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as exc:
            logger.warning("Nebius completion failed; using deterministic extractor: %s", exc)
            return None
