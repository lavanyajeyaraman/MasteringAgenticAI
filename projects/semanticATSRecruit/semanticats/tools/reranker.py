from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RerankInput:
    doc_id: str
    text: str
    score: float
    metadata: dict


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def rerank(self, query: str, docs: list[RerankInput], top_k: int = 20) -> list[RerankInput]:
        if os.getenv("SEMANTICATS_ENABLE_LOCAL_MODELS", "").casefold() not in {"1", "true", "yes"}:
            return docs[:top_k]
        try:
            model = self._load()
        except Exception as exc:  # pragma: no cover - depends on model availability
            logger.warning("Reranker unavailable; preserving hybrid order: %s", exc)
            return docs[:top_k]
        pairs = [(query, doc.text) for doc in docs]
        scores = model.predict(pairs)
        updated = [
            RerankInput(doc_id=doc.doc_id, text=doc.text, score=float(score), metadata=doc.metadata)
            for doc, score in zip(docs, scores, strict=False)
        ]
        return sorted(updated, key=lambda doc: doc.score, reverse=True)[:top_k]

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model
