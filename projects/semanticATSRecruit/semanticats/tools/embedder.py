from __future__ import annotations

import hashlib
import logging
import os
import random

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, dimension: int = 384) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model = None

    def _load(self) -> object | None:
        if self._model is not None:
            return self._model
        if os.getenv("SEMANTICATS_ENABLE_LOCAL_MODELS", "").casefold() not in {"1", "true", "yes"}:
            self._model = False
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # pragma: no cover - depends on local model availability
            logger.warning("Using deterministic fallback embeddings: %s", exc)
            self._model = False
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        if model:
            vectors = model.encode(texts, normalize_embeddings=True)
            return [vector.tolist() for vector in vectors]
        return [self._fallback_embedding(text) for text in texts]

    def _fallback_embedding(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        vector = [rng.uniform(-1.0, 1.0) for _ in range(self.dimension)]
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]
