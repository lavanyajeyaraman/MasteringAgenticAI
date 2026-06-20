from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VectorResult:
    doc_id: str
    score: float
    metadata: dict


class PineconeClient:
    def __init__(self, api_key: str | None, index_name: str) -> None:
        self.api_key = api_key
        self.index_name = index_name
        self._index = None
        self._memory: dict[str, tuple[list[float], dict]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def upsert(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        if not self.enabled:
            for doc_id, vector, metadata in vectors:
                self._memory[doc_id] = (vector, metadata)
            return
        try:  # pragma: no cover - requires external service
            from pinecone import Pinecone

            index = self._index or Pinecone(api_key=self.api_key).Index(self.index_name)
            self._index = index
            index.upsert(vectors=[{"id": i, "values": v, "metadata": m} for i, v, m in vectors])
        except Exception as exc:
            logger.warning("Pinecone unavailable; storing vectors in memory: %s", exc)
            for doc_id, vector, metadata in vectors:
                self._memory[doc_id] = (vector, metadata)

    def query(self, vector: list[float], top_k: int = 20) -> list[VectorResult]:
        if self.enabled and self._index:  # pragma: no cover - requires external service
            response = self._index.query(vector=vector, top_k=top_k, include_metadata=True)
            return [
                VectorResult(doc_id=match["id"], score=match["score"], metadata=match["metadata"])
                for match in response["matches"]
            ]
        scored = []
        for doc_id, (stored_vector, metadata) in self._memory.items():
            scored.append(VectorResult(doc_id=doc_id, score=_cosine(vector, stored_vector), metadata=metadata))
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b, strict=False))
    den_a = sum(x * x for x in a) ** 0.5
    den_b = sum(y * y for y in b) ** 0.5
    return numerator / max(den_a * den_b, 1e-9)
