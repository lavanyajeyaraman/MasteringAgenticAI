from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9+#.\-]+")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


@dataclass
class BM25Result:
    doc_id: str
    score: float
    text: str
    metadata: dict


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[tuple[str, str, dict, list[str]]] = []
        self.doc_freqs: Counter[str] = Counter()
        self.avgdl = 0.0

    def add_documents(self, docs: list[tuple[str, str, dict]]) -> None:
        for doc_id, text, metadata in docs:
            tokens = tokenize(text)
            self.documents.append((doc_id, text, metadata, tokens))
            self.doc_freqs.update(set(tokens))
        self.avgdl = sum(len(tokens) for _, _, _, tokens in self.documents) / max(
            len(self.documents), 1
        )

    def search(self, query: str, top_k: int = 20) -> list[BM25Result]:
        query_tokens = tokenize(query)
        scored = []
        for doc_id, text, metadata, tokens in self.documents:
            score = self._score(query_tokens, tokens)
            if score > 0:
                scored.append(BM25Result(doc_id=doc_id, score=score, text=text, metadata=metadata))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not self.documents or not doc_tokens:
            return 0.0
        term_freq = Counter(doc_tokens)
        score = 0.0
        for token in query_tokens:
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
            numerator = term_freq[token] * (self.k1 + 1)
            denominator = term_freq[token] + self.k1 * (
                1 - self.b + self.b * len(doc_tokens) / max(self.avgdl, 1)
            )
            score += idf * numerator / max(denominator, 1e-9)
        return score
