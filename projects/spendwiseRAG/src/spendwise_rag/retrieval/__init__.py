from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path

from spendwise_rag.core.models import Chunk, LocalIndex


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_index(chunks: list[Chunk], namespace: str) -> LocalIndex:
    return LocalIndex(namespace=namespace, chunks=chunks, tokenized=[tokenize(chunk.text) for chunk in chunks])


def save_index(index: LocalIndex, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(index, file)


def load_index(path: str | Path) -> LocalIndex:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def search(index: LocalIndex, query: str, top_k: int = 6) -> list[tuple[Chunk, float]]:
    query_terms = tokenize(query)
    if not query_terms or not index.chunks:
        return []
    doc_count = len(index.tokenized)
    doc_freq = Counter(term for doc in index.tokenized for term in set(doc))
    avg_len = sum(len(doc) for doc in index.tokenized) / max(1, doc_count)
    scores: list[tuple[int, float]] = []
    for idx, doc in enumerate(index.tokenized):
        counts = Counter(doc)
        score = 0.0
        for term in query_terms:
            if term not in counts:
                continue
            idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            tf = counts[term]
            denom = tf + 1.5 * (1 - 0.75 + 0.75 * len(doc) / max(1, avg_len))
            score += idf * (tf * 2.5) / denom
        if score > 0:
            scores.append((idx, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [(index.chunks[idx], score) for idx, score in scores[:top_k]]
