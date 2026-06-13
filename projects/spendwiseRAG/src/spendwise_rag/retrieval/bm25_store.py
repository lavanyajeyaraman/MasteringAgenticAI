from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from spendwise_rag.core.models import Chunk


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_bm25(chunks: list[Chunk]) -> BM25Okapi:
    return BM25Okapi([tokenize(chunk.text) for chunk in chunks])


def persist_bm25(chunks: list[Chunk], namespace: str, index_dir: str | Path = "data/indexes") -> str:
    output = Path(index_dir) / f"{namespace}_bm25.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "namespace": namespace,
        "chunk_texts": [chunk.text for chunk in chunks],
        "chunk_metadata": [chunk.metadata for chunk in chunks],
        "bm25": build_bm25(chunks),
    }
    with output.open("wb") as file:
        pickle.dump(payload, file)
    return str(output)


def load_bm25_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def search_bm25_payload(payload: dict[str, Any], query: str, top_k: int = 20) -> list[dict[str, Any]]:
    bm25 = payload.get("bm25")
    chunk_texts = list(payload.get("chunk_texts", []))
    chunk_metadata = list(payload.get("chunk_metadata", []))
    if bm25 is None or not chunk_texts:
        return []

    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
    results: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        if float(score) <= 0:
            continue
        metadata = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
        results.append(
            {
                "id": f"{payload.get('namespace', 'bm25')}_{idx:04d}",
                "text": chunk_texts[idx],
                "metadata": metadata,
                "score": float(score),
                "rank": rank,
                "source": "bm25",
            }
        )
        if len(results) >= top_k:
            break
    return results
