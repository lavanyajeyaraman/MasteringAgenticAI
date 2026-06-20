from __future__ import annotations

from collections import defaultdict
from typing import Protocol


class RankedItem(Protocol):
    doc_id: str
    score: float


def reciprocal_rank_fusion(
    result_sets: list[list[RankedItem]],
    *,
    k: int = 60,
    limit: int = 20,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for results in result_sets:
        for rank, item in enumerate(results, start=1):
            scores[item.doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
