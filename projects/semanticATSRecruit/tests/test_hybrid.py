from dataclasses import dataclass

from semanticats.tools.hybrid import reciprocal_rank_fusion


@dataclass
class Result:
    doc_id: str
    score: float


def test_rrf_merges_rankings_by_reciprocal_rank() -> None:
    merged = reciprocal_rank_fusion(
        [
            [Result("a", 0.9), Result("b", 0.8)],
            [Result("b", 0.99), Result("c", 0.4)],
        ],
        k=60,
        limit=3,
    )

    assert merged[0][0] == "b"
    assert {doc_id for doc_id, _ in merged} == {"a", "b", "c"}
