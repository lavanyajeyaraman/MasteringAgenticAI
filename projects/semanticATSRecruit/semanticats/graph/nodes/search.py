from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from semanticats.config import get_settings
from semanticats.models import JobDescription
from semanticats.state import RecruitingState
from semanticats.tools.bm25_index import BM25Index
from semanticats.tools.embedder import Embedder
from semanticats.tools.hybrid import reciprocal_rank_fusion
from semanticats.tools.pinecone_client import PineconeClient, VectorResult
from semanticats.tools.reranker import CrossEncoderReranker, RerankInput
from semanticats.tools.session_overrides import apply_filters
from semanticats.tools.skill_graph import SkillGraph


@dataclass
class _Ranked:
    doc_id: str
    score: float


def search_candidates(state: RecruitingState) -> RecruitingState:
    if state.get("ranked_candidates"):
        return state
    settings = get_settings()
    jd = JobDescription.model_validate(state.get("jd_structured", {}))
    chunks = state.get("indexed_candidates", [])
    skill_graph = SkillGraph.default()
    expanded_query = _expanded_query(jd, skill_graph)

    bm25 = BM25Index()
    bm25.add_documents(
        [(chunk["id"], chunk["source_text"], chunk) for chunk in chunks]
    )
    bm25_results = bm25.search(expanded_query, top_k=settings.top_k)

    embedder = Embedder(settings.embedding_model)
    pinecone = PineconeClient(settings.pinecone_api_key, settings.pinecone_index)
    pinecone.upsert(
        [
            (chunk["id"], chunk.get("embedding") or embedder.embed([chunk["source_text"]])[0], chunk)
            for chunk in chunks
        ]
    )
    query_vector = embedder.embed([expanded_query])[0]
    vector_results = pinecone.query(query_vector, top_k=settings.top_k)

    fused = reciprocal_rank_fusion(
        [
            [_Ranked(doc_id=result.doc_id, score=result.score) for result in bm25_results],
            [_Ranked(doc_id=result.doc_id, score=result.score) for result in vector_results],
        ],
        k=settings.rrf_k,
        limit=settings.top_k,
    )
    doc_map = {
        **{result.doc_id: (result.text, result.metadata) for result in bm25_results},
        **{
            result.doc_id: (result.metadata.get("source_text", ""), result.metadata)
            for result in vector_results
        },
    }
    rerank_inputs = [
        RerankInput(doc_id=doc_id, text=doc_map[doc_id][0], score=score, metadata=doc_map[doc_id][1])
        for doc_id, score in fused
        if doc_id in doc_map
    ]
    reranked = CrossEncoderReranker(settings.reranker_model).rerank(
        expanded_query, rerank_inputs, top_k=settings.top_k
    )
    ranked_candidates = _aggregate_candidates(reranked, jd, skill_graph)
    ranked_candidates = apply_filters(ranked_candidates, state.get("recruiter_filters", {}))
    taxonomy = _taxonomy_audit(ranked_candidates)
    return {**state, "ranked_candidates": ranked_candidates, "taxonomy_audit": taxonomy}


def hotl_taxonomy(state: RecruitingState) -> RecruitingState:
    return state


def _expanded_query(jd: JobDescription, skill_graph: SkillGraph) -> str:
    skills = [*jd.required_skills, *jd.preferred_skills]
    expanded = []
    for skill in skills:
        expanded.extend(skill_graph.expand(skill))
    return " ".join([*skills, *expanded, *jd.responsibilities, *jd.implicit_requirements])


def _aggregate_candidates(
    docs: list[RerankInput], jd: JobDescription, skill_graph: SkillGraph
) -> list[dict[str, Any]]:
    by_candidate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "candidate_id": "",
            "candidate_name": "",
            "score": 0.0,
            "evidence": [],
            "matched_skills": [],
            "semantic_matches": [],
            "years_experience": None,
        }
    )
    jd_skills = [*jd.required_skills, *jd.preferred_skills]
    for doc in docs:
        cid = doc.metadata["candidate_id"]
        item = by_candidate[cid]
        item["candidate_id"] = cid
        item["candidate_name"] = doc.metadata["candidate_name"]
        item["years_experience"] = doc.metadata.get("years_experience")
        item["score"] += float(doc.score)
        if doc.text not in item["evidence"]:
            item["evidence"].append(doc.text)
        for candidate_skill in doc.metadata.get("skills", []):
            if candidate_skill not in item["matched_skills"]:
                item["matched_skills"].append(candidate_skill)
            for jd_skill in jd_skills:
                match = skill_graph.match(jd_skill, candidate_skill)
                if match and match not in item["semantic_matches"]:
                    item["semantic_matches"].append(match)
                    item["score"] += float(match["confidence"])
    ranked = sorted(by_candidate.values(), key=lambda item: item["score"], reverse=True)
    return ranked


def _taxonomy_audit(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        for match in candidate.get("semantic_matches", []):
            if match.get("relation") != "direct":
                rows.append({**match, "candidate": candidate["candidate_name"]})
    return rows
