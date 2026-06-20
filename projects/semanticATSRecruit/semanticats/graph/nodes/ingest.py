from __future__ import annotations

from typing import Any

from semanticats.config import get_settings
from semanticats.models import CandidateProfile
from semanticats.state import RecruitingState
from semanticats.tools.chunking import chunk_candidate
from semanticats.tools.embedder import Embedder
from semanticats.tools.extraction import extract_jd, extract_resume
from semanticats.tools.pdf_parser import parse_pdf


def ingest_jd(state: RecruitingState) -> RecruitingState:
    if state.get("jd_structured") and state.get("jd_approved"):
        return state
    settings = get_settings()
    jd_text = state.get("jd_text", "")
    if not jd_text.strip():
        return _append_error(state, "Job description text is required.")
    jd = extract_jd(jd_text, settings=settings)
    return {**state, "jd_structured": jd.model_dump(), "jd_approved": bool(state.get("jd_approved"))}


def ingest_resumes(state: RecruitingState) -> RecruitingState:
    raw_candidates = state.get("candidates_raw", [])
    if raw_candidates and all("candidate_id" in item for item in raw_candidates):
        return state
    settings = get_settings()
    candidates = []
    for item in raw_candidates:
        text = item.get("text")
        if not text and item.get("path"):
            text = parse_pdf(item["path"])
        if not text:
            continue
        candidates.append(
            extract_resume(
                text,
                filename=item.get("filename", item.get("path", "resume.txt")),
                settings=settings,
            ).model_dump()
        )
    return {**state, "candidates_raw": candidates}


def index_candidates(state: RecruitingState) -> RecruitingState:
    if state.get("indexed_candidates"):
        return state
    settings = get_settings()
    embedder = Embedder(settings.embedding_model)
    chunks = []
    for candidate in state.get("candidates_raw", []):
        profile = CandidateProfile.model_validate(candidate)
        chunks.extend(chunk_candidate(profile))
    embeddings = embedder.embed([chunk.source_text for chunk in chunks]) if chunks else []
    indexed = []
    for chunk, embedding in zip(chunks, embeddings, strict=False):
        chunk.embedding = embedding
        indexed.append(chunk.model_dump())
    return {**state, "indexed_candidates": indexed}


def _append_error(state: RecruitingState, message: str) -> RecruitingState:
    return {**state, "errors": [*state.get("errors", []), message]}
