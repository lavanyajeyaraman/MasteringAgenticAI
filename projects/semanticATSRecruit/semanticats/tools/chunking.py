from __future__ import annotations

from semanticats.models import CandidateProfile, ResumeChunk


def chunk_candidate(profile: CandidateProfile) -> list[ResumeChunk]:
    chunks: list[ResumeChunk] = []
    for skill in profile.skills:
        chunks.append(
            ResumeChunk(
                id=f"{profile.candidate_id}:skill:{_slug(skill)}",
                candidate_id=profile.candidate_id,
                candidate_name=profile.name,
                years_experience=profile.years_experience,
                skills=[skill],
                chunk_type="skill",
                source_text=f"{profile.name} lists {skill} as a resume skill.",
            )
        )
    summary = _summary(profile)
    if summary:
        chunks.append(
            ResumeChunk(
                id=f"{profile.candidate_id}:summary",
                candidate_id=profile.candidate_id,
                candidate_name=profile.name,
                years_experience=profile.years_experience,
                skills=profile.skills,
                chunk_type="summary",
                source_text=summary,
            )
        )
    for index, text in enumerate(_sliding_windows(profile.source_text), start=1):
        chunks.append(
            ResumeChunk(
                id=f"{profile.candidate_id}:narrative:{index}",
                candidate_id=profile.candidate_id,
                candidate_name=profile.name,
                years_experience=profile.years_experience,
                skills=[skill for skill in profile.skills if skill.casefold() in text.casefold()],
                chunk_type="narrative",
                source_text=text,
            )
        )
    return chunks


def _summary(profile: CandidateProfile) -> str:
    pieces = [profile.name]
    if profile.years_experience is not None:
        pieces.append(f"{profile.years_experience:g} years experience")
    if profile.skills:
        pieces.append("Skills: " + ", ".join(profile.skills))
    if profile.projects:
        pieces.append("Projects: " + "; ".join(profile.projects[:3]))
    return ". ".join(pieces)


def _sliding_windows(text: str, window: int = 90, overlap: int = 25) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(window - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + window])
        if len(chunk.split()) >= 8:
            chunks.append(chunk)
        if start + window >= len(words):
            break
    return chunks


def _slug(value: str) -> str:
    return value.strip().casefold().replace(" ", "-")
