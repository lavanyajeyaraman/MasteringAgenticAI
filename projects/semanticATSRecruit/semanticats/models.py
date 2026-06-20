from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Recommendation(str, Enum):
    STRONG_HIRE = "Strong Hire"
    HIRE = "Hire"
    CONSIDER = "Consider"
    REJECT = "Reject"


class CandidateProfile(BaseModel):
    candidate_id: str
    name: str = "Unknown Candidate"
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    years_experience: float | None = None
    companies: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    source_text: str = ""


class JobDescription(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    seniority: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    implicit_requirements: list[str] = Field(default_factory=list)


class ResumeChunk(BaseModel):
    id: str
    candidate_id: str
    candidate_name: str
    years_experience: float | None = None
    skills: list[str] = Field(default_factory=list)
    chunk_type: str
    source_text: str
    embedding: list[float] | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "years_experience": self.years_experience,
            "skills": self.skills,
            "chunk_type": self.chunk_type,
            "source_text": self.source_text,
        }


class CandidateRank(BaseModel):
    candidate_id: str
    candidate_name: str
    score: float
    evidence: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    semantic_matches: list[dict[str, Any]] = Field(default_factory=list)


class CandidateReport(BaseModel):
    candidate_id: str
    candidate_name: str
    direct_matches: list[dict[str, str]] = Field(default_factory=list)
    semantic_matches: list[dict[str, Any]] = Field(default_factory=list)
    transferable_skills: list[dict[str, Any]] = Field(default_factory=list)
    transferability_score: float = 0.0
    gaps: list[str] = Field(default_factory=list)
    ramp_up_estimate: str = "Unknown"
    ats_rejection_reason: str = "No traditional ATS rejection reason identified."
    hiring_recommendation: Recommendation = Recommendation.CONSIDER
    recruiter_summary: str = ""


class VerificationResult(BaseModel):
    claim: str
    evidence: str
    confidence: float
    supported: bool
