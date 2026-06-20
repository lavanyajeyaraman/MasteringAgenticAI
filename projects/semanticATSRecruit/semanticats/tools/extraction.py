from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from semanticats.config import Settings
from semanticats.models import CandidateProfile, JobDescription
from semanticats.prompts.jd_extraction import JD_EXTRACTION_SYSTEM
from semanticats.prompts.resume_extraction import RESUME_EXTRACTION_SYSTEM
from semanticats.tools.llm_service import NebiusLLMService

KNOWN_SKILLS = [
    "Python",
    "FastAPI",
    "LangGraph",
    "CrewAI",
    "AutoGen",
    "LlamaIndex",
    "Playwright",
    "Selenium",
    "Cypress",
    "Puppeteer",
    "Kubernetes",
    "ECS",
    "GKE",
    "Helm",
    "Docker",
    "Docker Swarm",
    "Azure AI",
    "Vertex AI",
    "SageMaker",
    "Bedrock",
    "MLOps",
    "API Testing",
]


def extract_jd(text: str, settings: Settings, llm: NebiusLLMService | None = None) -> JobDescription:
    llm = llm or NebiusLLMService(settings.nebius_api_key)
    data = llm.complete_json(
        model=settings.llm_model_extract,
        system=JD_EXTRACTION_SYSTEM,
        user=text,
    )
    if data:
        return JobDescription.model_validate(data)
    skills = _find_skills(text)
    required, preferred = _split_required_preferred(text, skills)
    return JobDescription(
        required_skills=required,
        preferred_skills=preferred,
        seniority=_extract_seniority(text),
        responsibilities=_extract_bullets(text),
        implicit_requirements=_infer_implicit_requirements(text),
    )


def extract_resume(
    text: str,
    *,
    filename: str = "resume.txt",
    settings: Settings,
    llm: NebiusLLMService | None = None,
) -> CandidateProfile:
    llm = llm or NebiusLLMService(settings.nebius_api_key)
    data = llm.complete_json(
        model=settings.llm_model_extract,
        system=RESUME_EXTRACTION_SYSTEM,
        user=text,
    )
    if data:
        data.setdefault("candidate_id", Path(filename).stem or str(uuid4()))
        data.setdefault("source_text", text)
        return CandidateProfile.model_validate(data)
    return CandidateProfile(
        candidate_id=_candidate_id(filename),
        name=_extract_name(text, filename),
        skills=_find_skills(text),
        certifications=_extract_lines_after(text, "certification"),
        years_experience=_extract_years(text),
        companies=_extract_companies(text),
        projects=_extract_project_lines(text),
        education=_extract_lines_after(text, "education"),
        source_text=text,
    )


def _find_skills(text: str) -> list[str]:
    found = []
    lower = text.casefold()
    for skill in KNOWN_SKILLS:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(skill.casefold())}(?![A-Za-z0-9])", lower):
            found.append(skill)
    return found


def _split_required_preferred(text: str, skills: list[str]) -> tuple[list[str], list[str]]:
    required, preferred = [], []
    for skill in skills:
        window = _context_window(text, skill)
        if "preferred" in window or "nice to have" in window:
            preferred.append(skill)
        else:
            required.append(skill)
    return required, preferred


def _extract_seniority(text: str) -> str | None:
    for seniority in ("Principal", "Staff", "Senior", "Lead", "Mid", "Junior"):
        if seniority.casefold() in text.casefold():
            return seniority
    return None


def _extract_bullets(text: str) -> list[str]:
    bullets = []
    for line in text.splitlines():
        clean = line.strip(" -*\t")
        if len(clean.split()) >= 4:
            bullets.append(clean)
    return bullets[:12]


def _infer_implicit_requirements(text: str) -> list[str]:
    implicit = []
    lower = text.casefold()
    if "production" in lower:
        implicit.append("Production system experience")
    if "scale" in lower or "scalable" in lower:
        implicit.append("Scalable architecture experience")
    if "agent" in lower:
        implicit.append("Agentic workflow design")
    if "cloud" in lower:
        implicit.append("Cloud deployment experience")
    return implicit


def _extract_name(text: str, filename: str) -> str:
    for line in text.splitlines():
        clean = line.strip()
        if clean and len(clean.split()) <= 4 and not any(ch.isdigit() for ch in clean):
            return clean
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def _extract_years(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\+?\s+years?", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _extract_companies(text: str) -> list[str]:
    companies = []
    for pattern in (r"at\s+([A-Z][A-Za-z0-9&.\- ]{2,40})", r"Company:\s*(.+)"):
        companies.extend(match.strip() for match in re.findall(pattern, text))
    return list(dict.fromkeys(companies))[:8]


def _extract_project_lines(text: str) -> list[str]:
    return [
        line.strip(" -*\t")
        for line in text.splitlines()
        if "project" in line.casefold() or "built" in line.casefold() or "implemented" in line.casefold()
    ][:8]


def _extract_lines_after(text: str, keyword: str) -> list[str]:
    lines = text.splitlines()
    results = []
    for index, line in enumerate(lines):
        if keyword in line.casefold():
            results.extend(item.strip(" -*\t") for item in lines[index + 1 : index + 4] if item.strip())
    return list(dict.fromkeys(results))[:6]


def _context_window(text: str, skill: str, width: int = 80) -> str:
    lower = text.casefold()
    index = lower.find(skill.casefold())
    if index < 0:
        return ""
    return lower[max(0, index - width) : index + len(skill) + width]


def _candidate_id(filename: str) -> str:
    stem = Path(filename).stem.strip()
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-").casefold() or str(uuid4())
