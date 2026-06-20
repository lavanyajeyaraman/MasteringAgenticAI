from __future__ import annotations

from semanticats.models import CandidateReport, JobDescription, Recommendation
from semanticats.state import RecruitingState
from semanticats.tools.skill_graph import SkillGraph


def generate_reports(state: RecruitingState) -> RecruitingState:
    if state.get("reports"):
        return state
    jd = JobDescription.model_validate(state.get("jd_structured", {}))
    skill_graph = SkillGraph.default()
    reports = [
        _build_report(candidate, jd, skill_graph).model_dump(mode="json")
        for candidate in state.get("shortlist_approved", [])
    ]
    return {**state, "reports": reports}


def _build_report(candidate: dict, jd: JobDescription, skill_graph: SkillGraph) -> CandidateReport:
    jd_skills = [*jd.required_skills, *jd.preferred_skills]
    candidate_skills = candidate.get("matched_skills", [])
    evidence = candidate.get("evidence", [])
    direct_matches = []
    semantic_matches = []
    for jd_skill in jd_skills:
        for candidate_skill in candidate_skills:
            match = skill_graph.match(jd_skill, candidate_skill)
            if not match:
                continue
            evidence_text = _find_evidence(candidate_skill, evidence)
            if match["relation"] == "direct":
                direct_matches.append(
                    {"skill": jd_skill, "evidence": evidence_text or candidate_skill}
                )
            else:
                semantic_matches.append({**match, "evidence": evidence_text or candidate_skill})
    transferability = skill_graph.transferability_score(jd_skills, candidate_skills)
    gaps = [
        skill
        for skill in jd.required_skills
        if not any(match.get("jd_skill") == skill for match in [*semantic_matches])
        and not any(row.get("skill") == skill for row in direct_matches)
    ]
    recommendation = _recommendation(
        transferability,
        gaps,
        matched_count=len(direct_matches) + len(semantic_matches),
    )
    return CandidateReport(
        candidate_id=candidate["candidate_id"],
        candidate_name=candidate["candidate_name"],
        direct_matches=direct_matches,
        semantic_matches=semantic_matches,
        transferable_skills=semantic_matches,
        transferability_score=transferability,
        gaps=gaps,
        ramp_up_estimate=_ramp_up(transferability, gaps),
        ats_rejection_reason=_ats_rejection_reason(jd.required_skills, candidate_skills, semantic_matches),
        hiring_recommendation=recommendation,
        recruiter_summary=_summary(candidate["candidate_name"], recommendation, transferability, direct_matches, semantic_matches, gaps),
    )


def _find_evidence(skill: str, evidence: list[str]) -> str:
    for item in evidence:
        if skill.casefold() in item.casefold():
            return item
    return evidence[0] if evidence else ""


def _recommendation(score: float, gaps: list[str], matched_count: int = 0) -> Recommendation:
    if score >= 0.9 and len(gaps) <= 1:
        return Recommendation.STRONG_HIRE
    if score >= 0.75 and len(gaps) <= 2:
        return Recommendation.HIRE
    if score >= 0.45 or matched_count >= 2:
        return Recommendation.CONSIDER
    return Recommendation.REJECT


def _ramp_up(score: float, gaps: list[str]) -> str:
    if score >= 0.9:
        return "Low ramp-up: adjacent or direct experience covers most requirements."
    if score >= 0.75:
        return "Moderate ramp-up: transferable experience is present, with focused onboarding for gaps."
    return "High ramp-up: several required capabilities lack resume evidence."


def _ats_rejection_reason(
    required_skills: list[str], candidate_skills: list[str], semantic_matches: list[dict]
) -> str:
    missed = [
        skill
        for skill in required_skills
        if skill.casefold() not in {candidate.casefold() for candidate in candidate_skills}
        and any(match.get("jd_skill") == skill for match in semantic_matches)
    ]
    if missed:
        return "Traditional keyword ATS may reject this candidate because the resume uses adjacent terminology for: " + ", ".join(missed)
    return "No major keyword-only rejection pattern identified."


def _summary(
    name: str,
    recommendation: Recommendation,
    score: float,
    direct: list[dict],
    semantic: list[dict],
    gaps: list[str],
) -> str:
    matched = len(direct) + len(semantic)
    gap_text = f" Gaps: {', '.join(gaps)}." if gaps else " No critical gaps were identified from the extracted JD skills."
    return (
        f"{name} is a {recommendation.value} with transferability score {score:.2f}. "
        f"The report is based on {matched} evidence-backed direct or semantic matches."
        f"{gap_text}"
    )
