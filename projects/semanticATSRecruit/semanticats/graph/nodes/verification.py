from __future__ import annotations

from semanticats.config import get_settings
from semanticats.models import VerificationResult
from semanticats.state import RecruitingState


def verify_reports(state: RecruitingState) -> RecruitingState:
    if state.get("faithfulness_flags") or all(
        "verification" in report for report in state.get("reports", [])
    ):
        return state
    settings = get_settings()
    flags = []
    for report in state.get("reports", []):
        checks = _verify_report(report)
        report["verification"] = [check.model_dump() for check in checks]
        flags.extend(
            {
                "candidate_id": report["candidate_id"],
                "candidate_name": report["candidate_name"],
                **check.model_dump(),
            }
            for check in checks
            if check.confidence < settings.faithfulness_threshold or not check.supported
        )
    return {**state, "faithfulness_flags": flags}


def hotl_faithfulness(state: RecruitingState) -> RecruitingState:
    return state


def _verify_report(report: dict) -> list[VerificationResult]:
    checks = []
    for direct in report.get("direct_matches", []):
        skill = direct.get("skill", "")
        evidence = direct.get("evidence", "")
        supported = bool(skill and evidence and skill.casefold() in evidence.casefold())
        checks.append(
            VerificationResult(
                claim=f"Candidate has direct skill evidence for {skill}.",
                evidence=evidence,
                confidence=0.95 if supported else 0.45,
                supported=supported,
            )
        )
    for semantic in report.get("semantic_matches", []):
        matched = semantic.get("matched_skill", "")
        jd_skill = semantic.get("jd_skill", "")
        evidence = semantic.get("evidence", "")
        supported = bool(matched and evidence and matched.casefold() in evidence.casefold())
        checks.append(
            VerificationResult(
                claim=f"{matched} is transferable to {jd_skill}.",
                evidence=evidence,
                confidence=min(float(semantic.get("confidence", 0.0)), 0.92) if supported else 0.4,
                supported=supported,
            )
        )
    if not checks:
        checks.append(
            VerificationResult(
                claim="Report contains evidence-backed candidate match claims.",
                evidence="",
                confidence=0.0,
                supported=False,
            )
        )
    return checks
