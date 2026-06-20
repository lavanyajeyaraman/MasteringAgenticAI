from __future__ import annotations

from semanticats.state import RecruitingState


def hitl_jd_review(state: RecruitingState) -> RecruitingState:
    if state.get("jd_approved"):
        return state
    if not state.get("interactive_hitl"):
        return {**state, "jd_approved": True}
    return _interrupt(
        state,
        "jd_review",
        "Review extracted requirements, edit if needed, then approve.",
        state.get("jd_structured", {}),
    )


def hitl_shortlist(state: RecruitingState) -> RecruitingState:
    if state.get("shortlist_approved"):
        return state
    ranked = state.get("ranked_candidates", [])
    if not state.get("interactive_hitl"):
        return {**state, "shortlist_approved": ranked[:5]}
    return _interrupt(
        state,
        "shortlist_review",
        "Review ranked candidates and approve a subset.",
        ranked[:10],
    )


def hitl_report_gate(state: RecruitingState) -> RecruitingState:
    if state.get("selected_reports"):
        return state
    reports = state.get("reports", [])
    if not state.get("interactive_hitl"):
        return {
            **state,
            "selected_reports": [report.get("candidate_id") for report in reports],
        }
    return _interrupt(
        state,
        "report_gate",
        "Select candidates for detailed report rendering.",
        reports,
    )


def _interrupt(
    state: RecruitingState, checkpoint: str, message: str, payload: object
) -> RecruitingState:
    interrupts = [
        item
        for item in state.get("interrupts", [])
        if item.get("checkpoint") != checkpoint
    ]
    return {
        **state,
        "paused_at": checkpoint,
        "interrupts": [
            *interrupts,
            {
                "checkpoint": checkpoint,
                "message": message,
                "payload": payload,
                "status": "review_required",
            },
        ],
    }
