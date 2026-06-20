from __future__ import annotations

from semanticats.state import RecruitingState


def display_results(state: RecruitingState) -> RecruitingState:
    selected = set(state.get("selected_reports", []))
    if selected:
        visible = [report for report in state.get("reports", []) if report["candidate_id"] in selected]
    else:
        visible = state.get("reports", [])
    return {**state, "reports": visible}


def conversation_loop(state: RecruitingState) -> RecruitingState:
    return state
