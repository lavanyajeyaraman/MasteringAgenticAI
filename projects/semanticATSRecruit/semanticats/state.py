from __future__ import annotations

from typing import Any, TypedDict


class RecruitingState(TypedDict, total=False):
    jd_text: str
    jd_structured: dict[str, Any]
    jd_approved: bool
    candidates_raw: list[dict[str, Any]]
    indexed_candidates: list[dict[str, Any]]
    ranked_candidates: list[dict[str, Any]]
    shortlist_approved: list[dict[str, Any]]
    reports: list[dict[str, Any]]
    selected_reports: list[str]
    recruiter_filters: dict[str, Any]
    conversation_history: list[dict[str, str]]
    taxonomy_audit: list[dict[str, Any]]
    faithfulness_flags: list[dict[str, Any]]
    interrupts: list[dict[str, Any]]
    paused_at: str
    interactive_hitl: bool
    rejected_at: str
    rejection_reason: str
    errors: list[str]
