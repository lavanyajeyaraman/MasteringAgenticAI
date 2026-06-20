from __future__ import annotations

from typing import Any

import streamlit as st

from semanticats.graph.graph import build_graph
from semanticats.state import RecruitingState
from semanticats.tools.pdf_parser import parse_pdf
from semanticats.tools.session_overrides import parse_override
from semanticats.ui.components import report_to_pdf
from semanticats.ui.interrupt_widgets import render_interrupts


DEMO_RESUMES = {
    "selenium": """Alex Rivera
Senior QA Automation Engineer with 7 years experience.
Skills: Selenium, Python, API Testing, Test Framework Design, Docker.
Implemented project test suites for payment workflows using Selenium Grid and pytest.
""",
    "crewai": """Maya Chen
Agentic AI Engineer with 6 years experience.
Skills: CrewAI, LangChain, Python, FastAPI, LlamaIndex.
Built multi-agent workflow orchestration for support triage and retrieval augmented generation.
""",
    "ecs": """Jordan Patel
Cloud Platform Engineer with 8 years experience.
Skills: ECS, Docker, Terraform, AWS, Containerization.
Implemented production services on Amazon ECS with autoscaling and deployment automation.
""",
}


def run() -> None:
    st.set_page_config(page_title="semanticATSRecruit", layout="wide")
    _inject_styles()
    st.title("semanticATSRecruit")

    if "result" not in st.session_state:
        st.session_state.result = {}
    if "filters" not in st.session_state:
        st.session_state.filters = {}

    with st.sidebar:
        st.subheader("Resume Upload")
        uploads = st.file_uploader("Upload resumes", accept_multiple_files=True, type=["pdf", "txt"])
        include_demo = st.checkbox("Include demo resumes", value=True)
        st.subheader("Session Overrides")
        command = st.text_input("Natural language command", placeholder="Only show candidates with 5+ years")
        if st.button("Apply Override") and command:
            st.session_state.filters.update(parse_override(command))
        if st.session_state.filters:
            _render_filter_chips(st.session_state.filters)
        else:
            st.markdown('<div class="empty-state">No active filters</div>', unsafe_allow_html=True)
        st.subheader("Taxonomy Audit")
        audit = st.session_state.result.get("taxonomy_audit", [])
        if audit:
            st.dataframe(audit, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                '<div class="empty-state">Semantic matches will appear after search</div>',
                unsafe_allow_html=True,
            )

    search_tab, reports_tab, chat_tab = st.tabs(["Search", "Reports", "Chat"])

    with search_tab:
        jd_text = st.text_area(
            "Job Description",
            height=220,
            value="Senior engineer required with Playwright, LangGraph, Kubernetes, and Python experience for production AI systems.",
        )
        if st.button("Search Candidates", type="primary"):
            candidates = []
            for upload in uploads or []:
                candidates.append({"filename": upload.name, "text": _read_upload(upload)})
            if include_demo:
                candidates.extend(
                    {"filename": f"{name}.txt", "text": text}
                    for name, text in DEMO_RESUMES.items()
                )
            state: RecruitingState = {
                "jd_text": jd_text,
                "candidates_raw": candidates,
                "recruiter_filters": st.session_state.filters,
                "conversation_history": [],
                "interactive_hitl": True,
            }
            st.session_state.result = build_graph().invoke(
                state, config={"configurable": {"thread_id": "streamlit"}}
            )
        result = st.session_state.result
        if result:
            _render_summary_metrics(result)
            if result.get("rejected_at"):
                st.error(
                    f"Workflow rejected at {result['rejected_at']}: "
                    f"{result.get('rejection_reason', 'No reason provided.')}"
                )
            st.subheader("JD Review")
            _render_jd_review(result.get("jd_structured", {}))
            render_interrupts(result.get("interrupts", []))
            _render_hitl_controls(result)
            st.subheader("Candidate Ranking")
            _render_ranked_candidates(result.get("ranked_candidates", []))

    with reports_tab:
        result = st.session_state.result
        flags = result.get("faithfulness_flags", [])
        if flags:
            st.warning(f"{len(flags)} faithfulness warning(s) need recruiter review.")
            st.dataframe(flags, use_container_width=True, hide_index=True)
        if result.get("paused_at") == "report_gate":
            st.info("Approve the report gate checkpoint before rendering final reports.")
        elif not result.get("reports"):
            st.info("Run a search to generate candidate reports.")
        for report in [] if result.get("paused_at") == "report_gate" else result.get("reports", []):
            with st.expander(f"{report['candidate_name']} - {report['hiring_recommendation']}", expanded=True):
                st.write(report["recruiter_summary"])
                st.metric("Transferability", report["transferability_score"])
                st.write("Direct matches")
                _render_match_list(report.get("direct_matches", []), "No exact skill matches found.")
                st.write("Semantic matches")
                _render_match_list(
                    report.get("semantic_matches", []),
                    "No transferable skill matches found.",
                )
                st.download_button(
                    "Download PDF",
                    data=report_to_pdf(report),
                    file_name=f"{report['candidate_id']}-semanticatsrecruit-report.pdf",
                    mime="application/pdf",
                )

    with chat_tab:
        question = st.chat_input("Ask about rankings, gaps, or comparisons")
        if question:
            st.session_state.setdefault("chat", []).append({"role": "user", "content": question})
            answer = _answer_question(question, st.session_state.result)
            st.session_state.chat.append({"role": "assistant", "content": answer})
        for message in st.session_state.get("chat", []):
            with st.chat_message(message["role"]):
                st.write(message["content"])


def _answer_question(question: str, result: dict) -> str:
    lower = question.casefold()
    ranked = result.get("ranked_candidates", [])
    reports = result.get("reports", [])
    if not ranked and not reports:
        return "Run a candidate search first, then I can answer using the current ranking and reports."
    candidate_report = _find_report_for_question(lower, reports)
    candidate_rank = _find_rank_for_question(lower, ranked)
    if candidate_report or candidate_rank:
        return _candidate_fit_answer(candidate_report, candidate_rank)
    if any(phrase in lower for phrase in ("ranked higher", "rank higher", "top ranked", "highest ranked", "who is ranked", "ranking")):
        return _ranking_answer(ranked)
    if "why" in lower and ranked:
        candidate = ranked[0]
        return (
            f"{candidate['candidate_name']} ranked highly because the search found evidence for "
            f"{', '.join(candidate.get('matched_skills', []))}. Top evidence: "
            f"{candidate.get('evidence', ['No evidence available'])[0]}"
        )
    if "compare" in lower and len(reports) >= 2:
        a, b = reports[0], reports[1]
        return (
            f"{a['candidate_name']} scored {a['transferability_score']}; "
            f"{b['candidate_name']} scored {b['transferability_score']}."
        )
    if "kubernetes" in lower:
        names = [
            candidate["candidate_name"]
            for candidate in ranked
            if "kubernetes" in " ".join(candidate.get("matched_skills", [])).casefold()
            or "ecs" in " ".join(candidate.get("matched_skills", [])).casefold()
        ]
        return "Candidates with Kubernetes or adjacent orchestration evidence: " + (", ".join(names) or "none")
    return "I can explain rankings, evidence, skill gaps, semantic matches, and candidate comparisons from the current session."


def _ranking_answer(ranked: list[dict]) -> str:
    if not ranked:
        return "There are no ranked candidates in the current session."
    rows = [
        f"#{index} {candidate.get('candidate_name', 'Unknown')} with score {float(candidate.get('score', 0.0)):.2f}"
        for index, candidate in enumerate(ranked[:5], start=1)
    ]
    return "Current ranking: " + "; ".join(rows) + "."


def _candidate_fit_answer(report: dict | None, candidate: dict | None) -> str:
    name = (report or candidate or {}).get("candidate_name", "This candidate")
    recommendation = (report or {}).get("hiring_recommendation")
    transferability = (report or {}).get("transferability_score")
    gaps = (report or {}).get("gaps", [])
    direct = (report or {}).get("direct_matches", [])
    semantic = (report or {}).get("semantic_matches", [])
    skills = (candidate or {}).get("matched_skills", [])
    evidence = (candidate or {}).get("evidence", [])

    if report:
        answer = f"{name} is a {recommendation} for this role"
        if transferability is not None:
            answer += f" with transferability score {float(transferability):.2f}"
        answer += "."
        if direct or semantic:
            matches = [
                *(match.get("skill", "") for match in direct),
                *(f"{match.get('matched_skill')} -> {match.get('jd_skill')}" for match in semantic),
            ]
            answer += " Key matches: " + ", ".join(match for match in matches if match) + "."
        if gaps:
            answer += " Gaps: " + ", ".join(gaps) + "."
        return answer

    return (
        f"{name} appears in the ranking with skills {', '.join(skills) or 'not extracted'}. "
        f"Top evidence: {(evidence or ['No evidence available'])[0]}"
    )


def _find_report_for_question(lower_question: str, reports: list[dict]) -> dict | None:
    for report in reports:
        name = str(report.get("candidate_name", "")).casefold()
        parts = [part for part in name.split() if len(part) > 2]
        if name and (name in lower_question or any(part in lower_question for part in parts)):
            return report
    return None


def _find_rank_for_question(lower_question: str, ranked: list[dict]) -> dict | None:
    for candidate in ranked:
        name = str(candidate.get("candidate_name", "")).casefold()
        parts = [part for part in name.split() if len(part) > 2]
        if name and (name in lower_question or any(part in lower_question for part in parts)):
            return candidate
    return None


def _read_upload(upload: Any) -> str:
    data = upload.getvalue()
    if upload.name.lower().endswith(".pdf"):
        try:
            return parse_pdf(data)
        except Exception as exc:
            st.error(f"Could not parse {upload.name}: {exc}")
            return ""
    return data.decode("utf-8", errors="ignore")


def _render_hitl_controls(result: dict) -> None:
    checkpoint = result.get("paused_at")
    if not checkpoint:
        return

    st.markdown("**Workflow paused for recruiter approval.**")
    if checkpoint == "jd_review":
        edited_jd = _render_jd_editor(result.get("jd_structured", {}))
        approve_col, reject_col = st.columns(2)
        if approve_col.button("Approve JD and Continue", type="primary"):
            _resume_workflow(
                result,
                checkpoint,
                {"jd_approved": True, "jd_structured": edited_jd},
            )
        if reject_col.button("Reject JD and Stop"):
            _reject_workflow(result, checkpoint, "Job description requirements need revision.")
    elif checkpoint == "shortlist_review":
        shortlist = result.get("ranked_candidates", [])[:5]
        approve_col, reject_col = st.columns(2)
        if approve_col.button("Approve Top 5 Shortlist and Continue", type="primary"):
            _resume_workflow(result, checkpoint, {"shortlist_approved": shortlist})
        if reject_col.button("Reject Shortlist and Stop"):
            _reject_workflow(result, checkpoint, "Candidate shortlist was rejected by recruiter.")
    elif checkpoint == "report_gate":
        report_ids = [report["candidate_id"] for report in result.get("reports", [])]
        approve_col, reject_col = st.columns(2)
        if approve_col.button("Approve All Reports and Continue", type="primary"):
            _resume_workflow(result, checkpoint, {"selected_reports": report_ids})
        if reject_col.button("Reject Reports and Stop"):
            _reject_workflow(result, checkpoint, "Generated reports were rejected by recruiter.")


def _resume_workflow(result: dict, checkpoint: str, updates: dict[str, Any]) -> None:
    state = {
        **result,
        **updates,
        "paused_at": "",
        "interrupts": [
            item
            for item in result.get("interrupts", [])
            if item.get("checkpoint") != checkpoint
        ],
    }
    st.session_state.result = build_graph().invoke(
        state, config={"configurable": {"thread_id": "streamlit"}}
    )
    st.rerun()


def _reject_workflow(result: dict, checkpoint: str, reason: str) -> None:
    st.session_state.result = {
        **result,
        "paused_at": "",
        "rejected_at": checkpoint,
        "rejection_reason": reason,
        "interrupts": [
            {
                **item,
                "status": "rejected" if item.get("checkpoint") == checkpoint else item.get("status"),
            }
            for item in result.get("interrupts", [])
        ],
    }
    st.rerun()


def _render_jd_editor(jd: dict) -> dict:
    with st.container(border=True):
        required = st.text_input(
            "Required skills",
            value=", ".join(jd.get("required_skills", [])),
            key="jd_required_skills",
        )
        preferred = st.text_input(
            "Preferred skills",
            value=", ".join(jd.get("preferred_skills", [])),
            key="jd_preferred_skills",
        )
        seniority = st.text_input(
            "Seniority",
            value=jd.get("seniority") or "",
            key="jd_seniority",
        )
        responsibilities = st.text_area(
            "Responsibilities",
            value="\n".join(jd.get("responsibilities", [])),
            height=120,
            key="jd_responsibilities",
        )
        implicit = st.text_area(
            "Implicit requirements",
            value="\n".join(jd.get("implicit_requirements", [])),
            height=90,
            key="jd_implicit_requirements",
        )
    return {
        "required_skills": _parse_csv(required),
        "preferred_skills": _parse_csv(preferred),
        "seniority": seniority.strip() or None,
        "responsibilities": _parse_lines(responsibilities),
        "implicit_requirements": _parse_lines(implicit),
    }


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_lines(value: str) -> list[str]:
    return [line.strip(" -*\t") for line in value.splitlines() if line.strip(" -*\t")]


def _render_filter_chips(filters: dict[str, Any]) -> None:
    labels = {
        "min_years_experience": "Minimum years",
        "require_skill": "Requires skill",
        "exclude_missing_skill": "Must include",
        "boost_candidate": "Boost",
        "note": "Note",
    }
    chips = []
    for key, value in filters.items():
        label = labels.get(key, key.replace("_", " ").title())
        chips.append(f'<span class="chip">{label}: {value}</span>')
    st.markdown("".join(chips), unsafe_allow_html=True)


def _render_match_list(matches: list[dict[str, Any]], empty_message: str) -> None:
    if not matches:
        st.caption(empty_message)
        return
    for match in matches:
        if "skill" in match:
            st.markdown(
                f"""
                <div class="match-row">
                    <strong>{match.get('skill', 'Skill')}</strong>
                    <div>{match.get('evidence', 'No evidence text available')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="match-row">
                    <strong>{match.get('matched_skill', 'Matched skill')} -> {match.get('jd_skill', 'JD skill')}</strong>
                    <span class="confidence">{match.get('confidence', 0)}</span>
                    <div>{match.get('evidence', 'No evidence text available')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_summary_metrics(result: dict) -> None:
    ranked = result.get("ranked_candidates", [])
    reports = result.get("reports", [])
    audit = result.get("taxonomy_audit", [])
    flags = result.get("faithfulness_flags", [])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates", len(ranked))
    col2.metric("Reports", len(reports))
    col3.metric("Semantic Matches", len(audit))
    col4.metric("Faithfulness Flags", len(flags))


def _render_jd_review(jd: dict) -> None:
    required = ", ".join(jd.get("required_skills", [])) or "None detected"
    preferred = ", ".join(jd.get("preferred_skills", [])) or "None detected"
    seniority = jd.get("seniority") or "Not specified"
    col1, col2, col3 = st.columns([2, 2, 1])
    col1.markdown(f"**Required**  \n{required}")
    col2.markdown(f"**Preferred**  \n{preferred}")
    col3.markdown(f"**Seniority**  \n{seniority}")
    implicit = jd.get("implicit_requirements", [])
    if implicit:
        st.caption("Implicit requirements: " + ", ".join(implicit))


def _render_ranked_candidates(candidates: list[dict]) -> None:
    if not candidates:
        st.info("No candidates matched the current query and filters.")
        return
    for rank, candidate in enumerate(candidates, start=1):
        matches = candidate.get("semantic_matches", [])
        direct_skills = ", ".join(candidate.get("matched_skills", [])) or "No extracted skills"
        evidence = candidate.get("evidence", ["No evidence found"])
        with st.container(border=True):
            left, middle, right = st.columns([2, 3, 1])
            left.markdown(f"**#{rank} {candidate.get('candidate_name', 'Unknown')}**")
            left.caption(f"Skills: {direct_skills}")
            middle.write(evidence[0])
            right.metric("Score", f"{float(candidate.get('score', 0.0)):.2f}")
            if matches:
                st.caption(
                    "Semantic matches: "
                    + "; ".join(
                        f"{m['matched_skill']} -> {m['jd_skill']} ({m['confidence']})"
                        for m in matches[:5]
                    )
                )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 4rem;
        }
        [data-testid="stSidebar"] {
            min-width: 320px;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 14px 16px;
        }
        textarea {
            min-height: 180px;
        }
        .empty-state {
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 8px;
            color: rgba(250,250,250,0.62);
            padding: 12px 14px;
            background: rgba(255,255,255,0.025);
            font-size: 0.92rem;
        }
        .chip {
            display: inline-block;
            border: 1px solid rgba(255, 75, 75, 0.42);
            background: rgba(255, 75, 75, 0.12);
            border-radius: 999px;
            padding: 5px 10px;
            margin: 0 6px 6px 0;
            font-size: 0.86rem;
        }
        .match-row {
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: rgba(255,255,255,0.025);
        }
        .confidence {
            float: right;
            color: #ff6b6b;
            font-size: 0.86rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run()
