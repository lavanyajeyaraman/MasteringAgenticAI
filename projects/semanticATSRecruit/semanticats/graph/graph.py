from __future__ import annotations

from semanticats.graph.nodes.display import conversation_loop, display_results
from semanticats.graph.nodes.hitl import hitl_jd_review, hitl_report_gate, hitl_shortlist
from semanticats.graph.nodes.ingest import index_candidates, ingest_jd, ingest_resumes
from semanticats.graph.nodes.reasoning import generate_reports
from semanticats.graph.nodes.search import hotl_taxonomy, search_candidates
from semanticats.graph.nodes.verification import hotl_faithfulness, verify_reports
from semanticats.state import RecruitingState


def build_graph():
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph
    except Exception:  # pragma: no cover - fallback for minimal environments
        return SimpleSemanticATSGraph()

    graph = StateGraph(RecruitingState)
    graph.add_node("ingest_jd", ingest_jd)
    graph.add_node("hitl_jd_review", hitl_jd_review)
    graph.add_node("ingest_resumes", ingest_resumes)
    graph.add_node("index_candidates", index_candidates)
    graph.add_node("search_candidates", search_candidates)
    graph.add_node("hotl_taxonomy", hotl_taxonomy)
    graph.add_node("hitl_shortlist", hitl_shortlist)
    graph.add_node("generate_reports", generate_reports)
    graph.add_node("verify_reports", verify_reports)
    graph.add_node("hotl_faithfulness", hotl_faithfulness)
    graph.add_node("hitl_report_gate", hitl_report_gate)
    graph.add_node("display_results", display_results)
    graph.add_node("conversation_loop", conversation_loop)

    def route_checkpoint(checkpoint: str):
        def route(state: RecruitingState) -> str:
            return "pause" if state.get("paused_at") == checkpoint else "continue"

        return route

    graph.add_edge(START, "ingest_jd")
    graph.add_edge("ingest_jd", "hitl_jd_review")
    graph.add_conditional_edges(
        "hitl_jd_review",
        route_checkpoint("jd_review"),
        {"pause": END, "continue": "ingest_resumes"},
    )
    graph.add_edge("ingest_resumes", "index_candidates")
    graph.add_edge("index_candidates", "search_candidates")
    graph.add_edge("search_candidates", "hotl_taxonomy")
    graph.add_edge("hotl_taxonomy", "hitl_shortlist")
    graph.add_conditional_edges(
        "hitl_shortlist",
        route_checkpoint("shortlist_review"),
        {"pause": END, "continue": "generate_reports"},
    )
    graph.add_edge("generate_reports", "verify_reports")
    graph.add_edge("verify_reports", "hotl_faithfulness")
    graph.add_edge("hotl_faithfulness", "hitl_report_gate")
    graph.add_conditional_edges(
        "hitl_report_gate",
        route_checkpoint("report_gate"),
        {"pause": END, "continue": "display_results"},
    )
    graph.add_edge("display_results", "conversation_loop")
    graph.add_edge("conversation_loop", END)
    return graph.compile(checkpointer=MemorySaver())


class SimpleSemanticATSGraph:
    def invoke(self, state: RecruitingState, config: dict | None = None) -> RecruitingState:
        for node in (
            ingest_jd,
            hitl_jd_review,
            ingest_resumes,
            index_candidates,
            search_candidates,
            hotl_taxonomy,
            hitl_shortlist,
            generate_reports,
            verify_reports,
            hotl_faithfulness,
            hitl_report_gate,
            display_results,
            conversation_loop,
        ):
            state = node(state)
            if state.get("paused_at"):
                break
        return state
