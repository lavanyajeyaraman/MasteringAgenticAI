from semanticats.tools.skill_graph import SkillGraph


def test_skill_graph_identifies_transferable_browser_automation() -> None:
    graph = SkillGraph.default()

    match = graph.match("Playwright", "Selenium")

    assert match is not None
    assert match["relation"] == "equivalent"
    assert match["confidence"] >= 0.9


def test_transferability_score_uses_best_match_per_required_skill() -> None:
    graph = SkillGraph.default()

    score = graph.transferability_score(["LangGraph", "Kubernetes"], ["CrewAI", "ECS"])

    assert score >= 0.9
