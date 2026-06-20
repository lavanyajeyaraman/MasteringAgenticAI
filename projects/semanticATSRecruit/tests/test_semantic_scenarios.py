from semanticats.graph.graph import SimpleSemanticATSGraph


def _run(jd_skill: str, resume_skill: str, text: str) -> dict:
    jd = f"Senior engineer required with {jd_skill}, Python, and production delivery experience."
    state = {
        "jd_text": jd,
        "jd_approved": True,
        "candidates_raw": [{"filename": "candidate.txt", "text": text}],
        "recruiter_filters": {},
        "conversation_history": [],
    }
    return SimpleSemanticATSGraph().invoke(state)


def test_selenium_engineer_matches_playwright_jd() -> None:
    result = _run(
        "Playwright",
        "Selenium",
        "Alex Rivera\n7 years experience. Skills: Selenium, Python, API Testing. Built Selenium automation frameworks.",
    )

    report = result["reports"][0]
    assert report["transferability_score"] >= 0.9
    assert any(match["matched_skill"] == "Selenium" for match in report["semantic_matches"])
    assert result["taxonomy_audit"]


def test_crewai_engineer_matches_langgraph_jd() -> None:
    result = _run(
        "LangGraph",
        "CrewAI",
        "Maya Chen\n6 years experience. Skills: CrewAI, Python, LangChain. Built CrewAI multi-agent workflows.",
    )

    report = result["reports"][0]
    assert report["hiring_recommendation"] in {"Strong Hire", "Hire"}
    assert any(match["matched_skill"] == "CrewAI" for match in report["semantic_matches"])


def test_ecs_engineer_matches_kubernetes_jd() -> None:
    result = _run(
        "Kubernetes",
        "ECS",
        "Jordan Patel\n8 years experience. Skills: ECS, Docker, AWS. Implemented production services on ECS.",
    )

    report = result["reports"][0]
    assert any(match["matched_skill"] == "ECS" for match in report["semantic_matches"])
    assert not result["faithfulness_flags"]


def test_default_multi_skill_demo_does_not_reject_all_transferable_candidates() -> None:
    state = {
        "jd_text": (
            "Senior engineer required with Playwright, LangGraph, Kubernetes, "
            "and Python experience for production AI systems."
        ),
        "jd_approved": True,
        "candidates_raw": [
            {
                "filename": "selenium.txt",
                "text": (
                    "Alex Rivera\n7 years experience. Skills: Selenium, Python, "
                    "API Testing, Docker. Built Selenium automation frameworks."
                ),
            },
            {
                "filename": "crewai.txt",
                "text": (
                    "Maya Chen\n6 years experience. Skills: CrewAI, Python, "
                    "LangChain. Built CrewAI multi-agent workflows."
                ),
            },
            {
                "filename": "ecs.txt",
                "text": (
                    "Jordan Patel\n8 years experience. Skills: ECS, Docker, AWS. "
                    "Implemented production services on ECS."
                ),
            },
        ],
        "recruiter_filters": {},
        "conversation_history": [],
    }

    result = SimpleSemanticATSGraph().invoke(state)

    recommendations = {report["hiring_recommendation"] for report in result["reports"]}
    assert recommendations != {"Reject"}
    assert "Consider" in recommendations
