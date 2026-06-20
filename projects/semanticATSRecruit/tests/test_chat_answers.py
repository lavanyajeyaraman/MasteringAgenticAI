from semanticats.ui.streamlit_app import _answer_question


def _result() -> dict:
    return {
        "ranked_candidates": [
            {
                "candidate_name": "Alex Rivera",
                "score": 3.2,
                "matched_skills": ["Selenium", "Python"],
                "evidence": ["Built Selenium automation frameworks."],
            },
            {
                "candidate_name": "Maya Chen",
                "score": 2.5,
                "matched_skills": ["CrewAI", "Python"],
                "evidence": ["Built multi-agent workflows."],
            },
        ],
        "reports": [
            {
                "candidate_name": "Alex Rivera",
                "hiring_recommendation": "Consider",
                "transferability_score": 0.67,
                "direct_matches": [{"skill": "Python", "evidence": "Python"}],
                "semantic_matches": [
                    {"matched_skill": "Selenium", "jd_skill": "Playwright"}
                ],
                "gaps": ["LangGraph"],
            }
        ],
    }


def test_chat_answers_candidate_fit_question() -> None:
    answer = _answer_question("Is Alex right fit for QA role?", _result())

    assert "Alex Rivera" in answer
    assert "Consider" in answer
    assert "Selenium -> Playwright" in answer


def test_chat_answers_ranking_question() -> None:
    answer = _answer_question("Who is ranked higher?", _result())

    assert "#1 Alex Rivera" in answer
    assert "#2 Maya Chen" in answer
