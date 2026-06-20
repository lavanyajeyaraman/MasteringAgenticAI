from semanticats.graph.nodes.verification import verify_reports


def test_verification_flags_unsupported_claims() -> None:
    state = {
        "reports": [
            {
                "candidate_id": "a",
                "candidate_name": "A",
                "direct_matches": [{"skill": "Python", "evidence": "No relevant evidence"}],
                "semantic_matches": [],
            }
        ]
    }

    result = verify_reports(state)

    assert result["faithfulness_flags"]
