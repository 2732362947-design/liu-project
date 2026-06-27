from dev_tools.run_dev_review import build_review_summary


def _result(problem_id, status, final_answer=None):
    return {
        "problem_id": problem_id,
        "model_call_status": status,
        "final_answer": final_answer,
        "verification": {},
        "attempts": [],
    }


def test_all_api_failures_do_not_count_as_math_accuracy():
    results = [_result("p1", "failed"), _result("p2", "failed"), _result("p3", "failed")]

    summary = build_review_summary(results)

    assert summary["model_failed_count"] == 3
    assert summary["api_failed_count"] == 3
    assert summary["effective_evaluated_count"] == 0
    assert summary["math_passed_count"] == 0
    assert summary["math_failed_count"] == 0
    assert summary["math_unknown_count"] == 0
    assert summary["estimated_accuracy"] is None
    assert summary["suggested_action"] == "check_network_or_api"


def test_mixed_success_and_failure_uses_only_effective_denominator():
    results = [
        _result("p1", "success", "4"),
        _result("p2", "success", "5"),
        _result("p3", "failed"),
    ]
    questions = [
        {"problem_id": "p1", "expected_answer": "4"},
        {"problem_id": "p2", "expected_answer": "6"},
        {"problem_id": "p3", "expected_answer": "7"},
    ]

    summary = build_review_summary(results, questions)

    assert summary["api_failed_count"] == 1
    assert summary["effective_evaluated_count"] == 2
    assert summary["math_passed_count"] == 1
    assert summary["math_failed_count"] == 1
    assert summary["estimated_accuracy"] == 0.5
    assert summary["suggested_action"] == "check_network_or_api"


def test_all_success_unknown_suggests_checker_work_not_api_work():
    results = [_result("p1", "success", "4"), _result("p2", "success", "5")]

    summary = build_review_summary(results)

    assert summary["model_failed_count"] == 0
    assert summary["api_failed_count"] == 0
    assert summary["effective_evaluated_count"] == 2
    assert summary["math_unknown_count"] == 2
    assert summary["estimated_accuracy"] == 0.0
    assert summary["suggested_action"] in {"improve_math_checker", "review_unknown"}
    assert summary["suggested_action"] != "check_network_or_api"


def test_all_success_and_passed_continues():
    results = [_result("p1", "success", "4"), _result("p2", "success", "5")]
    questions = [
        {"problem_id": "p1", "expected_answer": "4"},
        {"problem_id": "p2", "expected_answer": "5"},
    ]

    summary = build_review_summary(results, questions)

    assert summary["effective_evaluated_count"] == 2
    assert summary["math_passed_count"] == 2
    assert summary["estimated_accuracy"] == 1.0
    assert summary["suggested_action"] == "continue"
