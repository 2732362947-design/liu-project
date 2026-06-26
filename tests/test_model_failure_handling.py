from runner import _confidence, _model_call_status, run_pipeline


def test_intern_s1_failure_does_not_promote_fallback(monkeypatch):
    def fake_solve(problem, domain, plan, retry_context=None):
        return "[intern-s1 error] simulated network failure"

    monkeypatch.setattr("runner.solve_problem", fake_solve)
    results = run_pipeline()

    for item in results:
        assert item["model_call_status"] == "failed"
        assert item["verification"]["status"] == "failed"
        assert item["confidence"] <= 0.2
        assert item["solution"].startswith("[intern-s1 error]")
        assert item["final_answer"] is None
        assert item["fallback_final_answer"]
        assert item["fallback_final_answer"] not in item["solution"]


def test_success_status_requires_non_error_solution():
    solution = "最终答案：x = 2, x = 3"
    assert _model_call_status(solution) == "success"
    assert not solution.startswith("[intern-s1 error]")


def test_failed_confidence_is_low():
    assert _confidence("failed", "failed", {"status": "failed"}, True) <= 0.2
