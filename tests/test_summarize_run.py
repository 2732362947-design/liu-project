import json

from dev_tools.summarize_run import build_markdown_report, write_markdown_report


def _result(problem_id, solver_key="algebra", verification_status="passed", math_answer="2"):
    return {
        "problem_id": problem_id,
        "domain": solver_key,
        "solver_key": solver_key,
        "model_call_status": "success",
        "final_answer": math_answer,
        "solution": "解得最终答案。",
        "verification": {"status": verification_status, "severity": "none"},
        "confidence": 0.9,
        "time_cost_seconds": 1.5,
    }


def test_build_markdown_report_contains_required_sections():
    results = [_result("p1"), _result("p2", solver_key="proof")]
    review = {
        "total": 2,
        "model_failed_count": 0,
        "api_failed_count": 0,
        "effective_evaluated_count": 2,
        "math_passed_count": 2,
        "math_failed_count": 0,
        "math_unknown_count": 0,
        "estimated_accuracy": 1.0,
        "suggested_action": "continue",
        "reviews": [],
    }

    report = build_markdown_report(results, review)

    assert "Intern-S1 Math Agent Run Report" in report
    assert "Overview" in report
    assert "Solver Routing Distribution" in report
    assert "Verification Summary" in report
    assert "Sample Structured Output" in report


def test_solver_key_distribution_counts_values():
    results = [_result("p1", "algebra"), _result("p2", "proof"), _result("p3", "algebra")]

    report = build_markdown_report(results, {"reviews": []})

    assert "| algebra | 2 |" in report
    assert "| proof | 1 |" in report


def test_failed_items_include_verification_and_math_failures():
    results = [
        _result("ok"),
        _result("bad_verify", verification_status="failed"),
        _result("bad_math"),
    ]
    review = {
        "reviews": [
            {"problem_id": "bad_math", "math_check": {"status": "failed", "reason": "mismatch"}}
        ]
    }

    report = build_markdown_report(results, review)

    assert "bad_verify" in report
    assert "bad_math" in report


def test_missing_review_still_generates_report():
    report = build_markdown_report([_result("p1")], review=None)

    assert "Review summary missing" in report
    assert "| total | N/A |" in report


def test_sample_solution_is_truncated():
    long_tail = "Z" * 200
    result = _result("p1")
    result["solution"] = "A" * 350 + long_tail

    report = build_markdown_report([result], {"reviews": []})

    assert long_tail not in report


def test_write_markdown_report_creates_output_file(tmp_path):
    results_path = tmp_path / "results.json"
    review_path = tmp_path / "review.json"
    output_path = tmp_path / "nested" / "report.md"
    results_path.write_text(json.dumps([_result("p1")], ensure_ascii=False), encoding="utf-8")
    review_path.write_text(json.dumps({"reviews": []}, ensure_ascii=False), encoding="utf-8")

    report = write_markdown_report(results_path, review_path, output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == report
