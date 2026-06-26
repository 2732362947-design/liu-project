import json

from dev_tools.run_dev_review import run_review


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_review_passed_with_matching_expected_answer(tmp_path):
    results = tmp_path / "results.json"
    questions = tmp_path / "questions.json"
    output = tmp_path / "summary.json"
    _write_json(results, [{"problem_id": "p1", "final_answer": r"\frac{3}{5}", "model_call_status": "success", "verification": {}, "attempts": []}])
    _write_json(questions, [{"problem_id": "p1", "problem": "probability", "expected_answer": "3/5", "answer_type": "number"}])

    summary = run_review(results, output, questions)

    assert summary["reviews"][0]["math_check"]["status"] == "passed"
    assert summary["math_passed_count"] == 1


def test_review_failed_with_mismatched_expected_answer(tmp_path):
    results = tmp_path / "results.json"
    questions = tmp_path / "questions.json"
    output = tmp_path / "summary.json"
    _write_json(results, [{"problem_id": "p1", "final_answer": "4", "model_call_status": "success", "verification": {}, "attempts": []}])
    _write_json(questions, [{"problem_id": "p1", "problem": "algebra", "expected_answer": "5", "answer_type": "number"}])

    summary = run_review(results, output, questions)

    assert summary["reviews"][0]["math_check"]["status"] == "failed"
    assert summary["math_failed_count"] == 1


def test_review_unknown_without_expected_answer(tmp_path):
    results = tmp_path / "results.json"
    output = tmp_path / "summary.json"
    _write_json(results, [{"problem_id": "p1", "final_answer": "4", "model_call_status": "success", "verification": {}, "attempts": []}])

    summary = run_review(results, output)

    assert summary["reviews"][0]["math_check"]["status"] == "unknown"
    assert summary["math_unknown_count"] == 1
