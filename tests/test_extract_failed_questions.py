import json

from dev_tools.extract_failed_questions import (
    extract_failed_questions,
    run_extract_failed_questions,
)


def test_extracts_model_verification_and_math_failures():
    questions = [
        {"problem_id": "ok", "problem": "1+1", "expected_answer": "2", "answer_type": "number"},
        {"problem_id": "model_failed", "problem": "2+2", "expected_answer": "4", "answer_type": "number"},
        {"problem_id": "verify_failed", "problem": "3+3", "expected_answer": "6", "answer_type": "number"},
        {"problem_id": "math_failed", "problem": "4+4", "expected_answer": "8", "answer_type": "number"},
    ]
    results = [
        {
            "problem_id": "ok",
            "model_call_status": "success",
            "verification": {"status": "passed"},
            "final_answer": "2",
        },
        {
            "problem_id": "model_failed",
            "model_call_status": "failed",
            "verification": {"status": "failed"},
            "final_answer": None,
        },
        {
            "problem_id": "verify_failed",
            "model_call_status": "success",
            "verification": {"status": "failed"},
            "final_answer": "6",
        },
        {
            "problem_id": "math_failed",
            "model_call_status": "success",
            "verification": {"status": "passed"},
            "final_answer": "9",
        },
    ]

    failed = extract_failed_questions(results, questions)

    assert [item["problem_id"] for item in failed] == [
        "model_failed",
        "verify_failed",
        "math_failed",
    ]


def test_embedded_math_check_failure_is_extracted():
    questions = [{"problem_id": "p1", "problem": "x", "expected_answer": "1"}]
    results = [
        {
            "problem_id": "p1",
            "model_call_status": "success",
            "verification": {"status": "passed"},
            "math_check": {"status": "failed"},
        }
    ]

    failed = extract_failed_questions(results, questions)

    assert failed == questions


def test_run_extract_failed_questions_writes_failed_question_json(tmp_path):
    results_path = tmp_path / "results.json"
    questions_path = tmp_path / "questions.json"
    output_path = tmp_path / "failed.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "problem_id": "p1",
                    "model_call_status": "success",
                    "verification": {"status": "passed"},
                    "final_answer": "1",
                },
                {
                    "problem_id": "p2",
                    "model_call_status": "failed",
                    "verification": {"status": "failed"},
                    "final_answer": None,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    questions_path.write_text(
        json.dumps(
            [
                {"problem_id": "p1", "problem": "ok", "expected_answer": "1", "answer_type": "number"},
                {"problem_id": "p2", "problem": "failed", "expected_answer": "2", "answer_type": "number"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    failed = run_extract_failed_questions(results_path, questions_path, output_path)

    assert failed == [{"problem_id": "p2", "problem": "failed", "expected_answer": "2", "answer_type": "number"}]
    assert json.loads(output_path.read_text(encoding="utf-8")) == failed
