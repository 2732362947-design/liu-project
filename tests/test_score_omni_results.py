import json

import pytest

from dev_tools.score_omni_results import (
    build_report,
    compare_answers,
    is_solver_compatible,
    proof_format,
    score_files,
    score_item,
)


@pytest.mark.parametrize(
    ("expected", "predicted"),
    [
        ("42", "+42"),
        ("1/2", "0.5"),
        ("50%", "1/2"),
        ("x=6", "6"),
        ("x=2,3", "x=3,x=2"),
        ("8 mod 15", "x ≡ 8 (mod 15)"),
        ("A", "option A"),
        ("无解", "no solution"),
        ("1e-3", "0.001"),
        ("1,000", "1000"),
        ("(1,2)", "(1,2)"),
    ],
)
def test_supported_equivalences(expected, predicted):
    assert compare_answers(expected, predicted) is True


def test_mismatching_simple_answers_are_incorrect():
    assert compare_answers("1/2", "2/3") is False


def test_complex_expression_requires_manual_review():
    assert compare_answers("sqrt(2)", "\\sqrt{2}") is None


def test_proof_short_answer_fails_format():
    status, reasons = proof_format("命题成立。", "worked_solution")

    assert status == "format_fail"
    assert reasons


def test_proof_with_steps_and_conclusion_passes_format_but_needs_review():
    item = {
        "idx": "proof",
        "problem": "Prove the claim.",
        "expected_answer": "proof",
        "expected_domain": "algebra",
        "answer_type": "proof",
        "response_mode": "worked_solution",
    }
    result = {
        "status": "success",
        "predicted_domain": "algebra",
        "response_mode": "worked_solution",
        "final_response": (
            "First, write n=2k for an integer k.\n"
            "Then n+n=4k=2(2k), which is divisible by 2.\n"
            "Therefore the sum is even, as required."
        ),
    }

    scored = score_item(item, result)

    assert scored["format_status"] == "format_pass"
    assert scored["score_status"] == "manual_review"


def test_suspected_label_is_excluded_from_automatic_accuracy():
    dataset = [
        {
            "idx": "bad-label",
            "problem": "Compute.",
            "expected_answer": "26",
            "expected_domain": "combinatorics",
            "response_mode": "short_answer",
            "label_status": "suspected_incorrect",
            "review_note": "known conflict",
        },
        {
            "idx": "good",
            "problem": "Compute.",
            "expected_answer": "2",
            "expected_domain": "algebra",
            "response_mode": "short_answer",
            "label_status": "unreviewed",
        },
    ]
    results = [
        {
            "idx": "bad-label",
            "status": "success",
            "final_response": "26",
            "predicted_domain": "combinatorics",
            "response_mode": "short_answer",
            "api_call_count": 1,
            "elapsed_seconds": 1,
        },
        {
            "idx": "good",
            "status": "success",
            "final_response": "2",
            "predicted_domain": "algebra",
            "response_mode": "short_answer",
            "api_call_count": 1,
            "elapsed_seconds": 3,
            "local_tool_name": "elementary_algebra",
            "retry_triggered": True,
            "verification_passed": True,
        },
    ]

    report = build_report(dataset, results)

    assert report["summary"]["auto_scored"] == 1
    assert report["summary"]["auto_correct"] == 1
    assert report["summary"]["automatic_accuracy"] == 1.0
    assert report["summary"]["suspected_labels"] == 1


def test_domain_aggregation_and_behavior_statistics():
    dataset = [
        {
            "idx": "one",
            "problem": "Compute.",
            "expected_answer": "1",
            "expected_domain": "algebra",
            "response_mode": "short_answer",
        },
        {
            "idx": "two",
            "problem": "Compute.",
            "expected_answer": "2",
            "expected_domain": "algebra",
            "response_mode": "short_answer",
        },
    ]
    results = [
        {
            "idx": "one",
            "status": "success",
            "final_response": "1",
            "predicted_domain": "algebra",
            "response_mode": "short_answer",
            "local_tool_name": "elementary_algebra",
            "retry_triggered": False,
            "verification_passed": True,
            "fallback_used": False,
            "api_call_count": 0,
            "elapsed_seconds": 1,
        },
        {
            "idx": "two",
            "status": "success",
            "final_response": "3",
            "predicted_domain": "algebra",
            "response_mode": "short_answer",
            "local_tool_name": None,
            "retry_triggered": True,
            "verification_passed": False,
            "fallback_used": True,
            "api_call_count": 2,
            "elapsed_seconds": 3,
        },
    ]

    report = build_report(dataset, results)
    domain = report["by_predicted_domain"]["algebra"]
    behavior = report["behavior"]

    assert domain == {"count": 2, "auto_scored": 2, "correct": 1, "accuracy": 0.5, "manual_review": 0, "runtime_error": 0}
    assert behavior["local_tool_hit_rate"] == 0.5
    assert behavior["retry_rate"] == 0.5
    assert behavior["retry_success_rate"] == 0.0
    assert behavior["average_api_calls"] == 1.0
    assert behavior["average_elapsed_seconds"] == 2.0


def test_exact_domain_match_and_solver_compatibility_are_distinct():
    dataset = [
        {
            "idx": "cross-domain",
            "problem": "Compute a probability by counting paths.",
            "expected_answer": "1/2",
            "expected_domain": "combinatorics",
            "response_mode": "short_answer",
        }
    ]
    results = [
        {
            "idx": "cross-domain",
            "status": "success",
            "final_response": "1/2",
            "predicted_domain": "probability",
            "solver_key": "probability",
            "response_mode": "short_answer",
            "fallback_used": False,
        }
    ]

    report = build_report(dataset, results)
    item = report["items"][0]

    assert item["exact_domain_match"] is False
    assert item["solver_compatible"] is True
    assert report["summary"]["exact_routing_accuracy"] == 0.0
    assert report["summary"]["compatible_routing_rate"] == 1.0
    assert report["summary"]["unknown_rate"] == 0.0
    assert is_solver_compatible("graph_theory", "combinatorics", "discrete") is True


def test_fallback_rates_are_split_by_solver_compatibility():
    dataset = [
        {"idx": "compatible", "problem": "Count.", "expected_answer": "1", "expected_domain": "combinatorics"},
        {"idx": "incompatible", "problem": "Factor.", "expected_answer": "2", "expected_domain": "algebra"},
    ]
    results = [
        {
            "idx": "compatible",
            "status": "success",
            "final_response": "未能得到可靠答案",
            "predicted_domain": "graph_theory",
            "solver_key": "discrete",
            "fallback_used": True,
        },
        {
            "idx": "incompatible",
            "status": "success",
            "final_response": "2",
            "predicted_domain": "geometry",
            "solver_key": "geometry",
            "fallback_used": False,
        },
    ]

    summary = build_report(dataset, results)["summary"]

    assert summary["fallback_rate_when_compatible"] == 1.0
    assert summary["fallback_rate_when_incompatible"] == 0.0


def test_optional_enriched_jsonl_is_new_file_with_normalized_score_fields(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    results_path = tmp_path / "results.jsonl"
    report_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    enriched_path = tmp_path / "enriched.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "idx": "one",
                "problem": "Compute.",
                "expected_answer": "1/2",
                "expected_domain": "algebra",
                "response_mode": "short_answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    original_result = {
        "idx": "one",
        "status": "success",
        "final_response": "0.5",
        "predicted_domain": "algebra",
        "solver_key": "algebra",
    }
    original_text = json.dumps(original_result) + "\n"
    results_path.write_text(original_text, encoding="utf-8")

    score_files(dataset_path, results_path, report_path, markdown_path, enriched_path)

    enriched = json.loads(enriched_path.read_text(encoding="utf-8"))
    assert results_path.read_text(encoding="utf-8") == original_text
    assert enriched["score_status"] == "correct"
    assert enriched["score_reason"] == "normalized_answers_equal"
    assert enriched["normalized_prediction"] == "0.5"
    assert enriched["normalized_reference"] == "1/2"
