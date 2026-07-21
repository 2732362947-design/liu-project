import json

import pytest

from dev_tools.analyze_omni_smoke import analyze_files, build_analysis, validate_dataset_fields


def _score_report(status="incorrect"):
    return {
        "items": [
            {
                "idx": "one",
                "score_status": status,
                "score_reason": "normalized_answers_differ",
            }
        ]
    }


def test_analysis_reads_expected_fields_not_legacy_aliases():
    dataset = [
        {
            "idx": "one",
            "expected_domain": "algebra",
            "expected_answer": "42",
            "domain": "WRONG_DOMAIN",
            "answer": None,
        }
    ]
    results = [
        {
            "idx": "one",
            "predicted_domain": "algebra",
            "solver_key": "algebra",
            "fallback_used": False,
            "output_quality_passed": True,
            "mathematical_verification_status": "passed",
        }
    ]

    analysis = build_analysis(dataset, results, _score_report("manual_review"))

    assert analysis["expected_domain_distribution"] == {"algebra": 1}
    assert analysis["exact_confusion_matrix"] == {"algebra": {"algebra": 1}}
    assert analysis["manual_review_items"][0]["expected_answer_present"] is True


@pytest.mark.parametrize("missing_field", ["expected_domain", "expected_answer"])
def test_analysis_rejects_missing_required_dataset_fields(missing_field):
    item = {"idx": "one", "expected_domain": "algebra", "expected_answer": "42"}
    item.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        validate_dataset_fields([item])


def test_analysis_separates_failure_categories_and_behavior():
    dataset = [{"idx": "one", "expected_domain": "algebra", "expected_answer": "42"}]
    results = [
        {
            "idx": "one",
            "predicted_domain": "unknown",
            "solver_key": "general",
            "fallback_used": True,
            "output_quality_passed": False,
            "output_quality_reason": "unbalanced_latex",
            "output_quality_subreason": "unbalanced_dollar",
            "mathematical_verification_status": "failed",
            "mathematical_verification_reason": "short_answer_verification_failed",
            "retry_triggered": True,
            "retry_reason_codes": ["unbalanced_latex"],
            "first_thinking_mode_requested": True,
            "first_thinking_mode_applied": True,
            "retry_thinking_mode_requested": False,
            "retry_thinking_mode_applied": True,
        }
    ]

    analysis = build_analysis(dataset, results, _score_report())

    assert len(analysis["unknown_items"]) == 1
    assert len(analysis["fallback_items"]) == 1
    assert len(analysis["quality_failures"]) == 1
    assert len(analysis["verifier_failures"]) == 1
    assert len(analysis["incorrect_items"]) == 1
    assert analysis["retry_reason_distribution"] == {"unbalanced_latex": 1}
    assert list(analysis["thinking_mode_behavior"].values()) == [1]


def test_analyze_files_writes_json_and_markdown(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    results_path = tmp_path / "results.jsonl"
    score_path = tmp_path / "score.json"
    output_json = tmp_path / "analysis.json"
    output_md = tmp_path / "analysis.md"
    dataset_path.write_text(
        json.dumps({"idx": "one", "expected_domain": "algebra", "expected_answer": "42"}) + "\n",
        encoding="utf-8",
    )
    results_path.write_text(
        json.dumps({"idx": "one", "predicted_domain": "algebra", "solver_key": "algebra"}) + "\n",
        encoding="utf-8",
    )
    score_path.write_text(json.dumps(_score_report("manual_review")), encoding="utf-8")

    analysis = analyze_files(dataset_path, results_path, score_path, output_json, output_md)

    assert json.loads(output_json.read_text(encoding="utf-8")) == analysis
    assert "Expected domain distribution" in output_md.read_text(encoding="utf-8")
