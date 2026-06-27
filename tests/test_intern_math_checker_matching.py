from dev_tools.intern_math_checker import check_math_result


def test_checker_passes_assignment_number_match():
    result = check_math_result(
        {
            "final_answer": "x = 6",
            "expected_answer": "6",
            "expected_answer_type": "number",
        }
    )

    assert result["status"] == "passed"
    assert result["method"] == "reference_answer"


def test_checker_reports_normalized_values_on_failure():
    result = check_math_result(
        {
            "final_answer": "x^2",
            "expected_answer": "x^2+C",
            "expected_answer_type": "expression",
        }
    )

    assert result["status"] == "failed"
    assert result["normalized_final"] == "x^2"
    assert result["normalized_expected"] == "x^2+c"
