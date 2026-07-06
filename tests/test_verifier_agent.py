from agents.verifier_agent import verify_solution


def test_verify_number_solution_with_equation_check_passes():
    result = verify_solution(
        "解方程 2x+5=17",
        "移项得 2x=12，所以 x=6。代回验证：2*6+5=17，成立。",
        "6",
        answer_type="number",
        domain="algebra",
        solver_key="algebra",
    )

    assert result["status"] == "passed"
    assert result["checks"]["equation_verified"] is True


def test_verify_empty_final_answer_fails():
    result = verify_solution("求 1+1", "1+1=2", None, answer_type="number")

    assert result["status"] == "failed"
    assert result["severity"] == "high"
    assert any(issue["code"] == "empty_final_answer" for issue in result["issues"])


def test_verify_intern_s1_error_fails():
    result = verify_solution(
        "任意题",
        "[intern-s1 error] timeout",
        "[intern-s1 error] timeout",
        answer_type="number",
    )

    assert result["status"] == "failed"
    assert any(issue["code"] == "error_diagnostic" for issue in result["issues"])


def test_verify_probability_in_range_does_not_fail():
    result = verify_solution(
        "probability: 求事件概率。",
        "有利情况为 3，总情况为 10，所以概率为 0.3。",
        "0.3",
        answer_type="number",
        domain="probability",
        solver_key="probability",
    )

    assert result["status"] != "failed"
    assert result["checks"]["probability_in_range"] is True


def test_verify_probability_out_of_range_has_issue():
    result = verify_solution(
        "probability: 求事件概率。",
        "计算得到 1.5。",
        "1.5",
        answer_type="number",
        domain="probability",
        solver_key="probability",
    )

    assert result["status"] in {"failed", "uncertain"}
    assert any(issue["code"] == "probability_out_of_range" for issue in result["issues"])


def test_verify_count_answer_with_unit_passes():
    result = verify_solution(
        "从5个不同元素中任选2个，有多少种选法？",
        "组合数 C(5,2)=10，所以共有 10 种选法。",
        "10种",
        answer_type="number",
        domain="discrete_math",
        solver_key="discrete",
    )

    assert result["status"] == "passed"
    assert result["checks"]["count_integer_nonnegative"] is True


def test_verify_negative_count_has_issue():
    result = verify_solution(
        "从5个不同元素中任选2个，有多少种选法？",
        "计算得到 -1。",
        "-1",
        answer_type="number",
        domain="discrete_math",
        solver_key="discrete",
    )

    assert result["status"] in {"failed", "uncertain"}
    assert any(issue["code"] == "count_not_nonnegative_integer" for issue in result["issues"])


def test_verify_proof_generic_final_answer_is_not_failed():
    result = verify_solution(
        "证明若 a_n 收敛于 a，则其子列也收敛于 a",
        "任取 epsilon>0，因为 a_n 收敛于 a，存在 N。对于 n_k>N，所以 |a_{n_k}-a|<epsilon，故子列收敛于 a。命题得证。",
        "命题得证",
        answer_type="proof",
        domain="real_analysis",
        solver_key="proof",
    )

    assert result["status"] != "failed"
    assert result["severity"] == "low"
    assert any(issue["code"] == "generic_final_answer" for issue in result["issues"])


def test_verify_punctuation_fragment_final_answer_fails():
    result = verify_solution("求答案", "最终答案：\".", "\".", answer_type="text")

    assert result["status"] == "failed"
    assert result["severity"] == "high"
    assert any(issue["code"] == "final_answer_not_meaningful" for issue in result["issues"])


def test_verify_digit_final_answer_is_meaningful():
    result = verify_solution("1+1=?", "1+1=2，所以答案为 2。", "2", answer_type="number")

    assert result["status"] == "passed"


def test_verify_number_answer_type_rejects_equation_list():
    result = verify_solution("求一个数", "最终答案：x = 2, x = 3", "x = 2, x = 3", answer_type="number")

    assert result["status"] == "failed"
    assert result["severity"] == "high"
    assert any(issue["code"] == "answer_type_mismatch" for issue in result["issues"])


def test_verify_number_answer_type_accepts_scalar_number():
    result = verify_solution("求一个数", "最终答案：26", "26", answer_type="number")

    assert result["status"] == "passed"


def test_verify_number_answer_type_rejects_placeholder_high_severity():
    result = verify_solution(
        "求一个数",
        '最终答案：<答案>", then concise reasoning.',
        '<答案>", then concise reasoning.',
        answer_type="number",
    )

    assert result["status"] == "failed"
    assert result["severity"] == "high"
    assert any(issue["code"] in {"final_answer_not_meaningful", "number_without_digits"} for issue in result["issues"])


def test_verify_number_answer_type_accepts_latex_fraction():
    result = verify_solution("求概率", r"最终答案：\frac{1}{2}", r"\frac{1}{2}", answer_type="number")

    assert result["status"] == "passed"


def test_verify_number_answer_type_accepts_no_solution_text():
    for final_answer in ("无解", "不存在", "No solution"):
        result = verify_solution(
            "求数论问题的答案",
            f"推理完成，最终答案：{final_answer}",
            final_answer,
            answer_type="number",
            domain="number_theory",
            solver_key="number_theory",
        )

        assert result["status"] == "passed"


def test_verify_number_answer_type_rejects_no_solution_in_unrelated_context():
    result = verify_solution(
        "What is the probability of heads?",
        "最终答案：无解",
        "无解",
        answer_type="number",
        domain="probability",
        solver_key="probability",
    )

    assert result["status"] == "failed"
    assert any(issue["code"] == "number_without_digits" for issue in result["issues"])


def test_verify_number_answer_type_accepts_modular_answers():
    for final_answer in ("8 mod 15", "x ≡ 8 mod 15"):
        result = verify_solution(
            "Solve the congruences.",
            f"CRT gives 最终答案：{final_answer}",
            final_answer,
            answer_type="number",
            domain="number_theory",
            solver_key="number_theory",
        )

        assert result["status"] == "passed"


def test_verify_number_answer_type_rejects_modular_answer_in_unrelated_context():
    result = verify_solution(
        "What is the probability of heads?",
        "最终答案：8 mod 15",
        "8 mod 15",
        answer_type="number",
        domain="probability",
        solver_key="probability",
    )

    assert result["status"] == "failed"
    assert any(issue["code"] == "answer_type_mismatch" for issue in result["issues"])


def test_verify_expression_accepts_congruence_answer():
    result = verify_solution(
        "Solve the congruence.",
        "最终答案：x ≡ 8 mod 15",
        "x ≡ 8 mod 15",
        answer_type="expression",
        domain="number_theory",
        solver_key="number_theory",
    )

    assert result["status"] == "passed"


def test_verify_expression_rejects_bare_number():
    result = verify_solution("求表达式", "最终答案：2", "2", answer_type="expression")

    assert result["status"] != "passed"
    assert any(issue["code"] == "expression_without_math_markers" for issue in result["issues"])


def test_verify_expression_accepts_non_placeholder_expression():
    result = verify_solution("求表达式", "最终答案：x+1", "x+1", answer_type="expression")

    assert result["status"] == "passed"
    assert not any(issue["code"] == "answer_type_mismatch" for issue in result["issues"])


def test_verify_expression_rejects_unknown_placeholder_text():
    result = verify_solution("求表达式", "最终答案：N/A", "N/A", answer_type="expression")

    assert result["status"] == "failed"
    assert any(issue["code"] == "final_answer_not_meaningful" for issue in result["issues"])


def test_verify_fallback_response_is_not_meaningful():
    result = verify_solution("求答案", "未能得到可靠答案", "未能得到可靠答案", answer_type="number")

    assert result["status"] == "failed"
    assert any(issue["code"] == "final_answer_not_meaningful" for issue in result["issues"])


def test_verify_number_answer_type_rejects_expression_answer():
    result = verify_solution("求一个数", "最终答案：x+1", "x+1", answer_type="number")

    assert result["status"] == "failed"
    assert any(issue["code"] == "answer_type_mismatch" for issue in result["issues"])
