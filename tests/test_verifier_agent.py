from agents.verifier_agent import verify_solution


def test_verify_number_solution_with_equation_check_passes():
    result = verify_solution(
        "解方程 2x+5=17",
        "移项得 2x=12，所以 x=6。代回验证：2*6+5=17，成立。",
        "x=6",
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
