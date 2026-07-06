from agents.answer_extractor_agent import extract_final_answer


def test_extract_quadratic_roots():
    result = extract_final_answer(
        "求方程 x^2 - 5x + 6 = 0 的两个实数根。",
        "解得 x=2, x=3。\n最终答案：x = 2, x = 3",
        "algebra",
    )

    assert result["status"] == "passed"
    assert "2" in result["final_answer"]
    assert "3" in result["final_answer"]


def test_extract_probability():
    result = extract_final_answer(
        "一个袋子里有3个红球和2个蓝球，随机取出1个球，取到红球的概率是多少？",
        "有利情况为3，总情况为5，所以概率为 3/5。",
        "probability",
    )

    assert result["status"] == "passed"
    assert result["final_answer"] in {"3/5", "0.6"} or "3/5" in result["final_answer"]


def test_extract_expression_answer_types():
    for answer in ("x+1", "a+b", r"\frac{x}{2}", "sqrt(x+1)", "2x+3"):
        result = extract_final_answer("求表达式。", f"推理过程。\n最终答案：{answer}", "algebra")

        assert result["status"] == "passed"
        assert result["final_answer"] == answer
        assert result["answer_type"] == "expression"


def test_extract_plain_number_answer_type():
    result = extract_final_answer("求数值。", "推理过程。\n最终答案：2", "algebra")

    assert result["status"] == "passed"
    assert result["final_answer"] == "2"
    assert result["answer_type"] == "number"


def test_extract_common_real_model_answer_formats():
    cases = [
        ("最终答案：\\boxed{42}", "42"),
        ("答案为 42。", "42"),
        ("Therefore, the final answer is 42.", "42"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        ("最终答案是 x+1", "x+1"),
        ("The answer is no solution.", "no solution"),
        ("不存在", "不存在"),
        ("无解", "无解"),
        ("8 mod 15", "8 mod 15"),
        (r"x \equiv 8 \pmod{15}", "x ≡ 8 mod 15"),
    ]

    for solution, expected in cases:
        result = extract_final_answer("求答案。", solution, "number_theory")

        assert result["status"] == "passed"
        assert result["final_answer"] == expected


def test_extract_error_failed():
    result = extract_final_answer("任意题", "[intern-s1 error] timeout", "unknown")

    assert result["status"] == "failed"
    assert result["final_answer"] is None


def test_extract_uncertain_when_no_answer():
    result = extract_final_answer("任意题", "这是一段没有明确答案的说明。", "unknown")

    assert result["status"] == "uncertain"
    assert result["final_answer"] is None


def test_extract_proof_conclusion_skips_generic_ending():
    result = extract_final_answer(
        "证明：若数列 a_n 收敛于 a，则任一子列也收敛于 a。",
        "根据收敛定义，对任意 epsilon 存在 N。故子列 {a_{n_k}} 收敛于 a。命题得证。",
        "real_analysis",
    )

    assert result["status"] == "passed"
    assert "子列" in result["final_answer"]
    assert "收敛" in result["final_answer"]
    assert result["final_answer"] != "命题得证"
