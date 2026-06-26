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


def test_extract_error_failed():
    result = extract_final_answer("任意题", "[intern-s1 error] timeout", "unknown")

    assert result["status"] == "failed"
    assert result["final_answer"] is None


def test_extract_uncertain_when_no_answer():
    result = extract_final_answer("任意题", "这是一段没有明确答案的说明。", "unknown")

    assert result["status"] == "uncertain"
    assert result["final_answer"] is None
