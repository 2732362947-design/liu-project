import math
import re
from fractions import Fraction


def _numbers_from_text(text: str) -> list[float]:
    values = []
    for numerator, denominator in re.findall(r"(-?\d+)\s*/\s*(-?\d+)", text):
        if int(denominator) != 0:
            values.append(float(Fraction(int(numerator), int(denominator))))
    for value in re.findall(r"(?<![\w/])-?\d+(?:\.\d+)?(?![\w/])", text):
        values.append(float(value))
    return values


def _has_number(text: str, expected: float) -> bool:
    return any(math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-6) for value in _numbers_from_text(text))


def _expected_values(problem: str) -> tuple[list[float], str, str]:
    compact = problem.replace(" ", "")
    if "x^2-5x+6=0" in compact:
        return [2.0, 3.0], "x = 2, x = 3", "quadratic_roots"

    probability_match = re.search(r"(\d+)个红球.*?(\d+)个蓝球", problem)
    if probability_match:
        red = int(probability_match.group(1))
        blue = int(probability_match.group(2))
        total = red + blue
        if total:
            return [red / total], f"{red}/{total}", "single_draw_probability"

    derivative_match = re.search(r"f\(x\)\s*=\s*x\^2.*?x\s*=\s*(-?\d+(?:\.\d+)?)", problem)
    if derivative_match:
        point = float(derivative_match.group(1))
        return [2 * point], str(int(2 * point)) if (2 * point).is_integer() else str(2 * point), "power_rule_derivative"

    return [], "", "unknown"


def verify_solution(problem: str, solution: str, final_answer: str | None = None) -> dict:
    solution_text = solution.strip()
    lower_solution = solution_text.lower()
    answer_text = (final_answer or "").strip()
    answer_lower = answer_text.lower()
    expected_values, expected_answer, expected_type = _expected_values(problem)
    is_model_error = lower_solution.startswith("[intern-s1 error]")
    is_mock_result = lower_solution.startswith("[mock intern-s1]")
    answer_is_error = answer_lower.startswith("[intern-s1 error]") or answer_lower.startswith("[mock intern-s1]")
    math_passed = bool(expected_values) and bool(answer_text) and all(
        _has_number(answer_text, value) for value in expected_values
    )

    checks = [
        {"name": "non_empty_solution", "passed": bool(solution_text)},
        {"name": "no_intern_s1_error", "passed": not is_model_error},
        {"name": "not_mock_result", "passed": not is_mock_result},
        {"name": "final_answer_present", "passed": bool(answer_text)},
        {"name": "final_answer_not_error", "passed": bool(answer_text) and not answer_is_error},
        {"name": "final_answer_not_too_short", "passed": len(answer_text) >= 1},
        {
            "name": "basic_math_check",
            "passed": math_passed if expected_values else None,
            "type": expected_type,
            "expected": expected_answer,
        },
    ]

    if not solution_text:
        status = "failed"
        feedback = "解答为空。"
    elif is_model_error:
        status = "failed"
        feedback = "Intern-S1 接口返回错误，需要检查网络、接口地址或服务状态。"
    elif is_mock_result:
        status = "failed"
        feedback = "Intern-S1 未配置或处于 mock 模式，不能视为正式解题结果。"
    elif not answer_text:
        status = "uncertain"
        feedback = "未抽取到明确 final_answer。"
    elif answer_is_error:
        status = "failed"
        feedback = "final_answer 看起来是错误信息。"
    elif expected_values and not math_passed:
        status = "failed"
        feedback = "基础数学校验未通过，final_answer 没有匹配到预期结果。"
    else:
        status = "passed"
        feedback = "基础结构和数学校验通过。"

    return {"status": status, "checks": checks, "feedback": feedback}
