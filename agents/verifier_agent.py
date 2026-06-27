import re


ERROR_MARKERS = (
    "[intern-s1 error]",
    "timeout",
    "connection_error",
    "dns_error",
    "auth_error",
)
GENERIC_PROOF_FINALS = {"命题得证", "得证", "证毕", "证明完毕"}
PROOF_MARKERS = ("因为", "所以", "故", "任取", "存在", "对于", "证明", "得证", "收敛", "推出")
EXPRESSION_MARKERS = ("=", "^", "frac", "sqrt", "sin", "cos", "e", "x", "y", "u", "->", "neg", "\\")
COUNT_PROBLEM_MARKERS = ("多少种", "选法", "组合", "排列")
EQUATION_PROBLEM_MARKERS = ("方程", "equation", "solve")


def _compact(text: str | None) -> str:
    return str(text or "").strip()


def _normalized_compact(text: str | None) -> str:
    return re.sub(r"[\s。.!！,，;；]", "", _compact(text).lower())


def _has_error_marker(*values: str | None) -> bool:
    joined = "\n".join(_compact(value).lower() for value in values)
    return any(marker in joined for marker in ERROR_MARKERS)


def _extract_number(text: str | None) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", _compact(text))
    return float(match.group(0)) if match else None


def _is_nonnegative_integer(text: str | None) -> bool:
    normalized = _compact(text).replace(" ", "")
    match = re.fullmatch(r"(-?\d+)(?:种(?:选法|方法)?|个(?:解)?|次)?", normalized)
    return bool(match and int(match.group(1)) >= 0)


def _add_issue(issues: list[dict], code: str, severity: str, message: str) -> None:
    issues.append({"code": code, "severity": severity, "message": message})


def _overall_severity(issues: list[dict]) -> str:
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    severity = "none"
    for issue in issues:
        if order[issue["severity"]] > order[severity]:
            severity = issue["severity"]
    return severity


def _status_from_severity(severity: str) -> str:
    if severity == "high":
        return "failed"
    if severity == "medium":
        return "uncertain"
    return "passed"


def verify_solution(
    problem: str,
    solution: str,
    final_answer: str | None,
    answer_type: str | None = None,
    domain: str | None = None,
    solver_key: str | None = None,
) -> dict:
    problem_text = _compact(problem)
    solution_text = _compact(solution)
    final_text = _compact(final_answer)
    kind = (answer_type or "unknown").lower()
    context = f"{domain or ''} {solver_key or ''} {problem_text}".lower()
    issues: list[dict] = []

    checks = {
        "has_solution": bool(solution_text),
        "has_final_answer": bool(final_text),
        "final_answer_supported": True,
        "answer_type_valid": True,
    }

    if not solution_text:
        checks["has_solution"] = False
        checks["final_answer_supported"] = False
        _add_issue(issues, "empty_solution", "high", "solution is empty")

    if not final_text:
        checks["has_final_answer"] = False
        checks["final_answer_supported"] = False
        _add_issue(issues, "empty_final_answer", "high", "final_answer is empty")

    if _has_error_marker(solution_text, final_text):
        checks["final_answer_supported"] = False
        _add_issue(issues, "error_diagnostic", "high", "solution or final_answer contains an error diagnostic")

    if kind == "number" and final_text and _extract_number(final_text) is None:
        checks["answer_type_valid"] = False
        _add_issue(issues, "number_without_digits", "medium", "number final_answer has no parseable number")

    if kind == "expression" and final_text:
        normalized_final = _normalized_compact(final_text)
        if normalized_final in {_normalized_compact(value) for value in GENERIC_PROOF_FINALS}:
            checks["answer_type_valid"] = False
            _add_issue(issues, "generic_expression_answer", "medium", "expression final_answer is generic")
        elif not any(marker in normalized_final for marker in EXPRESSION_MARKERS):
            checks["answer_type_valid"] = False
            _add_issue(issues, "expression_without_math_markers", "medium", "expression final_answer lacks math markers")

    if kind == "proof":
        normalized_final = _normalized_compact(final_text)
        if normalized_final in {_normalized_compact(value) for value in GENERIC_PROOF_FINALS}:
            _add_issue(issues, "generic_final_answer", "low", "proof final_answer is generic")
        checks["proof_markers_present"] = any(marker in solution_text for marker in PROOF_MARKERS)
        checks["has_substantive_conclusion"] = any(
            marker in solution_text for marker in ("故", "所以", "推出", "收敛", "成立")
        )
        if not checks["proof_markers_present"]:
            checks["final_answer_supported"] = False
            _add_issue(issues, "proof_markers_missing", "medium", "proof solution lacks proof markers")

    numeric_value = _extract_number(final_text)
    if "probability" in context or "概率" in context:
        checks["probability_in_range"] = numeric_value is not None and 0 <= numeric_value <= 1
        if numeric_value is not None and not checks["probability_in_range"]:
            _add_issue(issues, "probability_out_of_range", "high", "probability final_answer is outside [0, 1]")

    if any(marker in problem_text for marker in COUNT_PROBLEM_MARKERS):
        checks["count_integer_nonnegative"] = _is_nonnegative_integer(final_text)
        if not checks["count_integer_nonnegative"]:
            _add_issue(issues, "count_not_nonnegative_integer", "medium", "counting answer is not a nonnegative integer")

    if any(marker in problem_text.lower() for marker in EQUATION_PROBLEM_MARKERS):
        checks["equation_verified"] = any(marker in solution_text for marker in ("代回", "验证", "成立"))

    severity = _overall_severity(issues)
    status = _status_from_severity(severity)
    suggestion = "本地过程验证通过。"
    if status == "failed":
        suggestion = "请先修复空答案、模型错误或明显不合法的结果。"
    elif status == "uncertain":
        suggestion = "建议人工复核答案类型、表达式或过程细节。"
    elif issues:
        suggestion = "结果可接受，但建议让最终答案更具体。"

    return {
        "status": status,
        "issues": issues,
        "severity": severity,
        "suggestion": suggestion,
        "feedback": suggestion,
        "checks": checks,
    }
