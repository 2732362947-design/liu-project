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
EXPRESSION_MARKERS = (
    "=",
    "^",
    "+",
    "-",
    "*",
    "/",
    "frac",
    "sqrt",
    "sin",
    "cos",
    "e",
    "x",
    "y",
    "u",
    "a",
    "b",
    "->",
    "neg",
    "\\",
    "≡",
    "mod",
)
COUNT_PROBLEM_MARKERS = ("多少种", "选法", "组合", "排列")
EQUATION_PROBLEM_MARKERS = ("方程", "equation", "solve")
FALLBACK_RESPONSE = "未能得到可靠答案"
INVALID_FINAL_ANSWERS = {
    ".",
    ",",
    "。",
    "?",
    "!",
    "'",
    '"',
    '".',
    "'.",
    "''",
    '""',
    "`",
    "``",
    "n/a",
    "unknown",
}
PLACEHOLDER_PHRASES = (
    "<答案>",
    "答案",
    "<answer>",
    "<result>",
    "<final_answer>",
    "<单个整数>",
    "<单个数值或数值表达式>",
    "then concise reasoning",
    "具体整数",
    "实际答案",
    "待求答案",
    "本题计算结果",
    "placeholder",
)


def _compact(text: str | None) -> str:
    return str(text or "").strip()


def _normalized_compact(text: str | None) -> str:
    return re.sub(r"[\s。.!！,，;；]", "", _compact(text).lower())


def _has_error_marker(*values: str | None) -> bool:
    joined = "\n".join(_compact(value).lower() for value in values)
    return any(marker in joined for marker in ERROR_MARKERS)


def _is_meaningful_final_answer(answer: str | None) -> bool:
    if answer is None:
        return False
    text = _compact(answer)
    if not text or text == FALLBACK_RESPONSE:
        return False
    compact = re.sub(r"\s+", "", text)
    compact_lower = compact.lower()
    text_lower = text.lower()
    if compact_lower in INVALID_FINAL_ANSWERS:
        return False
    if "then concise reasoning" in text_lower or "thenconcisereasoning" in compact_lower:
        return False
    if any(phrase.lower() in compact_lower for phrase in PLACEHOLDER_PHRASES if phrase != "答案"):
        return False
    if "<" in compact and ">" in compact and any(token in compact_lower for token in ("答案", "answer", "result", "final")):
        return False
    has_digit_or_latex_or_variable = bool(re.search(r"[0-9A-Za-z\\=^]", compact))
    if "答案" in compact and not has_digit_or_latex_or_variable:
        return False
    latex_shell = compact_lower.strip("$")
    latex_shell = latex_shell.replace(r"\(", "").replace(r"\)", "")
    latex_shell = latex_shell.replace(r"\[", "").replace(r"\]", "")
    if latex_shell in {"", "{}", r"\text{}", r"\mathrm{}"}:
        return False
    if re.fullmatch(r"[\W_]+", compact, flags=re.UNICODE):
        return False
    if re.search(r"[0-9A-Za-z\u4e00-\u9fff]", compact):
        return True
    if re.search(r"\\[A-Za-z]+", compact):
        return True
    return len(compact) > 2


def _extract_number(text: str | None) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", _compact(text))
    return float(match.group(0)) if match else None


def _number_tokens(text: str | None) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+)?", _compact(text))


def _is_single_number_answer(text: str | None) -> bool:
    normalized = _compact(text).replace(" ", "")
    if not normalized:
        return False
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:/-?\d+(?:\.\d+)?)?", normalized):
        return True
    if re.fullmatch(r"\\frac\{-?\d+\}\{-?\d+\}", normalized):
        return True
    if re.fullmatch(r"(答案是|答案为|结果是|结果为)?-?\d+(?:\.\d+)?(?:种(?:选法|方法)?|个(?:解)?|次)?", normalized):
        return True
    return False


def _is_no_solution_answer(text: str | None) -> bool:
    normalized = _normalized_compact(text)
    return normalized in {
        "无解",
        "不存在",
        "nosolution",
        "noinverseexists",
        "nomultiplicativeorderexists",
    }


def _context_allows_no_solution_answer(context: str) -> bool:
    markers = (
        "number_theory",
        "congruence",
        "同余",
        "crt",
        "chinese remainder",
        "inverse",
        "逆元",
        "order",
        "乘法阶",
        "equation",
        "方程",
    )
    return any(marker in context for marker in markers)


def _is_modular_answer(text: str | None) -> bool:
    compact = _compact(text)
    normalized = compact.replace(r"\equiv", "≡").replace(r"\pmod", " mod ")
    return bool(
        re.fullmatch(r"-?\d+\s+mod\s+\d+", normalized, flags=re.IGNORECASE)
        or re.fullmatch(r"x\s*≡\s*-?\d+\s*(?:\(?\s*mod\s+\d+\s*\)?)", normalized, flags=re.IGNORECASE)
    )


def _context_allows_modular_answer(context: str) -> bool:
    markers = (
        "number_theory",
        "congruence",
        "同余",
        "crt",
        "chinese remainder",
        "modulo",
        " mod ",
        "模 ",
        "中国剩余",
    )
    padded = f" {context} "
    return any(marker in padded for marker in markers)


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
    elif not _is_meaningful_final_answer(final_text):
        checks["final_answer_supported"] = False
        checks["answer_type_valid"] = False
        _add_issue(
            issues,
            "final_answer_not_meaningful",
            "high",
            "final_answer is punctuation or not a meaningful math answer",
        )

    if _has_error_marker(solution_text, final_text):
        checks["final_answer_supported"] = False
        _add_issue(issues, "error_diagnostic", "high", "solution or final_answer contains an error diagnostic")

    if kind == "number" and final_text:
        if _is_no_solution_answer(final_text) and _context_allows_no_solution_answer(context):
            checks["number_special_form"] = True
        elif _is_modular_answer(final_text) and _context_allows_modular_answer(context):
            checks["number_special_form"] = True
        elif _extract_number(final_text) is None:
            checks["answer_type_valid"] = False
            checks["final_answer_supported"] = False
            _add_issue(issues, "number_without_digits", "high", "number final_answer has no parseable number")
        elif not _is_single_number_answer(final_text):
            checks["answer_type_valid"] = False
            checks["final_answer_supported"] = False
            _add_issue(
                issues,
                "answer_type_mismatch",
                "high",
                "number answer_type requires a single scalar numeric answer",
            )

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
