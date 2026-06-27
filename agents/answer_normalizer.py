import re


GENERIC_PROOF_ANSWERS = {"命题得证", "得证", "证毕", "证明完毕"}


def _normalize_fraction(text: str) -> str:
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    return text


def _normalize_partials(text: str) -> str:
    text = re.sub(
        r"\\frac\{\\partial\s+u\}\{\\partial\s+t\}",
        "u_t",
        text,
    )
    text = re.sub(
        r"\\frac\{\\partial\^2\s+u\}\{\\partial\s+x\^2\}",
        "u_xx",
        text,
    )
    text = re.sub(r"∂\s*u\s*/\s*∂\s*t", "u_t", text)
    text = re.sub(r"∂²\s*u\s*/\s*∂\s*x²", "u_xx", text)
    return text


def _extract_equation_values(text: str) -> str | None:
    values = re.findall(r"(?:x(?:_\d+)?|x_\{?\d+\}?)=(-?\d+(?:\.\d+)?)", text)
    if not values:
        return None
    ordered = sorted(set(values), key=lambda value: float(value))
    return "{" + ",".join(ordered) + "}"


def _normalize_set(text: str) -> str:
    if not (text.startswith("{") and text.endswith("}")):
        return text
    values = [value for value in text.strip("{}").split(",") if value]
    if not values:
        return "{}"
    try:
        values = sorted(set(values), key=lambda value: float(value))
    except ValueError:
        values = sorted(set(values))
    return "{" + ",".join(values) + "}"


def _strip_latex_wrappers(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        text = text[1:-1]
    for left, right in (("\\(", "\\)"), ("\\[", "\\]")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left) : -len(right)]
    return text.strip()


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""

    text = _strip_latex_wrappers(str(answer))
    text = re.sub(r"\s+", " ", text.strip()).lower()
    text = re.sub(r"[。.]$", "", text)
    text = _normalize_partials(text)
    text = _normalize_fraction(text)
    replacements = {
        "$": "",
        "\\(": "",
        "\\)": "",
        "\\[": "",
        "\\]": "",
        "。": "",
        "，": ",",
        "、": ",",
        "；": ";",
        "：": ":",
        "π": "pi",
        "\\pi": "pi",
        "α": "alpha",
        "\\alpha": "alpha",
        "κ": "kappa",
        "\\kappa": "kappa",
        "\\rightarrow": "->",
        "\\Rightarrow": "->",
        "\\to": "->",
        "\\neg": "not",
        "−": "-",
        "→": "->",
        "⇒": "->",
        " ": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"最终答案[:：]?", "", text)
    text = re.sub(r"答案[:：]?", "", text)
    text = re.sub(r"^是[:：]?", "", text)
    text = text.replace("非", "not")
    text = text.replace("¬", "not")
    text = text.replace("是一个", "是")
    text = text.replace("是的", "是")
    text = text.replace("常数", "c")

    equation_values = _extract_equation_values(text)
    if equation_values:
        return equation_values

    text = re.sub(r"\bx_\{?(\d+)\}?", r"x_\1", text)
    text = re.sub(r"[\[\]()]", "", text)
    text = _normalize_set(text)
    return text


def _extract_numeric_answer(text: str) -> str:
    text = normalize_answer(text)
    single_set_match = re.fullmatch(r"\{(-?\d+(?:\.\d+)?)\}", text)
    if single_set_match:
        return single_set_match.group(1)
    counted_number_match = re.fullmatch(
        r"(-?\d+(?:\.\d+)?)(?:种(?:选法|方法)?|个(?:解)?|次)",
        text,
    )
    if counted_number_match:
        return counted_number_match.group(1)
    match = re.fullmatch(r"[a-z](?:_\d+)?=(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    prefix_match = re.search(r"(?:答案是|答案为|为)(-?\d+(?:\.\d+)?)$", text)
    if prefix_match:
        return prefix_match.group(1)
    return text


def _strip_simple_left_side(text: str) -> str:
    if "=" not in text:
        return text
    left, right = text.split("=", 1)
    if re.fullmatch(r"[a-z](?:_\d+)?", left) and right:
        return right
    return text


def _is_heat_equation_context(*parts: str | None) -> bool:
    text = " ".join(str(part or "") for part in parts).lower()
    normalized = normalize_answer(text)
    return any(
        keyword in text or keyword in normalized
        for keyword in (
            "heat",
            "热方程",
            "pde",
            "partial",
            "偏导",
            "u_t",
            "u_xx",
        )
    )


def _normalize_heat_equation_coefficients(text: str) -> str:
    return re.sub(r"(?<=u_t=)(?:alpha|kappa|k)(?=u_xx)", "coef", text)


def _normalize_expression(
    answer: str | None,
    problem: str | None = None,
    expected_answer: str | None = None,
    final_answer: str | None = None,
) -> str:
    text = normalize_answer(answer)
    text = _strip_simple_left_side(text)
    text = text.replace("*", "")
    text = re.sub(r"\{|\}", "", text)
    if _is_heat_equation_context(problem, expected_answer, final_answer, answer):
        text = _normalize_heat_equation_coefficients(text)
    return text


def _keywords_match(expected: str, actual: str) -> bool:
    expected_keywords = [part for part in re.split(r"也|是|的|于|,|;|->", expected) if part]
    return bool(expected_keywords) and all(part in actual for part in expected_keywords)


def _proof_expected_in_solution(expected: str, solution: str | None) -> bool:
    normalized_solution = normalize_answer(solution)
    if expected and expected in normalized_solution:
        return True
    if expected == "子列也收敛于a":
        return "子列" in normalized_solution and "收敛于a" in normalized_solution
    return _keywords_match(expected, normalized_solution)


def answers_match(
    final_answer: str | None,
    expected_answer: str | None,
    answer_type: str | None = None,
    problem: str | None = None,
    solution: str | None = None,
) -> tuple[bool, str]:
    normalized_final = normalize_answer(final_answer)
    normalized_expected = normalize_answer(expected_answer)
    kind = (answer_type or "").lower()

    if not normalized_expected:
        return False, "expected_answer is empty"
    if not normalized_final and not (kind == "proof" and solution):
        return False, "final_answer is empty"

    if kind == "number":
        if _extract_numeric_answer(final_answer) == _extract_numeric_answer(expected_answer):
            return True, "normalized numeric final_answer matches expected_answer"
        return False, "normalized numeric final_answer does not match expected_answer"

    if kind == "expression":
        normalized_final_expression = _normalize_expression(
            final_answer,
            problem=problem,
            expected_answer=expected_answer,
            final_answer=final_answer,
        )
        normalized_expected_expression = _normalize_expression(
            expected_answer,
            problem=problem,
            expected_answer=expected_answer,
            final_answer=final_answer,
        )
        if normalized_final_expression == normalized_expected_expression:
            return True, "normalized expression final_answer matches expected_answer"
        return False, "normalized expression final_answer does not match expected_answer"

    if kind == "text":
        normalized_problem = normalize_answer(problem)
        if normalized_final == normalized_expected or normalized_expected in normalized_final:
            return True, "normalized final_answer contains expected_answer"
        if normalized_final in normalized_expected and normalized_final:
            remainder = normalized_expected.replace(normalized_final, "", 1)
            if not remainder or remainder in normalized_problem:
                return True, "normalized final_answer matches expected_answer using problem context"
        if _keywords_match(normalized_expected, normalized_final):
            return True, "normalized final_answer contains expected keywords"
        return False, "normalized text final_answer does not match expected_answer"

    if kind == "proof":
        if normalized_final not in {normalize_answer(value) for value in GENERIC_PROOF_ANSWERS}:
            if normalized_expected in normalized_final or _keywords_match(normalized_expected, normalized_final):
                return True, "normalized proof final_answer matches expected_answer"
        if _proof_expected_in_solution(normalized_expected, solution):
            return True, "solution contains expected proof conclusion"
        return False, "proof final_answer and solution do not match expected_answer"

    if normalized_final == normalized_expected:
        return True, "normalized final_answer matches expected_answer"
    return False, "normalized final_answer does not match expected_answer"
