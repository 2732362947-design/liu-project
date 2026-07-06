from __future__ import annotations

import math
import re


def solve_linear_equation(a: int, b: int, c: int) -> int | None:
    if a == 0:
        return None
    numerator = c - b
    if numerator % a != 0:
        return None
    return numerator // a


def solve_quadratic_integer_roots(a: int, b: int, c: int) -> tuple[int, ...] | None:
    if a == 0:
        root = solve_linear_equation(b, c, 0)
        return None if root is None else (root,)
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    sqrt_discriminant = math.isqrt(discriminant)
    if sqrt_discriminant * sqrt_discriminant != discriminant:
        return None
    denominator = 2 * a
    roots = []
    for numerator in (-b - sqrt_discriminant, -b + sqrt_discriminant):
        if numerator % denominator != 0:
            return None
        roots.append(numerator // denominator)
    return tuple(sorted(set(roots)))


def solve_2x2_linear_system(a1: int, b1: int, c1: int, a2: int, b2: int, c2: int) -> tuple[int, int] | None:
    determinant = a1 * b2 - a2 * b1
    if determinant == 0:
        return None
    x_numerator = c1 * b2 - c2 * b1
    y_numerator = a1 * c2 - a2 * c1
    if x_numerator % determinant != 0 or y_numerator % determinant != 0:
        return None
    return x_numerator // determinant, y_numerator // determinant


def _normalize_text(problem: str) -> str:
    text = str(problem or "").lower()
    replacements = {
        "（": "(",
        "）": ")",
        "，": ",",
        "。": ".",
        "＝": "=",
        "−": "-",
        "^2": "^2",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_solve_signal(text: str) -> bool:
    markers = (
        "solve",
        "solve for x",
        "equation",
        "解方程",
        "求 x",
        "求x",
        "方程组",
    )
    return any(marker in text for marker in markers)


def _parse_int(token: str | None, default: int = 1) -> int:
    if token in (None, "", "+"):
        return default
    if token == "-":
        return -default
    return int(token)


def _parse_linear_xy_expression(expression: str) -> tuple[int, int, int] | None:
    compact = expression.replace(" ", "")
    if not compact:
        return None
    normalized = compact.replace("-", "+-")
    if normalized.startswith("+"):
        normalized = normalized[1:]
    x_coeff = 0
    y_coeff = 0
    constant = 0
    for term in normalized.split("+"):
        if not term:
            continue
        if "x" in term:
            coefficient = term.replace("x", "")
            if term.count("x") != 1 or not re.fullmatch(r"[+-]?\d*", coefficient):
                return None
            x_coeff += _parse_int(coefficient)
        elif "y" in term:
            coefficient = term.replace("y", "")
            if term.count("y") != 1 or not re.fullmatch(r"[+-]?\d*", coefficient):
                return None
            y_coeff += _parse_int(coefficient)
        else:
            if not re.fullmatch(r"[+-]?\d+", term):
                return None
            constant += int(term)
    return x_coeff, y_coeff, constant


def _strip_to_linear_expression(text: str) -> str:
    match = re.search(r"(?<![a-z])[+-]?\s*\d*\s*[xy](?![a-z])", text)
    return text[match.start():] if match else text


def _parse_linear_equation(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"([+-]?\s*\d*\s*x\s*(?:[+-]\s*\d+)?)\s*=\s*([+-]?\d+)", text)
    if not match:
        return None
    left = _strip_to_linear_expression(match.group(1))
    right = int(match.group(2).replace(" ", ""))
    parsed = _parse_linear_xy_expression(left)
    if parsed is None:
        return None
    a, y_coeff, b = parsed
    if y_coeff != 0:
        return None
    return a, b, right


def _parse_quadratic_equation(text: str) -> tuple[int, int, int] | None:
    compact = text.replace(" ", "")
    match = re.search(r"([+-]?\d*)x\^2([+-]\d+x)?([+-]\d+)?=([+-]?\d+)", compact)
    if match:
        a = _parse_int(match.group(1))
        b = _parse_int(match.group(2)[:-1] if match.group(2) else None, default=0)
        c = int(match.group(3) or 0) - int(match.group(4))
        return a, b, c
    match = re.search(r"x\^2=([+-]?\d+)", compact)
    if match:
        return 1, 0, -int(match.group(1))
    return None


def _parse_system(text: str) -> tuple[int, int, int, int, int, int] | None:
    if "y" not in text or ("方程组" not in text and "system" not in text and "," not in text and ";" not in text):
        return None
    equations = []
    for part in re.split(r"[,;]", text):
        if "=" not in part or ("x" not in part and "y" not in part):
            continue
        left, right = part.split("=", 1)
        left = _strip_to_linear_expression(left)
        right_match = re.match(r"\s*([+-]?\d+)", right)
        if not right_match:
            continue
        parsed = _parse_linear_xy_expression(left)
        if parsed is None:
            continue
        x_coeff, y_coeff, constant = parsed
        equations.append((x_coeff, y_coeff, int(right_match.group(1)) - constant))
    if len(equations) != 2:
        return None
    return (*equations[0], *equations[1])


def _format_roots(roots: tuple[int, ...]) -> str:
    return "x=" + ",".join(str(root) for root in roots)


def solve_elementary_algebra_problem(problem: str) -> dict | None:
    text = _normalize_text(problem)
    if not _has_solve_signal(text):
        return None
    if re.search(r"\d+\.\d+", text):
        return None

    system = _parse_system(text)
    if system is not None:
        solution = solve_2x2_linear_system(*system)
        if solution is None:
            return None
        x_value, y_value = solution
        final_answer = f"x={x_value}, y={y_value}"
        return {
            "tool_name": "elementary_algebra_tool",
            "final_answer": final_answer,
            "details": {"problem_type": "linear_2x2_system", "coefficients": system, "answer": solution},
            "solution": f"用二元一次方程组消元或克拉默法则，得到 {final_answer}。",
        }

    quadratic = _parse_quadratic_equation(text)
    if quadratic is not None:
        roots = solve_quadratic_integer_roots(*quadratic)
        if roots is None:
            return None
        final_answer = _format_roots(roots)
        return {
            "tool_name": "elementary_algebra_tool",
            "final_answer": final_answer,
            "details": {"problem_type": "quadratic_integer_roots", "coefficients": quadratic, "answer": roots},
            "solution": f"计算二次方程判别式并取整数根，得到 {final_answer}。",
        }

    linear = _parse_linear_equation(text)
    if linear is not None:
        answer = solve_linear_equation(*linear)
        if answer is None:
            return None
        final_answer = f"x={answer}"
        return {
            "tool_name": "elementary_algebra_tool",
            "final_answer": final_answer,
            "details": {"problem_type": "linear_equation", "coefficients": linear, "answer": answer},
            "solution": f"将一元一次方程整理为 ax+b=c，解得 {final_answer}。",
        }

    return None
