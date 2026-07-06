from __future__ import annotations

import re


def arithmetic_nth(a1: int, d: int, n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    return a1 + (n - 1) * d


def arithmetic_sum(a1: int, d: int, n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    return n * (2 * a1 + (n - 1) * d) // 2


def geometric_nth(a1: int, r: int, n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    return a1 * (r ** (n - 1))


def geometric_sum(a1: int, r: int, n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    if r == 1:
        return a1 * n
    return a1 * (r**n - 1) // (r - 1)


def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be nonnegative")
    previous = 0
    current = 1
    for _ in range(n):
        previous, current = current, previous + current
    return previous


def lucas(n: int) -> int:
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return 2
    previous = 2
    current = 1
    for _ in range(1, n):
        previous, current = current, previous + current
    return current


def linear_recurrence_first_order(a0: int, r: int, c: int, n: int) -> int:
    if n < 0:
        raise ValueError("n must be nonnegative")
    value = a0
    for _ in range(1, n + 1):
        value = r * value + c
    return value


def linear_recurrence_second_order(a0: int, a1: int, p: int, q: int, n: int) -> int:
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return a0
    if n == 1:
        return a1
    previous_previous = a0
    previous = a1
    for _ in range(2, n + 1):
        current = p * previous + q * previous_previous
        previous_previous, previous = previous, current
    return previous


def _normalize_text(problem: str) -> str:
    text = str(problem or "").lower()
    replacements = {
        r"\{": "{",
        r"\}": "}",
        "（": "(",
        "）": ")",
        "，": ",",
        "。": ".",
        "＝": "=",
        "项数": "n",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"_\s*\{\s*(\d+)\s*\}", r"_\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_index(text: str) -> int | None:
    patterns = (
        r"\bf[_\s]*(\d+)\b",
        r"\bl[_\s]*(\d+)\b",
        r"\ba[_\s]*(\d+)\b",
        r"(?:第|前)\s*(\d+)\s*项",
        r"(?:the\s+)?(\d+)(?:st|nd|rd|th)\s+term",
        r"\bn\s*=\s*(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _extract_named_value(text: str, names: tuple[str, ...]) -> int | None:
    joined = "|".join(re.escape(name) for name in names)
    patterns = (
        rf"(?:{joined})\s*(?:is|=|为|是)\s*(-?\d+)",
        rf"(-?\d+)\s*(?:as\s+)?(?:{joined})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _solve_fibonacci(text: str) -> dict | None:
    if "fibonacci" not in text and "斐波那契" not in text:
        return None
    n = _extract_index(text)
    if n is None:
        return None
    answer = fibonacci(n)
    return {
        "tool_name": "recurrence_sequence_tool",
        "final_answer": str(answer),
        "details": {"problem_type": "fibonacci", "n": n, "answer": answer, "formula": "F_0=0, F_1=1"},
        "solution": f"斐波那契数列 F_0=0, F_1=1, F_n=F_(n-1)+F_(n-2)，所以 F_{n}={answer}。",
    }


def _solve_lucas(text: str) -> dict | None:
    if "lucas" not in text and "卢卡斯" not in text:
        return None
    n = _extract_index(text)
    if n is None:
        return None
    answer = lucas(n)
    return {
        "tool_name": "recurrence_sequence_tool",
        "final_answer": str(answer),
        "details": {"problem_type": "lucas", "n": n, "answer": answer, "formula": "L_0=2, L_1=1"},
        "solution": f"Lucas 数列 L_0=2, L_1=1, L_n=L_(n-1)+L_(n-2)，所以 L_{n}={answer}。",
    }


def _extract_sequence_params(text: str) -> tuple[int, int, int] | None:
    a1 = _extract_named_value(text, ("first term", "initial term", "首项", "a1", "a_1"))
    step = _extract_named_value(text, ("common difference", "公差", "difference"))
    if step is None:
        step = _extract_named_value(text, ("common ratio", "公比", "ratio"))
    n = _extract_index(text)
    if a1 is None or step is None or n is None:
        return None
    return a1, step, n


def _solve_arithmetic(text: str) -> dict | None:
    if "arithmetic" not in text and "等差" not in text:
        return None
    params = _extract_sequence_params(text)
    if params is None:
        return None
    a1, d, n = params
    is_sum = any(marker in text for marker in ("sum", "前", "和", "s_"))
    answer = arithmetic_sum(a1, d, n) if is_sum else arithmetic_nth(a1, d, n)
    return {
        "tool_name": "recurrence_sequence_tool",
        "final_answer": str(answer),
        "details": {
            "problem_type": "arithmetic_sum" if is_sum else "arithmetic_nth",
            "a1": a1,
            "d": d,
            "n": n,
            "answer": answer,
        },
        "solution": f"等差数列首项 {a1}，公差 {d}，{'前 n 项和' if is_sum else '第 n 项'}为 {answer}。",
    }


def _solve_geometric(text: str) -> dict | None:
    if "geometric" not in text and "等比" not in text:
        return None
    params = _extract_sequence_params(text)
    if params is None:
        return None
    a1, r, n = params
    is_sum = any(marker in text for marker in ("sum", "前", "和", "s_"))
    answer = geometric_sum(a1, r, n) if is_sum else geometric_nth(a1, r, n)
    return {
        "tool_name": "recurrence_sequence_tool",
        "final_answer": str(answer),
        "details": {
            "problem_type": "geometric_sum" if is_sum else "geometric_nth",
            "a1": a1,
            "r": r,
            "n": n,
            "answer": answer,
        },
        "solution": f"等比数列首项 {a1}，公比 {r}，{'前 n 项和' if is_sum else '第 n 项'}为 {answer}。",
    }


def _solve_first_order_recurrence(text: str) -> dict | None:
    if "递推" not in text and "recurrence" not in text and "a_n" not in text:
        return None
    match = re.search(
        r"a_0\s*=\s*(-?\d+).*?a_n\s*=\s*(-?\d+)?\s*\*?\s*a_\{?n-1\}?\s*([+-]\s*\d+)?",
        text,
    )
    target = re.search(r"(?:求|find)\s*a[_\s]*(\d+)", text)
    if not match or not target:
        return None
    a0 = int(match.group(1))
    r = int(match.group(2) or 1)
    c = int((match.group(3) or "+0").replace(" ", ""))
    n = int(target.group(1))
    answer = linear_recurrence_first_order(a0, r, c, n)
    return {
        "tool_name": "recurrence_sequence_tool",
        "final_answer": str(answer),
        "details": {"problem_type": "first_order_recurrence", "a0": a0, "r": r, "c": c, "n": n, "answer": answer},
        "solution": f"从 a_0={a0} 开始迭代 a_n={r}a_(n-1)+{c}，得到 a_{n}={answer}。",
    }


def _solve_second_order_recurrence(text: str) -> dict | None:
    if "递推" not in text and "recurrence" not in text and "a_n" not in text:
        return None
    match = re.search(
        r"a_0\s*=\s*(-?\d+).*?a_1\s*=\s*(-?\d+).*?a_n\s*=\s*(-?\d+)?\s*\*?\s*a_\{?n-1\}?\s*\+\s*(-?\d+)?\s*\*?\s*a_\{?n-2\}?",
        text,
    )
    target = re.search(r"(?:求|find)\s*a[_\s]*(\d+)", text)
    if not match or not target:
        return None
    a0 = int(match.group(1))
    a1 = int(match.group(2))
    p = int(match.group(3) or 1)
    q = int(match.group(4) or 1)
    n = int(target.group(1))
    answer = linear_recurrence_second_order(a0, a1, p, q, n)
    return {
        "tool_name": "recurrence_sequence_tool",
        "final_answer": str(answer),
        "details": {
            "problem_type": "second_order_recurrence",
            "a0": a0,
            "a1": a1,
            "p": p,
            "q": q,
            "n": n,
            "answer": answer,
        },
        "solution": f"从 a_0={a0}, a_1={a1} 开始迭代 a_n={p}a_(n-1)+{q}a_(n-2)，得到 a_{n}={answer}。",
    }


def solve_recurrence_sequence_problem(problem: str) -> dict | None:
    text = _normalize_text(problem)
    for solver in (
        _solve_fibonacci,
        _solve_lucas,
        _solve_arithmetic,
        _solve_geometric,
        _solve_second_order_recurrence,
        _solve_first_order_recurrence,
    ):
        result = solver(text)
        if result is not None:
            return result
    return None
