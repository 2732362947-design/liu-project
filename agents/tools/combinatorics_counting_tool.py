from __future__ import annotations

import re


def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be nonnegative")
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result


def comb(n: int, k: int) -> int:
    if n < 0:
        raise ValueError("n must be nonnegative")
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for value in range(1, k + 1):
        result = result * (n - k + value) // value
    return result


def derangement(n: int) -> int:
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return 1
    if n == 1:
        return 0
    previous_previous = 1
    previous = 0
    for value in range(2, n + 1):
        current = (value - 1) * (previous + previous_previous)
        previous_previous, previous = previous, current
    return previous


def stirling_second_kind(n: int, k: int) -> int:
    if n < 0 or k < 0:
        raise ValueError("n and k must be nonnegative")
    if k > n:
        return 0
    table = [[0] * (k + 1) for _ in range(n + 1)]
    table[0][0] = 1
    for row in range(1, n + 1):
        upper = min(row, k)
        for col in range(1, upper + 1):
            table[row][col] = table[row - 1][col - 1] + col * table[row - 1][col]
    return table[n][k]


def surjection_count(n: int, k: int) -> int:
    if n < 0 or k < 0:
        raise ValueError("n and k must be nonnegative")
    if k > n:
        return 0
    return factorial(k) * stirling_second_kind(n, k)


def _normalize_text(problem: str) -> str:
    text = str(problem or "").lower()
    replacements = {
        r"\(": "(",
        r"\)": ")",
        r"\{": "{",
        r"\}": "}",
        r"\,": " ",
        r"\text": " ",
        r"\mathrm": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("$", " ")
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"_\{([^}]+)\}", r"_\1", text)
    text = re.sub(r"\^?\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_derangement_signal(text: str) -> bool:
    markers = (
        "derangement",
        "derangements",
        "no fixed point",
        "no fixed points",
        "without fixed points",
        "without a fixed point",
        "错排",
        "没有固定点的排列",
        "无固定点排列",
    )
    if any(marker in text for marker in markers):
        return True
    return bool(re.search(r"permutation(?:s)?\s+of\s+\d+\s+(?:elements?|objects?)\s+with\s+no\s+fixed\s+points?", text))


def _has_surjection_signal(text: str) -> bool:
    markers = (
        "onto function",
        "onto functions",
        "surjection",
        "surjections",
        "surjective function",
        "surjective functions",
        "满射",
        "映上函数",
    )
    return any(marker in text for marker in markers)


def _has_stirling_signal(text: str) -> bool:
    markers = (
        "stirling number of the second kind",
        "stirling numbers of the second kind",
        "第二类 stirling 数",
        "第二类stirling数",
        "第二类斯特林数",
        "第二类 斯特林 数",
        "非空子集",
        "nonempty subsets",
        "non-empty subsets",
    )
    return any(marker in text for marker in markers)


def _extract_single_size(text: str) -> int | None:
    patterns = (
        r"(?:derangements?\s+(?:are\s+there\s+)?of|permutations?\s+of)\s+(\d+)\s+(?:elements?|objects?)",
        r"(\d+)\s*(?:elements?|objects?)",
        r"(\d+)\s*(?:个)?(?:元素|元)(?:的)?(?:错排|排列)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _extract_pair_from_s_notation(text: str) -> tuple[int, int] | None:
    match = re.search(r"\bs\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _extract_domain_codomain_sizes(text: str) -> tuple[int, int] | None:
    patterns = (
        r"from\s+(?:an?\s+)?(\d+)-?\s*element\s+set\s+(?:on)?to\s+(?:an?\s+)?(\d+)-?\s*element\s+set",
        r"from\s+(?:a\s+set\s+with\s+)?(\d+)\s+elements?\s+(?:on)?to\s+(?:a\s+set\s+with\s+)?(\d+)\s+elements?",
        r"从\s*(\d+)\s*(?:个)?元集合\s*到\s*(\d+)\s*(?:个)?元集合",
        r"从\s*(\d+)\s*(?:个)?元素(?:的)?集合\s*到\s*(\d+)\s*(?:个)?元素(?:的)?集合",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _extract_partition_sizes(text: str) -> tuple[int, int] | None:
    s_pair = _extract_pair_from_s_notation(text)
    if s_pair is not None:
        return s_pair

    patterns = (
        r"partition\s+(\d+)\s+distinct\s+elements?\s+into\s+(\d+)\s+(?:non-?empty\s+)?subsets?",
        r"把\s*(\d+)\s*个不同元素划分为\s*(\d+)\s*个非空子集",
        r"将\s*(\d+)\s*个不同元素划分为\s*(\d+)\s*个非空子集",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def solve_combinatorics_counting_problem(problem: str) -> dict | None:
    text = _normalize_text(problem)

    if _has_derangement_signal(text):
        n = _extract_single_size(text)
        if n is None:
            return None
        answer = derangement(n)
        return {
            "tool_name": "combinatorics_counting_tool",
            "final_answer": str(answer),
            "details": {
                "problem_type": "derangement_count",
                "n": n,
                "answer": answer,
                "formula": "D(0)=1, D(1)=0, D(n)=(n-1)(D(n-1)+D(n-2))",
            },
            "solution": f"错排数满足 D(0)=1, D(1)=0, D(n)=(n-1)(D(n-1)+D(n-2))，所以 D({n})={answer}。",
        }

    if _has_surjection_signal(text):
        sizes = _extract_domain_codomain_sizes(text)
        if sizes is None:
            return None
        n, k = sizes
        answer = surjection_count(n, k)
        return {
            "tool_name": "combinatorics_counting_tool",
            "final_answer": str(answer),
            "details": {
                "problem_type": "surjection_count",
                "n": n,
                "k": k,
                "answer": answer,
                "formula": "k! * S(n,k)",
            },
            "solution": f"从 {n} 元集合到 {k} 元集合的满射数为 k! * S(n,k)，因此为 {k}! * S({n},{k}) = {answer}。",
        }

    if _has_stirling_signal(text):
        sizes = _extract_partition_sizes(text)
        if sizes is None:
            return None
        n, k = sizes
        answer = stirling_second_kind(n, k)
        return {
            "tool_name": "combinatorics_counting_tool",
            "final_answer": str(answer),
            "details": {
                "problem_type": "stirling_second_kind",
                "n": n,
                "k": k,
                "answer": answer,
                "formula": "S(n,k)=S(n-1,k-1)+k*S(n-1,k)",
            },
            "solution": f"第二类 Stirling 数满足 S(n,k)=S(n-1,k-1)+k*S(n-1,k)，所以 S({n},{k})={answer}。",
        }

    return None
