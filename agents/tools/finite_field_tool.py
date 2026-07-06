from __future__ import annotations

import re


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def factor_int(n: int) -> dict[int, int]:
    if n <= 1:
        return {}
    factors: dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def prime_power_decomposition(q: int) -> tuple[int, int] | None:
    if q < 2:
        return None
    factors = factor_int(q)
    if len(factors) != 1:
        return None
    p, n = next(iter(factors.items()))
    return (p, n)


def divisors(n: int) -> list[int]:
    if n <= 0:
        return []
    result = []
    d = 1
    while d * d <= n:
        if n % d == 0:
            result.append(d)
            if d != n // d:
                result.append(n // d)
        d += 1
    return sorted(result)


def mobius(n: int) -> int:
    if n <= 0:
        return 0
    if n == 1:
        return 1
    factors = factor_int(n)
    for exp in factors.values():
        if exp > 1:
            return 0
    return (-1) ** len(factors)


def _count_primitive_elements(p: int, n: int) -> int:
    return sum(mobius(d) * (p ** (n // d)) for d in divisors(n))


def _count_monic_irreducible(p: int, n: int) -> int:
    if n <= 0:
        return 0
    total = sum(mobius(d) * (p ** (n // d)) for d in divisors(n))
    return total // n


def _normalize_text(problem: str) -> str:
    text = str(problem or "").lower()
    replacements = {
        r"\{": "{",
        r"\}": "}",
        r"\dots": "...",
        r"\ldots": "...",
        r"\alpha": "alpha",
        r"\beta": "beta",
        r"\gamma": "gamma",
        r"\mathbb": "",
        r"\mathrm": "",
        r"\text": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("$", " ")
    text = re.sub(r"_\{([^}]+)\}", r"_\1", text)
    text = re.sub(r"\^?\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_q_values(text: str) -> list[int]:
    q_values = []
    for match in re.finditer(r"(?:f[_\s]*|gf\(|field\s*(?:with|of)\s*)(\d+)", text):
        q = int(match.group(1))
        if q >= 2:
            q_values.append(q)
    for match in re.finditer(r"(\d+)\s*(?:元|elements|个元素)", text):
        q = int(match.group(1))
        if q >= 2:
            q_values.append(q)
    return q_values


def _extract_p_from_base_field(text: str) -> int | None:
    for match in re.finditer(r"f[_\s]*(\d+)", text):
        p_candidate = int(match.group(1))
        if is_prime(p_candidate):
            return p_candidate
    for match in re.finditer(r"gf\((\d+)\)", text):
        p_candidate = int(match.group(1))
        if is_prime(p_candidate):
            return p_candidate
    return None


def _detect_generator_extension_problem(text: str) -> bool:
    generator_markers = (
        "generate",
        "generated",
        "generator",
        "generates",
        "生成",
        "primitive element",
        "primitive elements",
    )
    extension_markers = (
        "extension",
        "扩张",
    )
    count_markers = (
        "number of elements",
        "how many elements",
        "count",
        "个数",
        "元素个数",
        "求",
        "t 中元素",
        "集合 t",
        "set t",
        "|t|",
    )
    f_q_equals_f_p_pattern = re.search(
        r"(?:f[_\s]*|gf\()\d+\s*\)?\s*=\s*(?:f[_\s]*|gf\()\d+\s*\)?\s*\(", text
    )
    f_p_alpha_pattern = re.search(
        r"(?:f[_\s]*|gf\()\d+\s*\)?\s*\(\s*(?:alpha|α|beta|β|gamma|γ|[a-z])\s*\)", text
    )
    has_generator = any(marker in text for marker in generator_markers)
    has_extension = any(marker in text for marker in extension_markers)
    has_count = any(marker in text for marker in count_markers)
    has_field_eq = bool(f_q_equals_f_p_pattern)
    has_field_alpha = bool(f_p_alpha_pattern)
    return has_count and (has_generator or has_extension or has_field_eq or has_field_alpha)


def _detect_irreducible_polynomial_problem(text: str) -> bool:
    markers = (
        "monic irreducible",
        "首一不可约",
        "irreducible polynomial",
        "不可约多项式",
    )
    degree_markers = (
        "degree",
        "次",
    )
    has_poly = any(marker in text for marker in markers)
    has_degree = any(marker in text for marker in degree_markers)
    over_field_pattern = re.search(
        r"(?:over|on)\s*(?:gf\(|f[_\s]*)\d+", text
    )
    has_field = bool(over_field_pattern)
    return has_poly and (has_degree or has_field)


def _extract_degree(text: str) -> int | None:
    for match in re.finditer(r"degree\s*(\d+)", text):
        return int(match.group(1))
    for match in re.finditer(r"(\d+)\s*次", text):
        return int(match.group(1))
    return None


def solve_finite_field_problem(problem: str) -> dict | None:
    text = _normalize_text(problem)

    q_values = _extract_q_values(text)
    p_base = _extract_p_from_base_field(text)

    is_generator_problem = _detect_generator_extension_problem(text)
    is_irreducible_problem = _detect_irreducible_polynomial_problem(text)

    if not is_generator_problem and not is_irreducible_problem:
        return None

    q = None
    p = None
    n = None

    for candidate_q in sorted(set(q_values), reverse=True):
        decomp = prime_power_decomposition(candidate_q)
        if decomp is not None:
            candidate_p, candidate_n = decomp
            if p_base is not None and candidate_p == p_base:
                q, p, n = candidate_q, candidate_p, candidate_n
                break
            elif p_base is None:
                q, p, n = candidate_q, candidate_p, candidate_n
                break

    if is_irreducible_problem and not is_generator_problem:
        degree = _extract_degree(text)
        if degree is not None and degree >= 2 and p_base is not None:
            p = p_base
            n = degree
            q = p ** n
        elif q is not None and p is not None and n is not None:
            pass
        else:
            return None
    else:
        if q is None and p_base is not None:
            degree = _extract_degree(text)
            if degree is not None and degree >= 1:
                p = p_base
                n = degree
                q = p ** n

        if p is None or n is None or q is None:
            return None

        if not is_prime(p) or n < 1:
            return None

    if is_generator_problem:
        answer = _count_primitive_elements(p, n)
        solution = (
            f"有限域 F_{q} 是 F_{p} 的 {n} 次扩张。"
            f"生成整个扩张 F_{{q}} = F_{{p}}(alpha) 的元素 alpha 的个数为 "
            f"sum_{{d|{n}}} mu(d) * {p}^({n}/d) = {answer}。"
        )
        return {
            "tool_name": "finite_field_tool",
            "final_answer": str(answer),
            "details": {
                "problem_type": "finite_field_generator_count",
                "p": p,
                "n": n,
                "q": q,
                "formula": f"sum_{{d|n}} mu(d) p^(n/d) with p={p}, n={n}",
                "answer": answer,
            },
            "solution": solution,
        }

    if is_irreducible_problem:
        answer = _count_monic_irreducible(p, n)
        solution = (
            f"F_{p} 上 {n} 次首一不可约多项式的个数为 "
            f"(1/{n}) * sum_{{d|{n}}} mu(d) * {p}^({n}/d) = {answer}。"
        )
        return {
            "tool_name": "finite_field_tool",
            "final_answer": str(answer),
            "details": {
                "problem_type": "monic_irreducible_polynomial_count",
                "p": p,
                "n": n,
                "q": q,
                "formula": f"(1/n) * sum_{{d|n}} mu(d) p^(n/d) with p={p}, n={n}",
                "answer": answer,
            },
            "solution": solution,
        }

    return None