from __future__ import annotations

import re


def gcd(a: int, b: int) -> int:
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    if old_r < 0:
        return -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def mod_inverse(a: int, m: int) -> int | None:
    if m <= 0:
        raise ValueError("modulus must be positive")
    g, x, _y = extended_gcd(a, m)
    if g != 1:
        return None
    return x % m


def factor_int(n: int) -> dict[int, int]:
    if n < 0:
        n = abs(n)
    if n <= 1:
        return {}
    factors: dict[int, int] = {}
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2
    divisor = 3
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            n //= divisor
        divisor += 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def euler_phi(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    result = n
    for prime in factor_int(n):
        result = result // prime * (prime - 1)
    return result


def crt_pair(a1: int, m1: int, a2: int, m2: int) -> tuple[int, int] | None:
    if m1 <= 0 or m2 <= 0:
        raise ValueError("moduli must be positive")
    g, x, _y = extended_gcd(m1, m2)
    difference = a2 - a1
    if difference % g != 0:
        return None
    lcm = m1 // g * m2
    step = (difference // g * x) % (m2 // g)
    solution = (a1 + m1 * step) % lcm
    return solution, lcm


def crt(congruences: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not congruences:
        return None
    current_a, current_m = congruences[0]
    if current_m <= 0:
        raise ValueError("moduli must be positive")
    current_a %= current_m
    for next_a, next_m in congruences[1:]:
        merged = crt_pair(current_a, current_m, next_a, next_m)
        if merged is None:
            return None
        current_a, current_m = merged
    return current_a, current_m


def multiplicative_order(a: int, n: int) -> int | None:
    if n <= 0:
        raise ValueError("modulus must be positive")
    if gcd(a, n) != 1:
        return None
    order = euler_phi(n)
    for prime, exponent in factor_int(order).items():
        for _ in range(exponent):
            candidate = order // prime
            if pow(a, candidate, n) != 1:
                break
            order = candidate
    return order


def _normalize_text(problem: str) -> str:
    text = str(problem or "").lower()
    replacements = {
        r"\(": "(",
        r"\)": ")",
        r"\{": "{",
        r"\}": "}",
        r"\,": " ",
        r"\equiv": "≡",
        r"\pmod": " mod ",
        r"\mod": " mod ",
        r"\phi": "phi",
        "φ": "phi",
        "−": "-",
        "（": "(",
        "）": ")",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("$", " ")
    text = re.sub(r"\^\s*\{\s*(-?\d+)\s*\}", r"^\1", text)
    text = re.sub(r"_\s*\{\s*(-?\d+)\s*\}", r"_\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_phi_signal(text: str) -> bool:
    markers = (
        "euler phi",
        "euler's totient",
        "eulers totient",
        "totient function",
        "欧拉函数",
        "欧拉 phi 函数",
        "欧拉phi函数",
    )
    return any(marker in text for marker in markers) or bool(re.search(r"\bphi\s*\(\s*\d+\s*\)", text))


def _extract_phi_n(text: str) -> int | None:
    patterns = (
        r"\bphi\s*\(\s*(\d+)\s*\)",
        r"(?:euler(?:'s)?\s+totient(?:\s+function)?|totient\s+function)\s+(?:of\s+)?(\d+)",
        r"欧拉(?:\s*phi\s*)?函数\s*(?:phi\s*)?\(?\s*(\d+)\s*\)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _has_mod_inverse_signal(text: str) -> bool:
    markers = (
        "modular inverse",
        "inverse of",
        "模逆元",
        "乘法逆元",
        "逆元",
    )
    return any(marker in text for marker in markers) or bool(re.search(r"\d+\s*\^\s*-1\s*(?:mod|模)\s*\d+", text))


def _extract_mod_inverse_args(text: str) -> tuple[int, int] | None:
    patterns = (
        r"modular inverse of\s+(-?\d+)\s+modulo\s+(\d+)",
        r"inverse of\s+(-?\d+)\s+mod(?:ulo)?\s+(\d+)",
        r"(-?\d+)\s*\^\s*-1\s*mod\s*(\d+)",
        r"求\s*(-?\d+)\s*在模\s*(\d+)\s*下的(?:乘法)?逆元",
        r"(-?\d+)\s*模\s*(\d+)\s*的(?:乘法)?逆元",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _extract_congruences(text: str) -> list[tuple[int, int]]:
    congruences: list[tuple[int, int]] = []
    patterns = (
        r"\bx\s*(?:≡|=)\s*(-?\d+)\s*\(\s*mod\s*(\d+)\s*\)",
        r"\bx\s*(?:≡|=)\s*(-?\d+)\s*mod\s*(\d+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            congruence = (int(match.group(1)), int(match.group(2)))
            if congruence not in congruences:
                congruences.append(congruence)
    return congruences


def _has_crt_signal(text: str, congruences: list[tuple[int, int]]) -> bool:
    markers = (
        "chinese remainder theorem",
        "crt",
        "solve the congruences",
        "system of congruences",
        "中国剩余定理",
        "解同余方程组",
        "同余方程组",
    )
    return len(congruences) >= 2 and (any(marker in text for marker in markers) or "≡" in text)


def _has_order_signal(text: str) -> bool:
    strong_markers = (
        "multiplicative order",
        "乘法阶",
        "最小正整数 k",
    )
    if any(marker in text for marker in strong_markers):
        return True
    if re.search(r"\bord_\d+\s*\(\s*-?\d+\s*\)", text):
        return True
    return bool(re.search(r"\border of\s+-?\d+\s+mod(?:ulo)?\s+\d+", text))


def _extract_order_args(text: str) -> tuple[int, int] | None:
    patterns = (
        r"multiplicative order of\s+(-?\d+)\s+modulo\s+(\d+)",
        r"order of\s+(-?\d+)\s+mod(?:ulo)?\s+(\d+)",
        r"\bord_(\d+)\s*\(\s*(-?\d+)\s*\)",
        r"求\s*(-?\d+)\s*模\s*(\d+)\s*的乘法阶",
        r"(-?\d+)\s*模\s*(\d+)\s*的阶",
        r"最小正整数 k 使\s*(-?\d+)\s*\^\s*k\s*≡\s*1\s*mod\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            first = int(match.group(1))
            second = int(match.group(2))
            if pattern.startswith(r"\bord_"):
                return second, first
            return first, second
    return None


def solve_number_theory_problem(problem: str) -> dict | None:
    text = _normalize_text(problem)

    if _has_phi_signal(text):
        n = _extract_phi_n(text)
        if n is None:
            return None
        answer = euler_phi(n)
        return {
            "tool_name": "number_theory_tool",
            "final_answer": str(answer),
            "details": {
                "problem_type": "euler_phi",
                "n": n,
                "answer": answer,
                "formula": "phi(n)=n product_{p|n}(1-1/p)",
            },
            "solution": f"Euler phi 函数满足 phi(n)=n product_{{p|n}}(1-1/p)，所以 phi({n})={answer}。",
        }

    if _has_mod_inverse_signal(text):
        args = _extract_mod_inverse_args(text)
        if args is None:
            return None
        a, m = args
        answer = mod_inverse(a, m)
        final_answer = str(answer) if answer is not None else "不存在"
        return {
            "tool_name": "number_theory_tool",
            "final_answer": final_answer,
            "details": {
                "problem_type": "modular_inverse",
                "a": a,
                "m": m,
                "answer": answer,
                "formula": "extended Euclidean algorithm",
            },
            "solution": f"用扩展欧几里得算法求 {a} 在模 {m} 下的逆元，结果为 {final_answer}。",
        }

    congruences = _extract_congruences(text)
    if _has_crt_signal(text, congruences):
        answer = crt(congruences)
        final_answer = f"{answer[0]} mod {answer[1]}" if answer is not None else "无解"
        return {
            "tool_name": "number_theory_tool",
            "final_answer": final_answer,
            "details": {
                "problem_type": "chinese_remainder_theorem",
                "congruences": congruences,
                "answer": answer,
                "formula": "successive CRT merge",
            },
            "solution": f"逐步合并同余式 {congruences}，得到 {final_answer}。",
        }

    if _has_order_signal(text):
        args = _extract_order_args(text)
        if args is None:
            return None
        a, n = args
        answer = multiplicative_order(a, n)
        final_answer = str(answer) if answer is not None else "不存在"
        return {
            "tool_name": "number_theory_tool",
            "final_answer": final_answer,
            "details": {
                "problem_type": "multiplicative_order",
                "a": a,
                "n": n,
                "answer": answer,
                "formula": "least k>0 such that a^k == 1 mod n",
            },
            "solution": f"寻找最小正整数 k 使 {a}^k ≡ 1 (mod {n})，结果为 {final_answer}。",
        }

    return None
