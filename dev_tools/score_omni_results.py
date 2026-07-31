"""Conservatively score Omni-MATH run results without symbolic algebra."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.classifier_agent import solver_key_for_domain


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PROOF_MARKERS = ("prove", "proof", "show that", "derive", "explain", "证明", "推导", "解释")
MAX_ANSWER_CHARS = 4096
MAX_NORMALIZATION_DEPTH = 12
MAX_COLLECTION_ITEMS = 64
MAX_CLOSED_EXPR_CHARS = 1024
MAX_CLOSED_EXPR_TOKENS = 256
MAX_CLOSED_EXPR_NODES = 256
MAX_CLOSED_EXPR_DEPTH = 32
MAX_CLOSED_EXPR_INTEGER_DIGITS = 128
MAX_CLOSED_EXPR_EXPONENT = 4096
MAX_CLOSED_EXPR_FACTORIAL = 512
MAX_CLOSED_EXPR_RESULT_BITS = 16384
MAX_CLOSED_EXPR_RADICAND_BITS = 32
MAX_CLOSED_EXPR_TERMS = 64
MAX_SIMPLE_NUMBER_DIGITS = 256
MAX_SIMPLE_NUMBER_EXPONENT = 4096
NO_SOLUTION = {
    "no solution",
    "no solutions",
    "there is no solution",
    "there are no solutions",
    "does not exist",
    "nonexistent",
    "impossible",
    "无解",
    "不存在",
    "没有解",
}
NO_SOLUTION_QUALIFIERS = {
    "real": "real",
    "real number": "real",
    "real numbers": "real",
    "integer": "integer",
    "integers": "integer",
    "integral": "integer",
    "positive integer": "positive_integer",
    "positive integers": "positive_integer",
    "nonnegative integer": "nonnegative_integer",
    "nonnegative integers": "nonnegative_integer",
    "negative integer": "negative_integer",
    "negative integers": "negative_integer",
    "rational": "rational",
    "rational number": "rational",
    "rational numbers": "rational",
    "complex": "complex",
    "complex number": "complex",
    "complex numbers": "complex",
    "natural number": "natural_number",
    "natural numbers": "natural_number",
}
COMPATIBLE_SOLVER_FAMILIES = {
    "algebra": {"algebra", "elementary_algebra", "optimization", "linear_regression", "discrete"},
    "number_theory": {"number_theory", "discrete"},
    "combinatorics": {"combinatorics", "discrete", "probability", "graph_theory", "number_theory"},
    "graph_theory": {"graph_theory", "discrete", "combinatorics"},
    "probability": {"probability", "statistics", "stochastic_processes", "combinatorics", "discrete"},
    "geometry": {"geometry", "differential_geometry"},
    "calculus": {"calculus", "real_analysis", "mathematical_analysis", "numerical_analysis"},
    "optimization": {"optimization", "algebra", "linear_regression"},
    "linear_algebra": {"linear_algebra", "linear_regression"},
    "discrete_math": {"discrete", "combinatorics", "graph_theory", "number_theory"},
}


class _ClosedExpressionReject(ValueError):
    """A bounded closed expression was not safely parseable/evaluable."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _ClosedToken:
    kind: str
    value: str
    offset: int


@dataclass(frozen=True)
class _ClosedNode:
    kind: str
    value: Any = None
    left: "_ClosedNode | None" = None
    right: "_ClosedNode | None" = None


@dataclass(frozen=True)
class _ClosedExactValue:
    """Finite sums of rational multiples of square roots of squarefree integers."""

    terms: tuple[tuple[int, Fraction], ...]

    @property
    def rational(self) -> Fraction | None:
        if not self.terms:
            return Fraction(0)
        if len(self.terms) == 1 and self.terms[0][0] == 1:
            return self.terms[0][1]
        return None


def _bounded_fraction(value: Fraction) -> None:
    if (
        abs(value.numerator).bit_length() > MAX_CLOSED_EXPR_RESULT_BITS
        or value.denominator.bit_length() > MAX_CLOSED_EXPR_RESULT_BITS
    ):
        raise _ClosedExpressionReject("result_bit_length_limit")


def _closed_value(terms: dict[int, Fraction]) -> _ClosedExactValue:
    cleaned: list[tuple[int, Fraction]] = []
    for radicand, coefficient in terms.items():
        if not coefficient:
            continue
        if radicand < 1 or radicand.bit_length() > MAX_CLOSED_EXPR_RADICAND_BITS:
            raise _ClosedExpressionReject("radicand_limit")
        _bounded_fraction(coefficient)
        cleaned.append((radicand, coefficient))
    if len(cleaned) > MAX_CLOSED_EXPR_TERMS:
        raise _ClosedExpressionReject("algebraic_term_limit")
    return _ClosedExactValue(tuple(sorted(cleaned)))


def _closed_rational(value: int | Fraction) -> _ClosedExactValue:
    fraction = Fraction(value)
    return _closed_value({1: fraction} if fraction else {})


def _closed_add(
    left: _ClosedExactValue,
    right: _ClosedExactValue,
    *,
    sign: int = 1,
) -> _ClosedExactValue:
    terms = dict(left.terms)
    for radicand, coefficient in right.terms:
        terms[radicand] = terms.get(radicand, Fraction(0)) + sign * coefficient
    return _closed_value(terms)


def _closed_multiply(
    left: _ClosedExactValue,
    right: _ClosedExactValue,
) -> _ClosedExactValue:
    terms: dict[int, Fraction] = {}
    for left_radicand, left_coefficient in left.terms:
        for right_radicand, right_coefficient in right.terms:
            common = math.gcd(left_radicand, right_radicand)
            radicand = (
                left_radicand // common
            ) * (
                right_radicand // common
            )
            coefficient = left_coefficient * right_coefficient * common
            terms[radicand] = terms.get(radicand, Fraction(0)) + coefficient
    return _closed_value(terms)


def _closed_divide(
    numerator: _ClosedExactValue,
    denominator: _ClosedExactValue,
) -> _ClosedExactValue:
    rational_denominator = denominator.rational
    if rational_denominator is None:
        raise _ClosedExpressionReject("nonrational_divisor_unsupported")
    if not rational_denominator:
        raise _ClosedExpressionReject("division_by_zero")
    return _closed_value(
        {
            radicand: coefficient / rational_denominator
            for radicand, coefficient in numerator.terms
        }
    )


def _closed_power(base: _ClosedExactValue, exponent: int) -> _ClosedExactValue:
    if exponent < 0:
        raise _ClosedExpressionReject("negative_exponent")
    if exponent > MAX_CLOSED_EXPR_EXPONENT:
        raise _ClosedExpressionReject("exponent_limit")
    if exponent == 0 and not base.terms:
        raise _ClosedExpressionReject("zero_to_zero")
    result = _closed_rational(1)
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _closed_multiply(result, factor)
        remaining //= 2
        if remaining:
            factor = _closed_multiply(factor, factor)
    return result


def _squarefree_decomposition(value: int) -> tuple[int, int]:
    if value < 1 or value.bit_length() > MAX_CLOSED_EXPR_RADICAND_BITS:
        raise _ClosedExpressionReject("radicand_limit")
    outside = 1
    squarefree = 1
    remaining = value
    divisor = 2
    while divisor * divisor <= remaining:
        exponent = 0
        while remaining % divisor == 0:
            remaining //= divisor
            exponent += 1
        if exponent:
            outside *= divisor ** (exponent // 2)
            if exponent % 2:
                squarefree *= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    if remaining > 1:
        squarefree *= remaining
    return outside, squarefree


def _closed_sqrt(value: _ClosedExactValue) -> _ClosedExactValue:
    rational = value.rational
    if rational is None:
        raise _ClosedExpressionReject("nonrational_radicand_not_exactly_supported")
    if rational < 0:
        raise _ClosedExpressionReject("negative_square_root")
    if not rational:
        return _closed_rational(0)
    numerator_outside, numerator_squarefree = _squarefree_decomposition(
        rational.numerator
    )
    denominator_outside, denominator_squarefree = _squarefree_decomposition(
        rational.denominator
    )
    radicand = numerator_squarefree * denominator_squarefree
    coefficient = Fraction(
        numerator_outside,
        denominator_outside * denominator_squarefree,
    )
    return _closed_value({radicand: coefficient})


def _tokenize_closed_expression(text: str) -> list[_ClosedToken]:
    if not text:
        raise _ClosedExpressionReject("empty_expression")
    if len(text) > MAX_CLOSED_EXPR_CHARS:
        raise _ClosedExpressionReject("expression_length_limit")
    tokens: list[_ClosedToken] = []
    simple = {
        "+": "PLUS",
        "-": "MINUS",
        "*": "MUL",
        "·": "MUL",
        "×": "MUL",
        "/": "DIV",
        "^": "POW",
        "!": "BANG",
        "(": "LPAREN",
        ")": "RPAREN",
        "{": "LBRACE",
        "}": "RBRACE",
    }
    command_kinds = {
        "frac": "FRAC",
        "dfrac": "FRAC",
        "tfrac": "FRAC",
        "sqrt": "SQRT",
        "cdot": "MUL",
        "times": "MUL",
    }
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char.isdigit() and char.isascii():
            tokens.append(_ClosedToken("DIGIT", char, index))
            index += 1
        elif char in simple:
            tokens.append(_ClosedToken(simple[char], char, index))
            index += 1
        elif char == "\\":
            match = re.match(r"\\([A-Za-z]+)", text[index:])
            if match is None:
                raise _ClosedExpressionReject("unsupported_latex_escape")
            command = match.group(1)
            kind = command_kinds.get(command)
            if kind is None:
                raise _ClosedExpressionReject(f"unsupported_latex_command:{command}")
            tokens.append(_ClosedToken(kind, command, index))
            index += len(match.group(0))
        elif char.isalpha() or char == "_":
            raise _ClosedExpressionReject("identifier_or_subscript_forbidden")
        else:
            raise _ClosedExpressionReject(f"forbidden_character:{ord(char)}")
        if len(tokens) > MAX_CLOSED_EXPR_TOKENS:
            raise _ClosedExpressionReject("token_limit")
    tokens.append(_ClosedToken("EOF", "", len(text)))
    return tokens


class _ClosedExpressionParser:
    def __init__(self, tokens: list[_ClosedToken]):
        self.tokens = tokens
        self.index = 0
        self.nodes = 0

    def _current(self) -> _ClosedToken:
        return self.tokens[self.index]

    def _accept(self, kind: str) -> _ClosedToken | None:
        if self._current().kind != kind:
            return None
        token = self._current()
        self.index += 1
        return token

    def _expect(self, kind: str) -> _ClosedToken:
        token = self._accept(kind)
        if token is None:
            raise _ClosedExpressionReject(
                f"expected_{kind.lower()}_at_{self._current().offset}"
            )
        return token

    def _node(
        self,
        kind: str,
        *,
        value: Any = None,
        left: _ClosedNode | None = None,
        right: _ClosedNode | None = None,
    ) -> _ClosedNode:
        self.nodes += 1
        if self.nodes > MAX_CLOSED_EXPR_NODES:
            raise _ClosedExpressionReject("ast_node_limit")
        return _ClosedNode(kind, value=value, left=left, right=right)

    @staticmethod
    def _check_depth(depth: int) -> None:
        if depth > MAX_CLOSED_EXPR_DEPTH:
            raise _ClosedExpressionReject("nesting_depth_limit")

    def parse(self) -> _ClosedNode:
        node = self._parse_expression(0)
        if self._current().kind != "EOF":
            raise _ClosedExpressionReject(
                f"unconsumed_token_at_{self._current().offset}"
            )
        return node

    def _parse_expression(self, depth: int) -> _ClosedNode:
        self._check_depth(depth)
        node = self._parse_term(depth)
        while self._current().kind in {"PLUS", "MINUS"}:
            operator = self._current().kind
            self.index += 1
            node = self._node(
                "add" if operator == "PLUS" else "subtract",
                left=node,
                right=self._parse_term(depth),
            )
        return node

    def _parse_term(self, depth: int) -> _ClosedNode:
        node = self._parse_unary(depth)
        while self._current().kind in {"MUL", "DIV"}:
            operator = self._current().kind
            self.index += 1
            node = self._node(
                "multiply" if operator == "MUL" else "divide",
                left=node,
                right=self._parse_unary(depth),
            )
        return node

    def _parse_unary(self, depth: int) -> _ClosedNode:
        self._check_depth(depth)
        if self._accept("PLUS") is not None:
            return self._node("positive", left=self._parse_unary(depth + 1))
        if self._accept("MINUS") is not None:
            return self._node("negative", left=self._parse_unary(depth + 1))
        return self._parse_power(depth)

    def _parse_power(self, depth: int) -> _ClosedNode:
        node = self._parse_postfix(depth)
        if self._accept("POW") is not None:
            if self._accept("LBRACE") is not None:
                exponent = self._parse_expression(depth + 1)
                self._expect("RBRACE")
            elif self._current().kind == "DIGIT":
                exponent = self._single_digit_node()
            else:
                raise _ClosedExpressionReject("power_requires_braced_or_single_digit_exponent")
            node = self._node("power", left=node, right=exponent)
        return node

    def _parse_postfix(self, depth: int) -> _ClosedNode:
        node = self._parse_primary(depth)
        if self._accept("BANG") is not None:
            node = self._node("factorial", left=node)
            if self._current().kind == "BANG":
                raise _ClosedExpressionReject("repeated_factorial_ambiguous")
        return node

    def _integer_node(self) -> _ClosedNode:
        digits: list[str] = []
        while self._current().kind == "DIGIT":
            digits.append(self._current().value)
            self.index += 1
            if len(digits) > MAX_CLOSED_EXPR_INTEGER_DIGITS:
                raise _ClosedExpressionReject("integer_digit_limit")
        if not digits:
            raise _ClosedExpressionReject(
                f"expected_integer_at_{self._current().offset}"
            )
        return self._node("integer", value=int("".join(digits)))

    def _single_digit_node(self) -> _ClosedNode:
        token = self._expect("DIGIT")
        return self._node("integer", value=int(token.value))

    def _group(self, closing: str, depth: int) -> _ClosedNode:
        node = self._parse_expression(depth + 1)
        self._expect(closing)
        return node

    def _tex_argument(self, depth: int) -> _ClosedNode:
        if self._accept("LBRACE") is not None:
            return self._group("RBRACE", depth)
        if self._current().kind == "DIGIT":
            return self._single_digit_node()
        raise _ClosedExpressionReject("tex_fraction_argument_must_be_braced_or_single_digit")

    def _parse_primary(self, depth: int) -> _ClosedNode:
        self._check_depth(depth)
        if self._current().kind == "DIGIT":
            return self._integer_node()
        if self._accept("LPAREN") is not None:
            return self._group("RPAREN", depth)
        if self._accept("LBRACE") is not None:
            return self._group("RBRACE", depth)
        if self._accept("FRAC") is not None:
            numerator = self._tex_argument(depth + 1)
            denominator = self._tex_argument(depth + 1)
            return self._node("divide", left=numerator, right=denominator)
        if self._accept("SQRT") is not None:
            self._expect("LBRACE")
            radicand = self._group("RBRACE", depth)
            return self._node("sqrt", left=radicand)
        raise _ClosedExpressionReject(
            f"expected_numeric_primary_at_{self._current().offset}"
        )


def _evaluate_closed_node(node: _ClosedNode, depth: int = 0) -> _ClosedExactValue:
    if depth > MAX_CLOSED_EXPR_DEPTH:
        raise _ClosedExpressionReject("evaluation_depth_limit")
    if node.kind == "integer":
        return _closed_rational(int(node.value))
    if node.left is None:
        raise _ClosedExpressionReject("invalid_internal_ast")
    left = _evaluate_closed_node(node.left, depth + 1)
    if node.kind == "positive":
        return left
    if node.kind == "negative":
        return _closed_multiply(_closed_rational(-1), left)
    if node.kind == "factorial":
        rational = left.rational
        if rational is None or rational.denominator != 1 or rational < 0:
            raise _ClosedExpressionReject("factorial_requires_nonnegative_integer")
        argument = rational.numerator
        if argument > MAX_CLOSED_EXPR_FACTORIAL:
            raise _ClosedExpressionReject("factorial_limit")
        return _closed_rational(math.factorial(argument))
    if node.kind == "sqrt":
        return _closed_sqrt(left)
    if node.right is None:
        raise _ClosedExpressionReject("invalid_internal_ast")
    right = _evaluate_closed_node(node.right, depth + 1)
    if node.kind == "add":
        return _closed_add(left, right)
    if node.kind == "subtract":
        return _closed_add(left, right, sign=-1)
    if node.kind == "multiply":
        return _closed_multiply(left, right)
    if node.kind == "divide":
        return _closed_divide(left, right)
    if node.kind == "power":
        rational_exponent = right.rational
        if (
            rational_exponent is None
            or rational_exponent.denominator != 1
            or rational_exponent < 0
        ):
            raise _ClosedExpressionReject("power_requires_nonnegative_integer_exponent")
        return _closed_power(left, rational_exponent.numerator)
    raise _ClosedExpressionReject("invalid_internal_ast")


def _fraction_text(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _canonical_closed_value(value: _ClosedExactValue) -> dict[str, Any]:
    return {
        "type": "closed_numeric",
        "terms": [
            {
                "coefficient": _fraction_text(coefficient),
                "sqrt_radicand": radicand,
            }
            for radicand, coefficient in value.terms
        ],
    }


def parse_closed_numeric_expression(value: Any) -> dict[str, Any]:
    normalized = _normalize_text(value)
    if normalized is None:
        return {
            "parsed": False,
            "value": None,
            "canonical": None,
            "reason": "normalization_rejected",
        }
    try:
        tokens = _tokenize_closed_expression(normalized)
        ast = _ClosedExpressionParser(tokens).parse()
        exact = _evaluate_closed_node(ast)
    except _ClosedExpressionReject as exc:
        return {
            "parsed": False,
            "value": None,
            "canonical": None,
            "reason": exc.reason,
        }
    return {
        "parsed": True,
        "value": exact,
        "canonical": _canonical_closed_value(exact),
        "reason": "closed_numeric_expression_parsed",
    }


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = _resolve(path)
    if not source.exists():
        return rows
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return bool(backslashes % 2)


def _matching_group_end(text: str, start: int) -> int | None:
    pairs = {"(": ")", "[": "]", "{": "}"}
    opening = text[start] if 0 <= start < len(text) else ""
    if opening not in pairs or _is_escaped(text, start):
        return None
    stack: list[str] = []
    for index in range(start, len(text)):
        char = text[index]
        if _is_escaped(text, index):
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack.pop() != char:
                return None
            if not stack:
                return index
    return None


def _fully_wrapped_group(text: str, opening: str, closing: str) -> bool:
    return (
        text.startswith(opening)
        and text.endswith(closing)
        and _matching_group_end(text, 0) == len(text) - 1
    )


def _strip_terminal_punctuation(text: str) -> str:
    text = text.strip()
    while text and text[-1] in "。;；!?！？":
        if text[-1] == "!":
            factorial_prefix = text[:-1].rstrip("!")
            if factorial_prefix and factorial_prefix[-1] in "0123456789)]}":
                break
        text = text[:-1].rstrip()
    while text.endswith(".") and (len(text) == 1 or not text[-2].isdigit()):
        text = text[:-1].rstrip()
    return text


def _full_dollar_wrapper(text: str) -> tuple[int, int] | None:
    delimiter_length = 2 if text.startswith("$$") and text.endswith("$$") else 1
    delimiter = "$" * delimiter_length
    if not text.startswith(delimiter) or not text.endswith(delimiter):
        return None
    interior = text[delimiter_length:-delimiter_length]
    if not interior:
        return None
    for index, char in enumerate(interior):
        if char == "$" and not _is_escaped(interior, index):
            return None
    return (delimiter_length, delimiter_length)


def _full_braced_command(text: str) -> str | None:
    match = re.match(r"\\(?:boxed|fbox|text|mathrm)\s*", text)
    if not match or match.end() >= len(text) or text[match.end()] != "{":
        return None
    end = _matching_group_end(text, match.end())
    if end != len(text) - 1:
        return None
    return text[match.end() + 1 : end]


def _normalize_text(value: Any, depth: int = 0) -> str | None:
    if depth > MAX_NORMALIZATION_DEPTH:
        return None
    text = str("" if value is None else value).strip()
    if len(text) > MAX_ANSWER_CHARS:
        return None
    text = text.replace("−", "-").replace("–", "-").replace("，", ",")
    text = re.sub(r"^(?:final\s+answer|answer|最终答案|答案)\s*[:：]\s*", "", text, flags=re.I)
    text = _strip_terminal_punctuation(text)

    for _ in range(MAX_NORMALIZATION_DEPTH - depth + 1):
        previous = text
        text = re.sub(r"\\(?:left|right)\b\s*", "", text).strip()
        dollar_wrapper = _full_dollar_wrapper(text)
        if dollar_wrapper is not None:
            left, right = dollar_wrapper
            text = text[left : len(text) - right].strip()
        elif text.startswith(r"\(") and text.endswith(r"\)"):
            text = text[2:-2].strip()
        elif text.startswith(r"\[") and text.endswith(r"\]"):
            text = text[2:-2].strip()
        elif text.startswith("`") and text.endswith("`") and text.count("`") == 2:
            text = text[1:-1].strip()
        else:
            command_body = _full_braced_command(text)
            if command_body is not None:
                text = command_body.strip()
        text = _strip_terminal_punctuation(text)
        if text == previous:
            break
    else:
        return None

    text = re.sub(r"\\(?:left|right)\b\s*", "", text)
    text = re.sub(r"\\pmod\{([+-]?\d+)\}", r"(mod \1)", text)
    text = text.replace("\\equiv", "≡").replace("\\%", "%")
    return text.strip()


def _clean(value: Any) -> str:
    normalized = _normalize_text(value)
    return normalized if normalized is not None else str(value or "").strip()


def _result_reliability(result: dict[str, Any] | None) -> str:
    status = str(
        (result or {}).get("mathematical_verification_status") or "not_evaluated"
    ).lower()
    if status == "passed":
        return "verified"
    if status == "failed":
        return "rejected"
    if (result or {}).get("output_quality_passed") is True and status in {
        "not_applicable",
        "unknown",
    }:
        return "unverified"
    return "unavailable"


def _latex_fraction_parts(text: str) -> tuple[str, str] | None:
    match = re.match(r"\\(?:frac|dfrac|tfrac)\s*", text)
    if not match or match.end() >= len(text) or text[match.end()] != "{":
        return None
    numerator_end = _matching_group_end(text, match.end())
    if numerator_end is None:
        return None
    denominator_start = numerator_end + 1
    while denominator_start < len(text) and text[denominator_start].isspace():
        denominator_start += 1
    if denominator_start >= len(text) or text[denominator_start] != "{":
        return None
    denominator_end = _matching_group_end(text, denominator_start)
    if denominator_end != len(text) - 1:
        return None
    return (
        text[match.end() + 1 : numerator_end],
        text[denominator_start + 1 : denominator_end],
    )


def _number(value: str, depth: int = 0) -> Fraction | None:
    if depth > MAX_NORMALIZATION_DEPTH:
        return None
    normalized = _normalize_text(value, depth)
    if normalized is None:
        return None
    text = normalized.strip()
    latex_fraction = _latex_fraction_parts(text)
    if latex_fraction is not None:
        numerator = _number(latex_fraction[0], depth + 1)
        denominator = _number(latex_fraction[1], depth + 1)
        if numerator is None or denominator is None or not denominator:
            return None
        return numerator / denominator
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", text):
        text = text.replace(",", "")
    digit_count = sum(char.isascii() and char.isdigit() for char in text)
    if digit_count > MAX_SIMPLE_NUMBER_DIGITS:
        return None
    exponent_match = re.search(r"[eE]([+-]?)(\d+)$", text)
    if exponent_match is not None:
        exponent_digits = exponent_match.group(2)
        if len(exponent_digits) > 6:
            return None
        exponent = int(exponent_digits)
        if exponent > MAX_SIMPLE_NUMBER_EXPONENT:
            return None
    try:
        if re.fullmatch(rf"{NUMBER}\s*/\s*{NUMBER}", text):
            numerator, denominator = re.split(r"\s*/\s*", text)
            den = Fraction(Decimal(denominator))
            return Fraction(Decimal(numerator)) / den if den else None
        if re.fullmatch(NUMBER, text):
            return Fraction(Decimal(text))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return None


def _split_top_level(text: str) -> list[str] | None:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    pieces: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if _is_escaped(text, index):
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack.pop() != char:
                return None
        elif char == "," and not stack:
            pieces.append(text[start:index].strip())
            start = index + 1
    if stack:
        return None
    pieces.append(text[start:].strip())
    return pieces


def _qualified_no_solution(lower: str) -> str | None:
    patterns = (
        r"(?:there\s+(?:is|are)\s+)?no\s+(.+?)\s+solutions?",
        r"no\s+solutions?\s+(?:in|over)\s+(?:the\s+)?(.+)",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, lower)
        if match:
            qualifier = re.sub(r"\s+", " ", match.group(1)).strip()
            return NO_SOLUTION_QUALIFIERS.get(qualifier)
    return None


def _explicit_set_body(text: str) -> str | None:
    if _fully_wrapped_group(text, "{", "}"):
        return text[1:-1]
    if text.startswith(r"\{") and text.endswith(r"\}"):
        return text[2:-2]
    return None


def _parse_answer(
    value: Any,
    depth: int = 0,
    answer_type: Any = None,
) -> tuple[str, Any] | None:
    if depth > MAX_NORMALIZATION_DEPTH:
        return None
    text = _normalize_text(value, depth)
    if text is None:
        return None
    lower = re.sub(r"\s+", " ", text.lower()).strip()
    normalized_answer_type = str(answer_type or "").strip().lower()
    if lower in NO_SOLUTION:
        return ("no_solution", None)
    qualifier = _qualified_no_solution(lower)
    if qualifier is not None:
        return ("no_solution", qualifier)
    if (
        normalized_answer_type in {"expression", "no_solution", "text"}
        and _fully_wrapped_group(text, "{", "}")
    ):
        grouped = _parse_answer(text[1:-1], depth + 1, answer_type=answer_type)
        if grouped is not None and grouped[0] == "no_solution":
            return grouped

    if _fully_wrapped_group(text, "(", ")"):
        pieces = _split_top_level(text[1:-1])
        if (
            pieces is not None
            and 2 <= len(pieces) <= MAX_COLLECTION_ITEMS
            and all(pieces)
        ):
            elements = tuple(_parse_answer(piece, depth + 1) for piece in pieces)
            if all(element is not None for element in elements):
                return ("tuple", elements)

    set_body = _explicit_set_body(text)
    if set_body is not None:
        ambiguous_markers = (
            r"\ldots",
            r"\dots",
            r"\cdots",
            "...",
            "…",
            r"\mid",
            "|",
            ":",
        )
        if any(marker in set_body for marker in ambiguous_markers):
            return None
        pieces = [] if not set_body.strip() else _split_top_level(set_body)
        if (
            pieces is not None
            and len(pieces) <= MAX_COLLECTION_ITEMS
            and all(pieces)
        ):
            elements = tuple(_parse_answer(piece, depth + 1) for piece in pieces)
            if all(element is not None for element in elements):
                return ("finite_set", elements)

    if normalized_answer_type in {"set", "finite_set"}:
        pieces = _split_top_level(text)
        if (
            pieces is not None
            and 2 <= len(pieces) <= MAX_COLLECTION_ITEMS
            and all(_fully_wrapped_group(piece, "(", ")") for piece in pieces)
        ):
            elements = tuple(_parse_answer(piece, depth + 1) for piece in pieces)
            if all(element is not None and element[0] == "tuple" for element in elements):
                return ("finite_set", elements)

    choice = re.fullmatch(r"(?:option|choice|选项)?\s*\(?([a-e])\)?", lower, flags=re.I)
    if choice:
        return ("choice", choice.group(1).upper())

    congruence = re.fullmatch(
        rf"(?:[a-z]\s*)?(?:≡|=)?\s*({NUMBER})\s*\(?\s*mod(?:ulo)?\s*({NUMBER})\s*\)?",
        lower,
    )
    if congruence:
        residue, modulus = _number(congruence.group(1)), _number(congruence.group(2))
        if residue is not None and modulus is not None and modulus.denominator == 1 and modulus:
            mod_int = abs(modulus.numerator)
            if residue.denominator == 1:
                return ("congruence", (residue.numerator % mod_int, mod_int))

    roots = re.fullmatch(r"[a-z]\s*=\s*(.+)", lower)
    if roots and "," in roots.group(1):
        pieces = [re.sub(r"^[a-z]\s*=\s*", "", part.strip()) for part in roots.group(1).split(",")]
        numbers = [_number(part) for part in pieces]
        if pieces and all(number is not None for number in numbers):
            return ("roots", frozenset(numbers))

    assignment = re.fullmatch(r"[a-z]\s*=\s*(.+)", lower)
    if assignment:
        assigned = assignment.group(1).strip()
        if assigned.endswith("%"):
            numeric = _number(assigned[:-1].strip(), depth + 1)
            return ("number", numeric / 100) if numeric is not None else None
        numeric = _number(assigned, depth + 1)
        return ("number", numeric) if numeric is not None else None

    if lower.endswith("%"):
        numeric = _number(lower[:-1].strip())
        return ("number", numeric / 100) if numeric is not None else None
    numeric = _number(lower)
    return ("number", numeric) if numeric is not None else None


def parse_simple_answer(value: Any, *, answer_type: Any = None) -> tuple[str, Any] | None:
    return _parse_answer(value, answer_type=answer_type)


def _parsed_equal(expected: tuple[str, Any], predicted: tuple[str, Any]) -> bool:
    expected_type, expected_value = expected
    predicted_type, predicted_value = predicted
    if expected_type != predicted_type:
        return False
    if expected_type == "tuple":
        return len(expected_value) == len(predicted_value) and all(
            _parsed_equal(expected_element, predicted_element)
            for expected_element, predicted_element in zip(expected_value, predicted_value)
        )
    if expected_type == "finite_set":
        if len(expected_value) != len(predicted_value):
            return False
        unmatched = list(predicted_value)
        for expected_element in expected_value:
            for index, predicted_element in enumerate(unmatched):
                if _parsed_equal(expected_element, predicted_element):
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched
    return expected_value == predicted_value


def _canonical_answer(parsed: tuple[str, Any] | None) -> Any:
    if parsed is None:
        return None
    answer_type, value = parsed
    if answer_type == "number":
        canonical_value: Any = _fraction_text(value)
    elif answer_type == "no_solution":
        canonical_value = value or "unqualified"
    elif answer_type == "congruence":
        canonical_value = {"residue": value[0], "modulus": value[1]}
    elif answer_type == "roots":
        canonical_value = sorted(_fraction_text(root) for root in value)
    elif answer_type == "tuple":
        canonical_value = [_canonical_answer(element) for element in value]
    elif answer_type == "finite_set":
        canonical_value = sorted(
            (_canonical_answer(element) for element in value),
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    else:
        canonical_value = value
    return {"type": answer_type, "value": canonical_value}


def compare_answers_detailed(
    expected: Any,
    predicted: Any,
    *,
    answer_type: Any = None,
) -> dict[str, Any]:
    expected_parsed = parse_simple_answer(expected, answer_type=answer_type)
    predicted_parsed = parse_simple_answer(predicted, answer_type=answer_type)
    v2_comparison_type = (
        expected_parsed[0]
        if expected_parsed is not None
        and predicted_parsed is not None
        and expected_parsed[0] == predicted_parsed[0]
        else "type_mismatch"
        if expected_parsed is not None and predicted_parsed is not None
        else "unparseable"
    )
    v2_equivalent = (
        _parsed_equal(expected_parsed, predicted_parsed)
        if expected_parsed is not None and predicted_parsed is not None
        else None
    )
    expected_closed: dict[str, Any] | None = None
    predicted_closed: dict[str, Any] | None = None
    equivalent = v2_equivalent
    comparison_type = v2_comparison_type
    canonical_reference = _canonical_answer(expected_parsed)
    canonical_candidate = _canonical_answer(predicted_parsed)
    parse_or_reject_reason = (
        f"v2_existing_{v2_comparison_type}_"
        f"{'equal' if v2_equivalent else 'differ'}"
        if v2_equivalent is not None
        else "v2_unparseable"
    )

    # Preserve every v2 decision. The strict closed-expression path is only an
    # exact fallback when v2 could not safely parse at least one side.
    if v2_equivalent is None:
        expected_closed = parse_closed_numeric_expression(expected)
        predicted_closed = parse_closed_numeric_expression(predicted)
        if expected_closed["parsed"] and predicted_closed["parsed"]:
            equivalent = expected_closed["value"] == predicted_closed["value"]
            comparison_type = "closed_numeric_expression"
            canonical_reference = expected_closed["canonical"]
            canonical_candidate = predicted_closed["canonical"]
            parse_or_reject_reason = (
                "closed_numeric_expression_exact_equal"
                if equivalent
                else "closed_numeric_expression_exact_differ"
            )
        else:
            rejected: list[str] = []
            if not expected_closed["parsed"]:
                rejected.append(f"reference:{expected_closed['reason']}")
            if not predicted_closed["parsed"]:
                rejected.append(f"candidate:{predicted_closed['reason']}")
            parse_or_reject_reason = (
                "closed_numeric_expression_rejected:" + ",".join(rejected)
            )

    return {
        "equivalent": equivalent,
        "comparison_type": comparison_type,
        "canonical_reference": canonical_reference,
        "canonical_candidate": canonical_candidate,
        "parse_or_reject_reason": parse_or_reject_reason,
        "v2_equivalent": v2_equivalent,
        "v2_comparison_type": v2_comparison_type,
        "v2_canonical_reference": _canonical_answer(expected_parsed),
        "v2_canonical_candidate": _canonical_answer(predicted_parsed),
        "closed_reference_reason": (
            expected_closed["reason"] if expected_closed is not None else "not_attempted"
        ),
        "closed_candidate_reason": (
            predicted_closed["reason"] if predicted_closed is not None else "not_attempted"
        ),
    }


def compare_answers(
    expected: Any,
    predicted: Any,
    *,
    answer_type: Any = None,
) -> bool | None:
    """Return True/False for safely decidable answers and None for manual review."""
    return compare_answers_detailed(
        expected,
        predicted,
        answer_type=answer_type,
    )["equivalent"]


def _legacy_clean(value: Any) -> str:
    """The scorer normalization at HEAD, frozen for profile compatibility."""
    text = str(value or "").strip()
    text = text.replace("−", "-").replace("–", "-").replace("，", ",")
    text = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", r"\1/\2", text)
    text = re.sub(r"\\pmod\{([+-]?\d+)\}", r"(mod \1)", text)
    text = text.replace("\\equiv", "≡").replace("\\%", "%")
    text = text.replace("$", "").replace("`", "")
    text = re.sub(
        r"^(?:final\s+answer|answer|最终答案|答案)\s*[:：]\s*",
        "",
        text,
        flags=re.I,
    )
    return text.strip().strip(".。;；").strip()


def _legacy_number(value: str) -> Fraction | None:
    text = value.strip()
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", text):
        text = text.replace(",", "")
    if sum(char.isascii() and char.isdigit() for char in text) > MAX_SIMPLE_NUMBER_DIGITS:
        return None
    exponent_match = re.search(r"[eE]([+-]?)(\d+)$", text)
    if exponent_match is not None:
        exponent_digits = exponent_match.group(2)
        if len(exponent_digits) > 6 or int(exponent_digits) > MAX_SIMPLE_NUMBER_EXPONENT:
            return None
    try:
        if re.fullmatch(rf"{NUMBER}\s*/\s*{NUMBER}", text):
            numerator, denominator = re.split(r"\s*/\s*", text)
            den = Fraction(Decimal(denominator))
            return Fraction(Decimal(numerator)) / den if den else None
        if re.fullmatch(NUMBER, text):
            return Fraction(Decimal(text))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return None


def _legacy_parse_simple_answer(value: Any) -> tuple[str, Any] | None:
    text = _legacy_clean(value)
    lower = re.sub(r"\s+", " ", text.lower()).strip()
    compact = re.sub(r"\s+", "", lower)
    if lower in NO_SOLUTION or compact in {
        item.replace(" ", "") for item in NO_SOLUTION
    }:
        return ("no_solution", True)

    choice = re.fullmatch(
        r"(?:option|choice|选项)?\s*\(?([a-e])\)?",
        lower,
        flags=re.I,
    )
    if choice:
        return ("choice", choice.group(1).upper())

    congruence = re.fullmatch(
        rf"(?:[a-z]\s*)?(?:≡|=)?\s*({NUMBER})\s*\(?\s*"
        rf"mod(?:ulo)?\s*({NUMBER})\s*\)?",
        lower,
    )
    if congruence:
        residue = _legacy_number(congruence.group(1))
        modulus = _legacy_number(congruence.group(2))
        if (
            residue is not None
            and modulus is not None
            and modulus.denominator == 1
            and modulus
        ):
            mod_int = abs(modulus.numerator)
            if residue.denominator == 1:
                return ("congruence", (residue.numerator % mod_int, mod_int))

    roots = re.fullmatch(r"[a-z]\s*=\s*(.+)", lower)
    if roots and "," in roots.group(1):
        pieces = [
            re.sub(r"^[a-z]\s*=\s*", "", part.strip())
            for part in roots.group(1).split(",")
        ]
        numbers = [_legacy_number(part) for part in pieces]
        if pieces and all(number is not None for number in numbers):
            return ("roots", frozenset(numbers))

    pair = re.fullmatch(
        rf"\(\s*({NUMBER}(?:\s*/\s*{NUMBER})?)\s*,\s*"
        rf"({NUMBER}(?:\s*/\s*{NUMBER})?)\s*\)",
        lower,
    )
    if pair:
        first = _legacy_number(pair.group(1))
        second = _legacy_number(pair.group(2))
        if first is not None and second is not None:
            return ("ordered_pair", (first, second))

    assignment = re.fullmatch(
        rf"[a-z]\s*=\s*({NUMBER}(?:\s*/\s*{NUMBER})?%?)",
        lower,
    )
    if assignment:
        lower = assignment.group(1)

    if lower.endswith("%"):
        numeric = _legacy_number(lower[:-1].strip())
        return ("number", numeric / 100) if numeric is not None else None
    numeric = _legacy_number(lower)
    return ("number", numeric) if numeric is not None else None


def _legacy_compare_answers(expected: Any, predicted: Any) -> bool | None:
    expected_parsed = _legacy_parse_simple_answer(expected)
    predicted_parsed = _legacy_parse_simple_answer(predicted)
    if expected_parsed is None or predicted_parsed is None:
        return None
    if expected_parsed[0] == predicted_parsed[0]:
        return expected_parsed[1] == predicted_parsed[1]
    return False


def _parse_tuple_enumeration_exact(value: Any) -> tuple[str, Any] | None:
    text = _normalize_text(value)
    if text is None:
        return None
    pieces = _split_top_level(text)
    if (
        pieces is None
        or not 2 <= len(pieces) <= MAX_COLLECTION_ITEMS
        or not all(_fully_wrapped_group(piece, "(", ")") for piece in pieces)
    ):
        return None
    elements = tuple(_parse_answer(piece) for piece in pieces)
    if not all(element is not None and element[0] == "tuple" for element in elements):
        return None
    return ("finite_set", elements)


def _parse_exact_value(value: Any) -> tuple[str, Any] | None:
    parsed = parse_simple_answer(value)
    if parsed is not None:
        if (
            parsed[0] == "finite_set"
            and len(parsed[1]) == 1
            and parsed[1][0][0] == "no_solution"
        ):
            parsed = parsed[1][0]
        return ("structured", parsed)
    enumeration = _parse_tuple_enumeration_exact(value)
    if enumeration is not None:
        return ("structured", enumeration)
    closed = parse_closed_numeric_expression(value)
    if closed["parsed"]:
        return ("closed", closed["value"])
    return None


def _canonical_exact_value(parsed: tuple[str, Any] | None) -> Any:
    if parsed is None:
        return None
    kind, value = parsed
    if kind == "structured":
        return _canonical_answer(value)
    return _canonical_closed_value(value)


def _canonical_exact_text(parsed: tuple[str, Any] | None, fallback: Any) -> str:
    canonical = _canonical_exact_value(parsed)
    if canonical is not None:
        return json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    normalized = _normalize_text(fallback)
    return normalized if normalized is not None else ""


def _exact_values_equal(
    expected: tuple[str, Any],
    predicted: tuple[str, Any],
) -> bool:
    expected_kind, expected_value = expected
    predicted_kind, predicted_value = predicted
    if expected_kind == predicted_kind == "structured":
        return _parsed_equal(expected_value, predicted_value)
    if expected_kind == predicted_kind == "closed":
        return expected_value == predicted_value
    if expected_kind == "structured" and expected_value[0] == "number":
        expected_value = _closed_rational(expected_value[1])
    elif expected_kind != "closed":
        return False
    if predicted_kind == "structured" and predicted_value[0] == "number":
        predicted_value = _closed_rational(predicted_value[1])
    elif predicted_kind != "closed":
        return False
    return expected_value == predicted_value


def _extract_balanced_boxed(text: str) -> tuple[list[str], bool]:
    candidates: list[str] = []
    malformed = False
    position = 0
    marker = r"\boxed"
    while True:
        found = text.find(marker, position)
        if found < 0:
            break
        group_start = found + len(marker)
        while group_start < len(text) and text[group_start].isspace():
            group_start += 1
        if group_start >= len(text) or text[group_start] != "{":
            malformed = True
            position = max(group_start, found + len(marker))
            continue
        group_end = _matching_group_end(text, group_start)
        if group_end is None:
            malformed = True
            position = group_start + 1
            continue
        candidates.append(text[group_start + 1 : group_end].strip())
        position = group_end + 1
    return candidates, malformed


def _consistent_candidates(candidates: list[str]) -> bool:
    parsed = [_parse_exact_value(candidate) for candidate in candidates]
    if any(value is None for value in parsed):
        return False
    first = parsed[0]
    return all(
        _exact_values_equal(first, value)
        for value in parsed[1:]
        if first is not None and value is not None
    )


def _candidate_can_be_mined(parsed: tuple[str, Any] | None) -> bool:
    return not (
        parsed is None
        or (parsed[0] == "structured" and parsed[1][0] == "choice")
    )


def _explicit_final_candidates(text: str) -> list[str]:
    tagged = re.findall(
        r"<final_answer>\s*(.*?)\s*</final_answer>",
        text,
        flags=re.I | re.S,
    )
    labelled = re.findall(
        r"(?im)^\s*(?:final\s+answer|answer|最终答案|答案)\s*[:：]\s*(.+?)\s*$",
        text,
    )
    return [candidate.strip() for candidate in (*tagged, *labelled) if candidate.strip()]


def _candidate_from_prediction(predicted_answer: Any) -> dict[str, Any]:
    text = str("" if predicted_answer is None else predicted_answer).strip()
    empty = {
        "status": "unresolved",
        "candidate": "",
        "source": "none",
        "reason": "prediction_empty",
    }
    if not text:
        return empty
    if len(text) > MAX_ANSWER_CHARS:
        return {
            **empty,
            "reason": "prediction_length_limit",
        }

    boxed, malformed = _extract_balanced_boxed(text)
    if malformed:
        return {
            **empty,
            "source": "boxed",
            "reason": "malformed_boxed_candidate",
        }
    if boxed:
        if len(boxed) > 1 and not _consistent_candidates(boxed):
            return {
                **empty,
                "source": "boxed",
                "reason": "conflicting_boxed_candidates",
            }
        if not _candidate_can_be_mined(_parse_exact_value(boxed[0])):
            return {
                **empty,
                "source": "boxed",
                "reason": "boxed_candidate_unparseable",
            }
        return {
            "status": "candidate",
            "candidate": boxed[0],
            "source": (
                "boxed"
                if len(boxed) == 1
                else "consistent_multiple_boxed"
            ),
            "reason": "candidate_extracted",
        }

    if _parse_exact_value(text) is not None:
        return {
            "status": "candidate",
            "candidate": text,
            "source": "full_response",
            "reason": "candidate_extracted",
        }

    finals = _explicit_final_candidates(text)
    if finals:
        if len(finals) > 1 and not _consistent_candidates(finals):
            return {
                **empty,
                "source": "explicit_final",
                "reason": "conflicting_final_candidates",
            }
        if _candidate_can_be_mined(_parse_exact_value(finals[0])):
            return {
                "status": "candidate",
                "candidate": finals[0],
                "source": (
                    "explicit_final"
                    if len(finals) == 1
                    else "consistent_multiple_final"
                ),
                "reason": "candidate_extracted",
            }
    return {
        **empty,
        "reason": "candidate_unparseable",
    }


_OPEN_TASK_MARKERS_V4 = (
    "what is a number that",
    "give an example",
    "find an example",
    "provide an example",
    "find any",
)


def _validate_open_constraint_v4(
    problem_text: Any,
    candidate: Any,
) -> tuple[bool | None, str]:
    problem = re.sub(r"\s+", " ", str(problem_text or ""))
    if not any(marker in problem.lower() for marker in _OPEN_TASK_MARKERS_V4):
        return None, "policy_not_applicable"
    progression = re.search(
        r"counts?\s+by\s+(\d+)s?\s+starting\s+at\s+(\d+)",
        problem,
        flags=re.I,
    )
    if progression is None:
        return None, "policy_validator_unavailable"
    parsed = _parse_exact_value(candidate)
    if parsed is None or parsed[0] != "structured":
        return False, "policy_candidate_not_integer"
    answer_type, value = parsed[1]
    if (
        answer_type != "number"
        or value.denominator != 1
    ):
        return False, "policy_candidate_not_integer"
    step = int(progression.group(1))
    start = int(progression.group(2))
    integer = value.numerator
    return (
        integer >= start and (integer - start) % step == 0,
        "policy_arithmetic_progression_membership",
    )


def _score_answer_exact_v4(
    predicted_answer: Any,
    expected_answer: Any,
) -> dict[str, Any]:
    candidate_info = _candidate_from_prediction(predicted_answer)
    candidate = candidate_info["candidate"]
    predicted_parsed = (
        _parse_exact_value(candidate)
        if candidate_info["status"] == "candidate"
        else None
    )
    normalized_prediction = _canonical_exact_text(predicted_parsed, candidate)
    if predicted_parsed is None:
        return {
            "status": "unresolved",
            "profile": "exact_v4",
            "equivalence_class": "none",
            "normalized_prediction": normalized_prediction,
            "normalized_expected": "",
            "candidate_source": candidate_info["source"],
            "reason_code": candidate_info["reason"],
        }

    # expected_answer is intentionally first accessed only after candidate
    # extraction and candidate parsing have completed.
    expected_parsed = _parse_exact_value(expected_answer)
    normalized_expected = _canonical_exact_text(expected_parsed, expected_answer)
    if expected_parsed is None:
        return {
            "status": "unresolved",
            "profile": "exact_v4",
            "equivalence_class": "none",
            "normalized_prediction": normalized_prediction,
            "normalized_expected": normalized_expected,
            "candidate_source": candidate_info["source"],
            "reason_code": "expected_unparseable",
        }
    equivalent = _exact_values_equal(expected_parsed, predicted_parsed)
    return {
        "status": "correct" if equivalent else "incorrect",
        "profile": "exact_v4",
        "equivalence_class": "exact" if equivalent else "none",
        "normalized_prediction": normalized_prediction,
        "normalized_expected": normalized_expected,
        "candidate_source": candidate_info["source"],
        "reason_code": (
            "exact_equivalent"
            if equivalent
            else "exact_not_equivalent"
        ),
    }


def score_answer(
    predicted_answer: Any,
    expected_answer: Any,
    *,
    profile: str = "legacy",
    problem_text: Any = None,
) -> dict[str, Any]:
    """Score a saved answer with a stable, JSON-serializable offline profile."""
    if profile not in {"legacy", "exact_v4", "policy_v4"}:
        raise ValueError(f"unsupported scoring profile: {profile}")
    if profile == "legacy":
        comparison = _legacy_compare_answers(expected_answer, predicted_answer)
        return {
            "status": (
                "correct"
                if comparison is True
                else "incorrect"
                if comparison is False
                else "unresolved"
            ),
            "profile": "legacy",
            "equivalence_class": "legacy" if comparison is True else "none",
            "normalized_prediction": _legacy_clean(predicted_answer),
            "normalized_expected": _legacy_clean(expected_answer),
            "candidate_source": "full_response",
            "reason_code": (
                "legacy_equal"
                if comparison is True
                else "legacy_differ"
                if comparison is False
                else "legacy_unparseable"
            ),
        }

    exact = _score_answer_exact_v4(predicted_answer, expected_answer)
    if profile == "exact_v4":
        return exact
    if exact["status"] == "correct":
        return {
            **exact,
            "profile": "policy_v4",
        }

    candidate_info = _candidate_from_prediction(predicted_answer)
    if candidate_info["status"] != "candidate":
        return {
            **exact,
            "profile": "policy_v4",
        }
    valid, policy_reason = _validate_open_constraint_v4(
        problem_text,
        candidate_info["candidate"],
    )
    if valid is None:
        if policy_reason == "policy_not_applicable":
            return {
                **exact,
                "profile": "policy_v4",
            }
        return {
            **exact,
            "status": "unresolved",
            "profile": "policy_v4",
            "equivalence_class": "none",
            "reason_code": policy_reason,
        }
    return {
        **exact,
        "status": "correct" if valid else "incorrect",
        "profile": "policy_v4",
        "equivalence_class": "policy" if valid else "none",
        "candidate_source": candidate_info["source"],
        "reason_code": (
            "policy_arithmetic_progression_member"
            if valid
            else policy_reason
        ),
    }


def is_solver_compatible(expected_domain: Any, predicted_domain: Any, solver_key: Any = None) -> bool:
    expected = str(expected_domain or "unknown").strip().lower()
    predicted = str(predicted_domain or "unknown").strip().lower()
    if expected == "unknown" or predicted == "unknown":
        return False
    actual_solver = str(solver_key or solver_key_for_domain(predicted)).strip().lower()
    allowed = COMPATIBLE_SOLVER_FAMILIES.get(expected, {solver_key_for_domain(expected)})
    return actual_solver in allowed


def _is_proof(item: dict[str, Any]) -> bool:
    if item.get("response_mode") == "worked_solution" or item.get("answer_type") == "proof":
        return True
    problem = str(item.get("problem") or "").lower()
    return any(marker in problem for marker in PROOF_MARKERS)


def proof_format(final_response: Any, response_mode: Any) -> tuple[str, list[str]]:
    text = str(final_response or "").strip()
    reasons: list[str] = []
    if not text:
        return "format_fail", ["empty final_response"]
    chunks = [part.strip() for part in re.split(r"\n+|(?<=[.!?。！？；;])\s*", text) if part.strip()]
    substantive = [part for part in chunks if len(part) >= 8]
    if len(substantive) < 2 and len(text) < 120:
        reasons.append("only a short conclusion")
    step_markers = re.findall(
        r"(?:first|second|then|therefore|because|hence|step\s*\d+|首先|其次|然后|因为|所以|故)",
        text.lower(),
    )
    if len(substantive) < 2 and len(step_markers) < 2:
        reasons.append("fewer than two substantive steps")
    if not re.search(r"(?:therefore|thus|hence|consequently|proved|qed|∎|所以|故|综上|结论|证毕)", text, re.I):
        reasons.append("missing explicit final conclusion")
    if str(response_mode) != "worked_solution":
        reasons.append("response_mode is not worked_solution")
    return ("format_fail", reasons) if reasons else ("format_pass", [])


def score_item(item: dict[str, Any], result: dict[str, Any] | None) -> dict[str, Any]:
    expected_domain = str(item.get("expected_domain") or "unknown")
    predicted_domain = str((result or {}).get("predicted_domain") or "unknown")
    normalized_reference = _clean(item.get("expected_answer"))
    normalized_prediction = _clean((result or {}).get("final_response"))
    verification_status = str(
        (result or {}).get("mathematical_verification_status") or "not_evaluated"
    ).lower()
    reliability_status = _result_reliability(result)
    pipeline_acceptance = bool(
        (result or {}).get(
            "pipeline_acceptance_passed",
            (result or {}).get(
                "pipeline_acceptance",
                (result or {}).get("final_acceptance_passed", False),
            ),
        )
    )
    scored = {
        "idx": str(item.get("idx") or ""),
        "expected_domain": expected_domain,
        "predicted_domain": predicted_domain,
        "exact_domain_match": expected_domain != "unknown" and predicted_domain == expected_domain,
        "solver_compatible": is_solver_compatible(expected_domain, predicted_domain, (result or {}).get("solver_key")),
        "normalized_prediction": normalized_prediction,
        "normalized_reference": normalized_reference,
        "format_status": None,
        "score_status": "manual_review",
        "score_reason": "not_scored",
        "pipeline_acceptance": pipeline_acceptance,
        "pipeline_acceptance_passed": pipeline_acceptance,
        "mathematical_verification_status": verification_status,
        "mathematical_verification_passed": verification_status == "passed",
        "answer_reliability_status": reliability_status,
        "manual_review_required": reliability_status != "verified",
        "deterministic_answer_override": bool(
            (result or {}).get("deterministic_answer_override")
        ),
        "mathematical_correctness": "manual_review",
        "proof_review_required": bool((result or {}).get("proof_review_required")),
        "proof_risk_signals": list((result or {}).get("proof_risk_signals") or []),
    }
    if item.get("label_status") == "suspected_incorrect":
        scored["score_status"] = "label_suspected"
        scored["score_reason"] = "suspected_reference_label"
        return scored
    if not result or result.get("status") != "success":
        scored["score_status"] = "runtime_error"
        scored["score_reason"] = "missing_or_failed_run"
        scored["mathematical_correctness"] = "unknown"
        return scored
    final_response = str(result.get("final_response") or "").strip()
    if _is_proof(item):
        format_status, reasons = proof_format(final_response, result.get("response_mode"))
        scored["format_status"] = format_status
        scored["format_reasons"] = reasons
        scored["score_status"] = "format_fail" if format_status == "format_fail" else "manual_review"
        scored["score_reason"] = "proof_format_failed" if format_status == "format_fail" else "proof_requires_manual_review"
        scored["mathematical_correctness"] = "manual_review"
        return scored
    if not final_response:
        scored["score_status"] = "format_fail"
        scored["score_reason"] = "empty_final_response"
        scored["mathematical_correctness"] = "unknown"
        return scored
    comparison = compare_answers(
        item.get("expected_answer"),
        final_response,
        answer_type=item.get("answer_type"),
    )
    scored["score_status"] = "correct" if comparison is True else "incorrect" if comparison is False else "manual_review"
    scored["score_reason"] = (
        "normalized_answers_equal"
        if comparison is True
        else "normalized_answers_differ"
        if comparison is False
        else "unsupported_answer_format"
    )
    scored["mathematical_correctness"] = (
        "correct" if comparison is True else "incorrect" if comparison is False else "manual_review"
    )
    return scored


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[rank], 6)


def _risk_lists(
    dataset: list[dict[str, Any]], results: dict[str, dict[str, Any]], scores: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    risks: dict[str, list[dict[str, Any]]] = {
        "api_errors": [],
        "empty_answers": [],
        "routing_errors": [],
        "suspected_local_tool_misfires": [],
        "retry_still_failed": [],
        "fallbacks": [],
        "proof_short_conclusions": [],
        "response_mode_mismatches": [],
        "answer_conflicts": [],
        "suspected_labels": [],
        "proof_review_required": [],
    }
    score_map = {row["idx"]: row for row in scores}
    for item in dataset:
        idx = str(item.get("idx") or "")
        result = results.get(idx, {})
        score = score_map[idx]
        entry = {"idx": idx}
        if not result or result.get("status") != "success":
            risks["api_errors"].append({**entry, "error": result.get("error", "missing result")})
        if result.get("status") == "success" and not str(result.get("final_response") or "").strip():
            risks["empty_answers"].append(entry)
        expected_domain = str(item.get("expected_domain") or "unknown")
        predicted_domain = str(result.get("predicted_domain") or "unknown")
        if expected_domain != "unknown" and predicted_domain != expected_domain:
            risks["routing_errors"].append(
                {
                    **entry,
                    "expected": expected_domain,
                    "predicted": predicted_domain,
                    "solver_compatible": score.get("solver_compatible", False),
                }
            )
            if result.get("local_tool_name"):
                risks["suspected_local_tool_misfires"].append({**entry, "tool": result["local_tool_name"]})
        if result.get("retry_triggered") and not result.get(
            "pipeline_acceptance_passed", result.get("final_acceptance_passed")
        ):
            risks["retry_still_failed"].append(entry)
        if result.get("fallback_used"):
            risks["fallbacks"].append(entry)
        if score.get("format_status") == "format_fail":
            risks["proof_short_conclusions"].append({**entry, "reasons": score.get("format_reasons", [])})
        if score.get("proof_review_required"):
            risks["proof_review_required"].append(
                {**entry, "signals": score.get("proof_risk_signals", [])}
            )
        if result and item.get("response_mode") != result.get("response_mode"):
            risks["response_mode_mismatches"].append(
                {**entry, "expected": item.get("response_mode"), "actual": result.get("response_mode")}
            )
        if score["score_status"] in {"incorrect", "manual_review"} and not _is_proof(item):
            risks["answer_conflicts"].append(entry)
        if item.get("label_status") == "suspected_incorrect":
            risks["suspected_labels"].append({**entry, "review_note": item.get("review_note", "")})
    return risks


def build_report(dataset: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Last record wins, so a resumed success supersedes an earlier failed attempt.
    results = {str(row.get("idx") or ""): row for row in result_rows}
    scores = [score_item(item, results.get(str(item.get("idx") or ""))) for item in dataset]
    score_counts = Counter(row["score_status"] for row in scores)
    run_rows = [results.get(str(item.get("idx") or ""), {}) for item in dataset]
    successful = sum(row.get("status") == "success" for row in run_rows)
    errors = len(dataset) - successful
    empty = sum(row.get("status") == "success" and not str(row.get("final_response") or "").strip() for row in run_rows)
    auto_scored = score_counts["correct"] + score_counts["incorrect"]
    proof_rows = [row for row in scores if row.get("format_status")]
    proof_pass = sum(row.get("format_status") == "format_pass" for row in proof_rows)
    routing_rows = [
        (score, result)
        for score, result in zip(scores, run_rows)
        if result and score.get("expected_domain") != "unknown"
    ]
    compatible_rows = [(score, result) for score, result in routing_rows if score.get("solver_compatible")]
    incompatible_rows = [(score, result) for score, result in routing_rows if not score.get("solver_compatible")]
    summary = {
        "total": len(dataset),
        "run_success": successful,
        "runtime_errors": errors,
        "empty_final_response": empty,
        "auto_scored": auto_scored,
        "auto_correct": score_counts["correct"],
        "automatic_accuracy": round(score_counts["correct"] / auto_scored, 6) if auto_scored else None,
        "manual_review": score_counts["manual_review"],
        "suspected_labels": score_counts["label_suspected"],
        "proof_format_pass_rate": round(proof_pass / len(proof_rows), 6) if proof_rows else None,
        "exact_routing_accuracy": (
            round(sum(bool(score.get("exact_domain_match")) for score, _ in routing_rows) / len(routing_rows), 6)
            if routing_rows
            else None
        ),
        "compatible_routing_rate": (
            round(len(compatible_rows) / len(routing_rows), 6) if routing_rows else None
        ),
        "unknown_rate": (
            round(sum(score.get("predicted_domain") == "unknown" for score, _ in routing_rows) / len(routing_rows), 6)
            if routing_rows
            else None
        ),
        "fallback_rate_when_compatible": (
            round(sum(bool(result.get("fallback_used")) for _, result in compatible_rows) / len(compatible_rows), 6)
            if compatible_rows
            else None
        ),
        "fallback_rate_when_incompatible": (
            round(sum(bool(result.get("fallback_used")) for _, result in incompatible_rows) / len(incompatible_rows), 6)
            if incompatible_rows
            else None
        ),
        "score_status_distribution": dict(sorted(score_counts.items())),
    }

    domains: dict[str, dict[str, Any]] = {}
    domain_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        domain_groups[score["predicted_domain"]].append(score)
    for domain, rows in sorted(domain_groups.items()):
        counts = Counter(row["score_status"] for row in rows)
        domain_auto = counts["correct"] + counts["incorrect"]
        domains[domain] = {
            "count": len(rows),
            "auto_scored": domain_auto,
            "correct": counts["correct"],
            "accuracy": round(counts["correct"] / domain_auto, 6) if domain_auto else None,
            "manual_review": counts["manual_review"],
            "runtime_error": counts["runtime_error"],
        }

    attempted = [row for row in run_rows if row]
    count = len(attempted)
    retries = [row for row in attempted if row.get("retry_triggered")]
    elapsed = [float(row.get("elapsed_seconds") or 0) for row in attempted]
    tool_counts = Counter(str(row["local_tool_name"]) for row in attempted if row.get("local_tool_name"))
    pipeline_acceptance_count = sum(
        bool(row.get("pipeline_acceptance_passed", row.get("final_acceptance_passed")))
        for row in attempted
    )
    deterministically_verified_count = sum(
        row.get("mathematical_verification_status") == "passed" for row in attempted
    )
    unverified_accepted_count = sum(
        bool(row.get("pipeline_acceptance_passed", row.get("final_acceptance_passed")))
        and row.get("mathematical_verification_status") in {"not_applicable", "unknown"}
        for row in attempted
    )
    deterministically_rejected_count = sum(
        row.get("mathematical_verification_status") == "failed" for row in attempted
    )
    reliability_counts = Counter(_result_reliability(row) for row in attempted)
    deterministic_override_count = sum(
        bool(row.get("deterministic_answer_override")) for row in attempted
    )
    manual_review_required_count = count - reliability_counts["verified"]
    summary.update(
        {
            "pipeline_acceptance_count": pipeline_acceptance_count,
            "deterministically_verified_count": deterministically_verified_count,
            "unverified_accepted_count": unverified_accepted_count,
            "deterministically_rejected_count": deterministically_rejected_count,
            "verified_count": reliability_counts["verified"],
            "unverified_count": reliability_counts["unverified"],
            "rejected_count": reliability_counts["rejected"],
            "unavailable_count": reliability_counts["unavailable"],
            "deterministic_override_count": deterministic_override_count,
            "manual_review_required_count": manual_review_required_count,
        }
    )
    behavior = {
        "local_tool_hit_rate": round(sum(tool_counts.values()) / count, 6) if count else 0.0,
        "local_tool_name_distribution": dict(sorted(tool_counts.items())),
        "retry_rate": round(len(retries) / count, 6) if count else 0.0,
        "retry_success_rate": (
            round(
                sum(
                    row.get("status") == "success"
                    and bool(row.get("pipeline_acceptance_passed", row.get("final_acceptance_passed")))
                    for row in retries
                )
                / len(retries),
                6,
            )
            if retries
            else None
        ),
        "pipeline_acceptance_rate": round(pipeline_acceptance_count / count, 6) if count else 0.0,
        "deterministically_verified_rate": round(deterministically_verified_count / count, 6) if count else 0.0,
        "unverified_answer_rate": round(unverified_accepted_count / count, 6) if count else 0.0,
        "deterministic_rejection_rate": round(deterministically_rejected_count / count, 6) if count else 0.0,
        "fallback_rate": round(sum(bool(row.get("fallback_used")) for row in attempted) / count, 6) if count else 0.0,
        "average_api_calls": round(mean(float(row.get("api_call_count") or 0) for row in attempted), 6) if count else 0.0,
        "average_elapsed_seconds": round(mean(elapsed), 6) if elapsed else 0.0,
        "p50_elapsed_seconds": _percentile(elapsed, 0.50),
        "p95_elapsed_seconds": _percentile(elapsed, 0.95),
    }
    return {
        "summary": summary,
        "by_predicted_domain": domains,
        "behavior": behavior,
        "risks": _risk_lists(dataset, results, scores),
        "items": scores,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    behavior = report["behavior"]
    lines = [
        "# Omni-MATH Intern-S API Evaluation Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        if key != "score_status_distribution":
            lines.append(f"| {key} | {value} |")
    lines.extend(["", "## By predicted domain", "", "| Domain | Count | Auto scored | Correct | Accuracy | Manual review | Runtime error |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for domain, values in report["by_predicted_domain"].items():
        lines.append(
            f"| {domain} | {values['count']} | {values['auto_scored']} | {values['correct']} | "
            f"{values['accuracy']} | {values['manual_review']} | {values['runtime_error']} |"
        )
    lines.extend(["", "## Behavior", "", "| Metric | Value |", "| --- | ---: |"])
    for key, value in behavior.items():
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
        lines.append(f"| {key} | {rendered} |")
    lines.extend(["", "## Risks", ""])
    for key, entries in report["risks"].items():
        lines.append(f"- {key}: {len(entries)}")
        for entry in entries:
            details = {name: value for name, value in entry.items() if name != "idx"}
            suffix = f" — {json.dumps(details, ensure_ascii=False)}" if details else ""
            lines.append(f"  - `{entry.get('idx', '')}`{suffix}")
    return "\n".join(lines) + "\n"


def score_files(
    dataset_path: str | Path,
    results_path: str | Path,
    output_json: str | Path,
    output_md: str | Path,
    output_enriched_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    dataset = load_jsonl(dataset_path)
    result_rows = load_jsonl(results_path)
    report = build_report(dataset, result_rows)
    json_path, md_path = _resolve(output_json), _resolve(output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    if output_enriched_jsonl is not None:
        enriched_path = _resolve(output_enriched_jsonl)
        if enriched_path.resolve() == _resolve(results_path).resolve():
            raise ValueError("--output-enriched-jsonl must not overwrite the original results")
        enriched_path.parent.mkdir(parents=True, exist_ok=True)
        score_map = {str(row.get("idx") or ""): row for row in report["items"]}
        with enriched_path.open("w", encoding="utf-8") as stream:
            for result in result_rows:
                score = score_map.get(str(result.get("idx") or ""), {})
                enriched = dict(result)
                for field in (
                    "score_status",
                    "score_reason",
                    "normalized_prediction",
                    "normalized_reference",
                    "pipeline_acceptance",
                    "pipeline_acceptance_passed",
                    "mathematical_verification_status",
                    "mathematical_verification_passed",
                    "mathematical_correctness",
                    "answer_reliability_status",
                    "manual_review_required",
                    "deterministic_answer_override",
                    "proof_review_required",
                    "proof_risk_signals",
                ):
                    enriched[field] = score.get(field)
                stream.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-enriched-jsonl", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = score_files(
        args.dataset,
        args.results,
        args.output_json,
        args.output_md,
        args.output_enriched_jsonl,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
