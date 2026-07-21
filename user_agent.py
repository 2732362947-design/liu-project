from __future__ import annotations

from fractions import Fraction
import inspect
import math
from typing import Any

from agents.answer_extractor_agent import extract_final_answer
import re

from agents.classifier_agent import classify_problem, domain_from_hint, solver_key_for_domain
from agents.planner_agent import make_plan
from agents.solver_agent import build_solver_prompt, normalize_solver_key
from agents.tools.combinatorics_counting_tool import solve_combinatorics_counting_problem
from agents.tools.combinatorics_graph_tool import solve_divisibility_subset_problem
from agents.tools.elementary_algebra_tool import solve_elementary_algebra_problem
from agents.tools.finite_field_tool import solve_finite_field_problem
from agents.tools.number_theory_tool import solve_number_theory_problem
from agents.tools.recurrence_sequence_tool import solve_recurrence_sequence_problem
from agents.verifier_agent import verify_solution


FALLBACK_RESPONSE = "未能得到可靠答案"
MAX_SOLVE_ATTEMPTS = 2
SHORT_ANSWER = "short_answer"
WORKED_SOLUTION = "worked_solution"
VERIFICATION_PASSED = "passed"
VERIFICATION_FAILED = "failed"
VERIFICATION_UNKNOWN = "unknown"
VERIFICATION_NOT_APPLICABLE = "not_applicable"
VERIFICATION_NOT_EVALUATED = "not_evaluated"
SUPPORTED_ANSWER_TYPES = {
    "number",
    "fraction",
    "equation",
    "multiple_roots",
    "matrix",
    "vector",
    "expression",
    "proof",
    "derivation",
    "explanation",
    "numeric_derivation",
    "unknown",
}
SENSITIVE_MARKERS = ("authorization", "bearer", "api_key", "token")
METADATA_DENYLIST = {
    "answer",
    "answer_type",
    "expected_answer",
    "expected_answer_type",
    "gold_answer",
    "reference_answer",
    "solution",
    "gold",
    "reference",
    "ground_truth",
    "expected",
    "expected_solution",
    "official_answer",
    "label",
    "target",
}
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
PRIMARY_SYSTEM_MESSAGE = (
    "Return only a polished mathematical answer for grading. "
    "The first characters of your response must be <final_solution> "
    "and the response must end with </final_solution>. "
    "Do not include planning, prompt discussion, identity discussion, "
    "draft notes, or text outside the tags."
)
RETRY_SYSTEM_MESSAGE = (
    "Your entire response must be one <final_solution>...</final_solution> "
    "block containing the polished mathematical solution."
)
SHORT_ANSWER_SYSTEM_MESSAGE = (
    "Give a concise exact derivation in one <final_solution>...</final_solution> block. "
    "Use at most 8 substantive steps. Avoid exhaustive case listings unless necessary. "
    "End with one boxed final answer."
)
SHORT_ANSWER_RETRY_SYSTEM_MESSAGE = (
    "Recompute independently. Give at most 8 concise exact steps. "
    "Do not repeat the conclusion. End once with one boxed final answer inside one "
    "<final_solution>...</final_solution> block."
)
# Kept as a public compatibility name; the protocol is now enforced as a system message.
FINAL_SOLUTION_PROTOCOL = PRIMARY_SYSTEM_MESSAGE
STRONG_META_REASONING_MARKERS = (
    "thinking process",
    "analyze the request",
    "deconstruct the problem",
    "drafting",
    "refining the output",
    "review against constraints",
    "self-correction",
    "final polish",
    "plan integration",
    "let's write",
    "ready to generate",
    "wait,",
    "i need to",
    "system instruction",
    "user prompt",
    "model identity",
    "intern-s1",
    "intern s1",
    "intern-s2-preview",
    "intern s2-preview",
    "persona",
    "role-play",
    "metadata.subject",
    "分析请求",
    "草拟答案",
    "自我修正",
    "检查约束",
    "提示词",
    "系统指令",
    "用户指令",
    "模型身份",
)
WEAK_META_REASONING_MARKERS = (
    "analysis:",
    "draft:",
    "planning notes",
    "self check",
    "constraint review",
    "prompt analysis",
    "response strategy",
)
LEGACY_SOLUTION_MARKER_RE = re.compile(
    r"(?im)^[ \t]*(最终解答|正式解答|最终答案|证明|结论|final\s+answer|final\s+solution|solution|conclusion)\s*[:：]\s*"
)


def _safe_text(value: Any, limit: int = 500) -> str:
    text = str(value or "")
    lowered = text.lower()
    for marker in SENSITIVE_MARKERS:
        if marker in lowered:
            return "[redacted]"
    return text[:limit]


def _trace(step: str, content: Any) -> dict:
    return {"step": step, "content": _safe_text(content, limit=1400)}


def _safe_snippet(value: Any, limit: int) -> str:
    return _safe_text(value, limit=limit)


def _model_output_trace(prefix: str, solution: str | None) -> str:
    text = str(solution or "")
    return f"status=success, {prefix}_chars={len(text)}"


def _supports_public_kwarg(callable_obj, name: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _model_call_event(
    step: str,
    prefix: str,
    solution: str | None,
    *,
    thinking_mode_requested: bool,
    thinking_mode_applied: bool,
) -> dict:
    event = _trace(step, _model_output_trace(prefix, solution))
    event["thinking_mode_requested"] = thinking_mode_requested
    event["thinking_mode_applied"] = thinking_mode_applied
    return event


def _final_answer_trace(
    extraction: dict,
    final_answer: str | None,
    extracted_answer_type: str | None,
    expected_answer_type: str | None,
) -> str:
    answer_text = str(final_answer or "")
    return (
        f"status={extraction.get('status')}, "
        f"extracted_answer_type={extracted_answer_type}, "
        f"expected_answer_type={expected_answer_type}, "
        f"meaningful_final={_is_meaningful_final_answer(final_answer)}, "
        f"final_answer_chars={len(answer_text)}, "
        f"has_final={bool(final_answer)}"
    )


def _response_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        return "".join(_response_content(item) for item in response)
    if isinstance(response, dict):
        if "reasoning" in str(response.get("type") or "").lower():
            return ""
        if "content" in response:
            return _response_content(response.get("content") or "")
        if response.get("text") is not None:
            return str(response["text"])
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            return _response_content(choices[0])
        message = response.get("message")
        if message is not None:
            return _response_content(message)
        if "reasoning_content" in response:
            return ""
    content = getattr(response, "content", None)
    if content is not None:
        return _response_content(content)
    choices = getattr(response, "choices", None)
    if choices:
        return _response_content(choices[0])
    message = getattr(response, "message", None)
    if message is not None:
        return _response_content(message)
    if getattr(response, "reasoning_content", None) is not None:
        return ""
    return str(response or "")


def _safe_metadata(metadata: dict | None) -> dict:
    if not isinstance(metadata, dict):
        return {}
    safe = {}
    for key, value in metadata.items():
        key_text = str(key)
        normalized_key = re.sub(r"[\s-]+", "_", key_text.strip().lower())
        if normalized_key in METADATA_DENYLIST:
            continue
        safe[key_text] = value
    return safe


def _determine_response_mode(
    problem: str,
    domain: str | None,
    expected_answer_type: str | None,
) -> str:
    text = str(problem or "")
    lowered = text.lower()
    chinese_worked_signals = (
        "推导",
        "证明",
        "验证",
        "检查",
        "说明",
        "解释",
        "论证",
        "写出过程",
        "计算并验证",
        "计算并说明",
        "求出并检查",
        "试证",
        "证得",
        "导出",
        "说明为什么",
        "解释为什么",
        "请说明",
        "给出证明",
        "证明下列",
    )
    english_worked_patterns = (
        r"\bprove\b",
        r"\bshow\s+that\b",
        r"\bderive\b",
        r"\bverify\b",
        r"\bcheck\b",
        r"\bexplain\b",
        r"\bjustify\b",
        r"\bexplain\s+why\b",
        r"\bcompute\s+and\s+verify\b",
        r"\bcalculate\s+and\s+explain\b",
        r"\bdemonstrate\b",
        r"\bgive\s+a\s+proof\b",
    )
    if any(signal in text for signal in chinese_worked_signals):
        return WORKED_SOLUTION
    if any(re.search(pattern, lowered) for pattern in english_worked_patterns):
        return WORKED_SOLUTION
    if str(domain or "").strip().lower() == "proof":
        return WORKED_SOLUTION
    if str(expected_answer_type or "").strip().lower() in {
        "proof",
        "derivation",
        "explanation",
        "numeric_derivation",
    }:
        return WORKED_SOLUTION
    return SHORT_ANSWER


def _expected_answer_type_from_problem(problem: str, response_mode: str | None = None) -> str:
    """Infer the judgeable answer shape from the problem text, without label metadata."""
    text = str(problem or "")
    lowered = text.lower()

    numeric_derivation_markers = (
        "newton's method",
        "newton method",
        "newton iteration",
        "牛顿法",
        "牛顿迭代",
    )
    if any(marker in lowered for marker in numeric_derivation_markers) and (
        re.search(r"\b(?:compute|calculate|iterate|derive)\b", lowered)
        or any(marker in text for marker in ("计算", "迭代", "推导"))
    ):
        return "numeric_derivation"

    proof_patterns = (
        r"\bprove\b",
        r"\bproof\b",
        r"\bshow\s+that\b",
        r"\bdemonstrate\b",
    )
    if any(token in text for token in ("证明", "试证", "证得", "给出证明")) or any(
        re.search(pattern, lowered) for pattern in proof_patterns
    ):
        return "proof"

    derivation_patterns = (r"\bderive\b", r"\bdeduce\b", r"\bderive\s+the\b")
    if any(token in text for token in ("推导", "导出", "写出过程")) or any(
        re.search(pattern, lowered) for pattern in derivation_patterns
    ):
        return "derivation"

    explanation_patterns = (r"\bexplain\b", r"\bjustify\b")
    if any(token in text for token in ("解释", "说明为什么", "论证")) or any(
        re.search(pattern, lowered) for pattern in explanation_patterns
    ):
        return "explanation"

    explicit_expression_patterns = (
        r"\bfind\s+(?:an|the)\s+expression\b",
        r"\bgive\s+(?:a|the)\s+(?:closed\s+)?formula\b",
        r"\bfind\s+(?:a|the)\s+closed\s+formula\b",
        r"\bfactor\s+(?:the|this|a)\s+polynomial\b",
        r"\bsimplify\s+(?:the|this|an?)\s+expression\b",
    )
    explicit_expression_markers = (
        "求表达式",
        "给出公式",
        "写出公式",
        "因式分解这个多项式",
        "因式分解该多项式",
        "化简表达式",
        "化简该表达式",
    )
    if any(re.search(pattern, lowered) for pattern in explicit_expression_patterns) or any(
        marker in text for marker in explicit_expression_markers
    ):
        return "expression"

    numeric_target_patterns = (
        r"\bhow\s+many\b",
        r"\bfind\s+the\s+value\b",
        r"\bcompute\s+the\s+value\b",
        r"\bfind\s+the\s+greatest\s+common\s+divisor\b",
        r"\bcompute\s+the\s+greatest\s+common\s+divisor\b",
        r"\bdetermine\s+the\s+gcd\b",
        r"\bfind\s+the\s+gcd\b",
        r"\bleast\s+common\s+multiple\b",
        r"\bfind\s+the\s+lcm\b",
        r"\bcompute\s+[mnk]\b",
        r"\bdetermine\s+[mnk]\b",
        r"\bwhat\s+is\s+the\s+(?:number|probability)\b",
        r"\bwhat\s+is\s+the\s+(?:least|greatest)\s+number\b",
    )
    if any(re.search(pattern, lowered) for pattern in numeric_target_patterns):
        return "number"

    matrix_markers = (
        "matrix",
        "矩阵",
        "transition matrix",
        "转移矩阵",
        "p^2",
        "p²",
        "normal equations",
        "正规方程",
        "design matrix",
        "设计矩阵",
    )
    if any(marker in lowered for marker in matrix_markers):
        return "matrix"

    if any(marker in lowered for marker in ("vector", "向量", "column vector", "row vector")):
        return "vector"
    if re.search(r"\b(?:all\s+)?roots\b|\bsolutions\b", lowered) or any(
        marker in text for marker in ("所有根", "全部解", "多个根")
    ):
        return "multiple_roots"
    if any(marker in lowered for marker in ("fraction", "rational number", "分数", "最简分数")):
        return "fraction"
    if re.search(r"\b(?:solve|solution\s+of)\b.*\bequation\b|\bequation\b.*\bsolve\b", lowered) or any(
        marker in text for marker in ("解方程", "方程的解")
    ):
        return "equation"
    if re.search(
        r"\b(?:compute|calculate|evaluate)\b|\bfind\s+(?:the\s+)?(?:number|value|integer|minimum|maximum|smallest|largest)\b",
        lowered,
    ) or any(
        marker in text for marker in ("计算", "求值", "求出", "多少", "最小整数", "最大整数")
    ):
        return "number"
    if response_mode == WORKED_SOLUTION:
        return "unknown"
    return "unknown"


def _has_meta_reasoning_leak(text: str) -> bool:
    normalized = str(text or "").lower()
    if any(marker in normalized for marker in STRONG_META_REASONING_MARKERS):
        return True
    weak_hits = sum(marker in normalized for marker in WEAK_META_REASONING_MARKERS)
    return weak_hits >= 2


def _extract_judgeable_solution(
    raw_solution: str,
    *,
    response_mode: str,
) -> str | None:
    candidate, _ = _extract_judgeable_solution_with_diagnostics(raw_solution, response_mode=response_mode)
    return candidate


def _extract_judgeable_solution_with_diagnostics(
    raw_solution: str,
    *,
    response_mode: str,
) -> tuple[str | None, dict[str, Any]]:
    text = str(raw_solution or "").strip()
    details: dict[str, Any] = {
        "subreason": None,
        "raw_chars": len(str(raw_solution or "")),
        "clean_candidate_chars": 0,
        "tag_status": "absent",
        "latex_balance_status": "unknown",
        "ending_status": "unknown",
        "ending_reason": "not_checked",
        "compatibility_recovery": None,
        "compatibility_recovered": False,
    }
    if not text:
        details["subreason"] = "no_clean_candidate"
        details["tag_status"] = "absent"
        return None, details

    opening_tags = re.findall(r"<final_solution>", text, flags=re.IGNORECASE)
    closing_tags = re.findall(r"</final_solution>", text, flags=re.IGNORECASE)
    tagged_blocks = list(
        re.finditer(r"<final_solution>(.*?)</final_solution>", text, flags=re.IGNORECASE | re.DOTALL)
    )
    if opening_tags or closing_tags:
        if len(opening_tags) == 1 and not closing_tags:
            opening_match = re.search(r"<final_solution>", text, flags=re.IGNORECASE)
            prefix = text[: opening_match.start()].strip() if opening_match else text
            body = text[opening_match.end() :].strip() if opening_match else ""
            details["tag_status"] = "opening_only"
            details["subreason"] = "missing_closing_tag"
            if not prefix and _candidate_passes_recovery_gate(body, response_mode):
                details["clean_candidate_chars"] = len(body)
                details["tag_status"] = "opening_only_recovered"
                details["compatibility_recovery"] = "opening_only"
                details["compatibility_recovered"] = True
                return body, details
            return None, details
        if closing_tags and not opening_tags:
            closing_match = re.search(r"</final_solution>", text, flags=re.IGNORECASE)
            body = text[: closing_match.start()].strip() if closing_match else ""
            suffix = text[closing_match.end() :].strip() if closing_match else text
            details["tag_status"] = "closing_only"
            details["subreason"] = "missing_opening_tag"
            if not suffix and _candidate_passes_recovery_gate(body, response_mode):
                details["clean_candidate_chars"] = len(body)
                details["tag_status"] = "closing_only_recovered"
                details["compatibility_recovery"] = "closing_only"
                details["compatibility_recovered"] = True
                return body, details
            return None, details
        if len(opening_tags) != len(closing_tags) or len(tagged_blocks) != len(opening_tags):
            details["tag_status"] = "invalid"
            details["subreason"] = "no_clean_candidate"
            return None, details
        block = tagged_blocks[-1].group(1).strip() if tagged_blocks else ""
        if not block or re.search(r"</?final_solution>", block, flags=re.IGNORECASE):
            details["tag_status"] = "invalid"
            details["subreason"] = "no_clean_candidate"
            return None, details
        details["tag_status"] = "complete"
        details["clean_candidate_chars"] = len(block)
        return block, details

    if _has_meta_reasoning_leak(text):
        markers = list(LEGACY_SOLUTION_MARKER_RE.finditer(text))
        if markers:
            marker = markers[-1]
            body = text[marker.end() :].strip()
            marker_context = text[max(0, marker.start() - 160) : marker.start()].lower()
            preceding_line = marker_context.splitlines()[-1].strip() if marker_context.splitlines() else ""
            unsafe_section_context = bool(
                re.search(r"\b(?:draft|example|plan|template|sample)\b|草稿|示例|计划", preceding_line)
            )
            if not unsafe_section_context and _candidate_passes_recovery_gate(body, response_mode):
                details["clean_candidate_chars"] = len(body)
                details["tag_status"] = "absent"
                details["compatibility_recovery"] = "final_section"
                details["compatibility_recovered"] = True
                return body, details
        details["subreason"] = "no_clean_candidate"
        return None, details

    markers = list(LEGACY_SOLUTION_MARKER_RE.finditer(text))
    if markers:
        marker = markers[-1]
        body = text[marker.end() :].strip()
        if not body or _has_meta_reasoning_leak(body):
            details["subreason"] = "no_clean_candidate"
            return None, details
        marker_name = marker.group(1).lower().replace(" ", "")
        prefix = text[: marker.start()].strip()
        if response_mode == WORKED_SOLUTION and prefix and marker_name in {"最终答案", "finalanswer"}:
            details["clean_candidate_chars"] = len(text)
            return text, details
        details["clean_candidate_chars"] = len(body)
        return body, details

    details["clean_candidate_chars"] = len(text)
    return text, details


def _candidate_passes_recovery_gate(candidate: str, response_mode: str) -> bool:
    text = str(candidate or "").strip()
    if not text or _has_meta_reasoning_leak(text):
        return False
    if _latex_balance_subreason(text) is not None:
        return False
    ending_status, _ = _ending_diagnostics(text)
    if ending_status == "suspicious":
        return False
    if response_mode == WORKED_SOLUTION:
        return _validate_worked_solution_quality(text, None)[0]
    return _validate_short_solution_quality(text, None)[0]


def _latex_balance_subreason(text: str) -> str | None:
    unescaped_dollars = len(re.findall(r"(?<!\\)\$", text))
    if unescaped_dollars % 2:
        return "unbalanced_dollar"
    if text.count(r"\[") != text.count(r"\]") or text.count(r"\(") != text.count(r"\)"):
        return "unbalanced_dollar"
    begin_counts: dict[str, int] = {}
    end_counts: dict[str, int] = {}
    for environment in re.findall(r"\\begin\{([^{}]+)\}", text):
        begin_counts[environment] = begin_counts.get(environment, 0) + 1
    for environment in re.findall(r"\\end\{([^{}]+)\}", text):
        end_counts[environment] = end_counts.get(environment, 0) + 1
    if begin_counts != end_counts:
        return "unbalanced_environment"
    depth = 0
    for index, character in enumerate(text):
        if index and text[index - 1] == "\\":
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return "unbalanced_braces"
    return None if depth == 0 else "unbalanced_braces"


def _latex_is_balanced(text: str) -> bool:
    return _latex_balance_subreason(text) is None


def _has_incomplete_ending(text: str) -> bool:
    status, _ = _ending_diagnostics(text)
    return status == "suspicious"


def _ending_subreason(text: str) -> str | None:
    status, reason = _ending_diagnostics(text)
    return reason if status == "suspicious" else None


def _ending_diagnostics(text: str) -> tuple[str, str]:
    stripped = str(text or "").rstrip()
    if not stripped:
        return "unknown", "empty_candidate"
    if stripped.count("```") % 2:
        return "suspicious", "unclosed_code_fence"

    lowered = stripped.lower()
    trailing_connector_patterns = (
        r"\b(?:because|since|where|such\s+that)\s*$",
        r"\btherefore\s*[,;:]\s*$",
        r"\b(?:the\s+correct\s+answer\s+is|thus\s+the\s+answer\s+is|the\s+answer\s+is|we\s+obtain|which\s+gives)\s*$",
        r"\b(?:hence|therefore)\s*[,，:]?\s*$",
        r"\b(?:wait|let['’]?s)\s*,?\s*$",
        r"\bi\s+(?:need|should)\s+to\s*$",
        r"(?:因为|所以有|其中|即|从而有)\s*$",
    )
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in trailing_connector_patterns):
        return "suspicious", "trailing_connector"

    operator_tail = re.sub(r"(?:\$\$|\$|\\\]|\\\))\s*$", "", stripped).rstrip()
    if re.search(
        r"(?:=|\+|-|\\cdot|\\times|\\frac\s*\{?|\\begin\s*\{[^}]*\})\s*$",
        operator_tail,
        flags=re.IGNORECASE,
    ):
        return "suspicious", "trailing_operator"
    if re.search(r"[,，:：]\s*$", stripped):
        return "suspicious", "trailing_punctuation"

    compact = stripped.rstrip("。.!！")
    if re.search(r"(?:[A-Za-z](?:_[A-Za-z0-9{}]+)?|\\[A-Za-z]+(?:_[A-Za-z0-9{}]+)?)$", compact):
        return "complete", "ends_with_variable"
    if re.search(r"\d$", compact):
        return "complete", "ends_with_number"
    if re.search(r"(?:\)|\]|\}|\$|\\\]|\\\))$", compact):
        return "complete", "ends_with_closed_formula"
    if re.search(
        r"(?:故证|证毕|结论成立|thus\s+the\s+result\s+follows|thus\s+the\s+convergence\s+is\s+not\s+uniform)$",
        compact,
        flags=re.IGNORECASE,
    ):
        return "complete", "ends_with_conclusion"
    return "complete", "ends_with_text"


def _invalid_extracted_conclusion(extracted_answer: str | None) -> bool:
    if extracted_answer is None:
        return False
    answer = str(extracted_answer).strip()
    lowered = answer.lower()
    if not _is_meaningful_final_answer(answer) or _has_meta_reasoning_leak(answer) or _has_incomplete_ending(answer):
        return True
    invalid_fragments = (
        "或定理",
        "or theorem",
        "looking at the prompt",
        "i should",
        "i need to",
        "wait",
        "final answer should",
    )
    invalid_exact = {"therefore", "thus", "hence", "because", "其中", "即"}
    normalized = re.sub(r"[\s。.!！,，;；:：]+", "", lowered)
    return normalized in invalid_exact or any(fragment in lowered for fragment in invalid_fragments)


def _validate_worked_solution_quality(
    solution: str,
    extracted_answer: str | None,
) -> tuple[bool, str]:
    text = str(solution or "").strip()
    if not text:
        return False, "missing_judgeable_solution"
    if _has_meta_reasoning_leak(text):
        return False, "meta_reasoning_leak"
    if not _latex_is_balanced(text):
        return False, "unbalanced_latex"
    if _has_incomplete_ending(text):
        return False, "incomplete_output"
    if _invalid_extracted_conclusion(extracted_answer):
        return False, "invalid_extracted_conclusion"

    normalized = re.sub(r"[\s。.!！,，;；:：]+", "", text.lower())
    if normalized in {"thusnotuniform", "命题成立", "结论成立", "得证", "证毕"}:
        return False, "worked_solution_too_short"

    chunks = [
        chunk.strip()
        for chunk in re.split(r"\n+|(?<=[。！？.!?;；])\s*", text)
        if chunk.strip()
    ]
    substantive_chunks = [
        chunk
        for chunk in chunks
        if len(re.sub(r"\s+", "", chunk)) >= 10 and re.search(r"[0-9A-Za-z\u4e00-\u9fff\\=]", chunk)
    ]
    step_markers = re.findall(
        r"\b(?:first|second|then|since|because|therefore|thus|hence|let|assume)\b|"
        r"首先|其次|第一步|第二步|然后|因为|所以|因此|故|于是|任取|假设|从定义",
        text,
        flags=re.IGNORECASE,
    )
    if len(substantive_chunks) >= 2:
        return True, "passed"
    if len(re.sub(r"\s+", "", text)) >= 24 and len(step_markers) >= 2:
        return True, "passed"
    return False, "worked_solution_too_short"


def _normalize_repetition_segment(segment: str) -> str:
    lowered = str(segment or "").lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff\\{}]+", "", lowered)


def _short_repetition_subreason(text: str, raw_chars: int | None = None) -> str | None:
    candidate = str(text or "").strip()
    lowered = candidate.lower().rstrip()
    incomplete_suffix_patterns = (
        r"\bthe\s+correct\s+answer\s+is\s*$",
        r"\bthus\s+the\s+answer\s+is\s*$",
        r"\bthe\s+answer\s+is\s*$",
        r"\bwe\s+obtain\s*$",
        r"\bwhich\s+gives\s*$",
        r"\b(?:therefore|hence)\s*[,，:]?\s*$",
        r"\\boxed\s*\{\s*$",
    )
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in incomplete_suffix_patterns):
        return "incomplete_repeated_suffix"

    separated = re.sub(r"(\$\$.*?\$\$|\\\[.*?\\\])", r"\n\1\n", candidate, flags=re.DOTALL)
    raw_segments = [
        segment.strip()
        for segment in re.split(r"\n+|[.!?。！？;；]+\s*", separated)
        if segment.strip()
    ]
    substantive: list[tuple[str, bool]] = []
    for segment in raw_segments:
        normalized = _normalize_repetition_segment(segment)
        if len(normalized) < 20:
            continue
        is_final = bool(
            re.search(
                r"\\boxed|\b(?:final|correct)\s+answer\b|\bthe\s+answer\s+is\b|最终答案|答案(?:是|为)",
                segment,
                flags=re.IGNORECASE,
            )
        )
        substantive.append((normalized, is_final))

    for index in range(len(substantive) - 2):
        window = substantive[index : index + 3]
        if len({item[0] for item in window}) == 1:
            return "repeated_final_answer_loop" if any(item[1] for item in window) else "repeated_sentence_loop"

    counts: dict[str, int] = {}
    final_segments: set[str] = set()
    for normalized, is_final in substantive:
        counts[normalized] = counts.get(normalized, 0) + 1
        if is_final:
            final_segments.add(normalized)
    if any(counts.get(segment, 0) >= 3 for segment in final_segments):
        return "repeated_final_answer_loop"

    tail = substantive[-20:]
    if len(tail) >= 6:
        tail_counts: dict[str, int] = {}
        for normalized, _ in tail:
            tail_counts[normalized] = tail_counts.get(normalized, 0) + 1
        dominant = max(tail_counts.values(), default=0)
        if dominant >= 3 and dominant / len(tail) >= 0.5:
            dominant_segment = max(tail_counts, key=tail_counts.get)
            return (
                "repeated_final_answer_loop"
                if dominant_segment in final_segments
                else "repeated_sentence_loop"
            )

    measured_chars = max(len(candidate), int(raw_chars or 0))
    if measured_chars > 4000:
        tokens = re.findall(r"\\[A-Za-z]+|[A-Za-z]+|\d+|[\u4e00-\u9fff]", lowered)
        repetition_rates: list[float] = []
        for size in (5, 8):
            ngrams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
            if len(ngrams) >= 40:
                repetition_rates.append((len(ngrams) - len(set(ngrams))) / len(ngrams))
        if repetition_rates and max(repetition_rates) >= 0.55:
            return "high_ngram_repetition"
    return None


def _validate_short_solution_quality(
    solution: str,
    extracted_answer: str | None,
) -> tuple[bool, str]:
    text = str(solution or "").strip()
    if not text:
        return False, "missing_judgeable_solution"
    if _has_meta_reasoning_leak(text):
        return False, "meta_reasoning_leak"
    if _short_repetition_subreason(text, len(text)) is not None:
        return False, "repetitive_output"
    if not _latex_is_balanced(text):
        return False, "unbalanced_latex"
    if _has_incomplete_ending(text):
        return False, "incomplete_output"
    if _invalid_extracted_conclusion(extracted_answer):
        return False, "invalid_extracted_conclusion"
    return True, "passed"


def _validate_output_quality(
    solution: str | None,
    extracted_answer: str | None,
    response_mode: str,
) -> tuple[bool, str]:
    if solution is None:
        return False, "missing_judgeable_solution"
    if response_mode == WORKED_SOLUTION:
        return _validate_worked_solution_quality(solution, extracted_answer)
    return _validate_short_solution_quality(solution, extracted_answer)


def _assess_output_quality(
    raw_solution: str,
    solution: str | None,
    extracted_answer: str | None,
    response_mode: str,
    extraction_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details = dict(extraction_details or {})
    details.setdefault("subreason", None)
    details.setdefault("raw_chars", len(str(raw_solution or "")))
    details.setdefault("clean_candidate_chars", len(str(solution or "")))
    details.setdefault("tag_status", "absent")
    details.setdefault("ending_reason", "not_checked")
    details.setdefault("compatibility_recovery", None)
    details.setdefault("compatibility_recovered", False)
    raw_text = str(raw_solution or "")
    clean_text = str(solution or "").strip()

    if solution is None:
        if _has_meta_reasoning_leak(raw_text):
            details["subreason"] = "no_clean_candidate"
            reason = "meta_reasoning_leak"
        elif details.get("tag_status") in {"opening_only", "closing_only", "invalid"}:
            reason = "incomplete_output"
        else:
            details["subreason"] = details.get("subreason") or "no_clean_candidate"
            reason = "missing_judgeable_solution"
        details["latex_balance_status"] = "unknown"
        details["ending_status"] = "unknown"
        details["ending_reason"] = "no_clean_candidate"
        return {"passed": False, "reason": reason, "details": details}

    if _has_meta_reasoning_leak(clean_text):
        details["subreason"] = "no_clean_candidate"
        details["latex_balance_status"] = "unknown"
        details["ending_status"] = "unknown"
        details["ending_reason"] = "meta_reasoning_leak"
        return {"passed": False, "reason": "meta_reasoning_leak", "details": details}

    if response_mode == SHORT_ANSWER:
        repetition_subreason = _short_repetition_subreason(
            clean_text, int(details.get("raw_chars") or len(raw_text))
        )
        if repetition_subreason is not None:
            details["subreason"] = repetition_subreason
            details["ending_status"] = (
                "suspicious" if repetition_subreason == "incomplete_repeated_suffix" else "complete"
            )
            details["ending_reason"] = repetition_subreason
            details["latex_balance_status"] = "unknown"
            return {"passed": False, "reason": "repetitive_output", "details": details}

    latex_subreason = _latex_balance_subreason(clean_text)
    details["latex_balance_status"] = "passed" if latex_subreason is None else "failed"
    if latex_subreason is not None:
        details["subreason"] = latex_subreason
        details["ending_status"] = "suspicious"
        details["ending_reason"] = "unclosed_formula"
        return {"passed": False, "reason": "unbalanced_latex", "details": details}

    ending_status, ending_reason = _ending_diagnostics(clean_text)
    details["ending_status"] = ending_status
    details["ending_reason"] = ending_reason
    if ending_status == "suspicious":
        details["subreason"] = "truncated_sentence"
        return {"passed": False, "reason": "incomplete_output", "details": details}

    if response_mode == SHORT_ANSWER and not _is_meaningful_final_answer(extracted_answer):
        details["subreason"] = "no_clean_candidate"
        return {"passed": False, "reason": "missing_judgeable_solution", "details": details}
    if _invalid_extracted_conclusion(extracted_answer):
        details["subreason"] = "no_clean_candidate"
        return {"passed": False, "reason": "missing_judgeable_solution", "details": details}

    passed, reason = (
        _validate_worked_solution_quality(clean_text, extracted_answer)
        if response_mode == WORKED_SOLUTION
        else _validate_short_solution_quality(clean_text, extracted_answer)
    )
    if not passed and reason == "worked_solution_too_short":
        details["subreason"] = "too_short_for_worked_solution"
    elif passed and not details.get("compatibility_recovered"):
        details["subreason"] = None
    return {"passed": passed, "reason": reason, "details": details}


def _quality_trace_content(quality: dict[str, Any]) -> str:
    details = quality.get("details") if isinstance(quality.get("details"), dict) else {}
    return (
        f"status={'passed' if quality.get('passed') else 'failed'}, "
        f"reason={quality.get('reason')}, "
        f"subreason={details.get('subreason')}, "
        f"raw_chars={details.get('raw_chars')}, "
        f"clean_candidate_chars={details.get('clean_candidate_chars')}, "
        f"tag_status={details.get('tag_status')}, "
        f"latex_balance_status={details.get('latex_balance_status')}, "
        f"ending_status={details.get('ending_status')}, "
        f"ending_reason={details.get('ending_reason')}, "
        f"compatibility_recovery={details.get('compatibility_recovery')}, "
        f"compatibility_recovered={bool(details.get('compatibility_recovered'))}"
    )


def _final_acceptance_passed(output_quality: dict[str, Any], verification: dict | None) -> bool:
    return _final_acceptance_passed_for_mode(output_quality, verification, SHORT_ANSWER)


def _final_acceptance_passed_for_mode(
    output_quality: dict[str, Any],
    verification: dict | None,
    response_mode: str,
) -> bool:
    if not output_quality.get("passed") or not isinstance(verification, dict):
        return False
    status = str(verification.get("status") or "").lower()
    if response_mode == SHORT_ANSWER:
        return status in {VERIFICATION_PASSED, VERIFICATION_NOT_APPLICABLE, VERIFICATION_UNKNOWN}
    return status in {VERIFICATION_PASSED, VERIFICATION_UNKNOWN, VERIFICATION_NOT_APPLICABLE}


def _reliability_fields(
    verification: dict[str, Any] | None,
    pipeline_acceptance_passed: bool,
    output_quality_passed: bool | None = None,
) -> dict[str, Any]:
    status = str((verification or {}).get("status") or VERIFICATION_NOT_EVALUATED).lower()
    if output_quality_passed is None:
        # Compatibility for direct callers predating the explicit quality field.
        output_quality_passed = bool(pipeline_acceptance_passed)
    if status == VERIFICATION_PASSED:
        reliability = "verified"
        manual_review_required = False
    elif status == VERIFICATION_FAILED:
        reliability = "rejected"
        manual_review_required = True
    elif output_quality_passed and status in {
        VERIFICATION_NOT_APPLICABLE,
        VERIFICATION_UNKNOWN,
    }:
        reliability = "unverified"
        manual_review_required = True
    else:
        reliability = "unavailable"
        manual_review_required = True
    return {
        "pipeline_acceptance_passed": bool(pipeline_acceptance_passed),
        "mathematical_verification_status": status,
        "mathematical_verification_reason": (verification or {}).get("reason"),
        "mathematical_verification_subreason": (
            (verification or {}).get("verification_subreason")
            or (verification or {}).get("subreason")
        ),
        "mathematical_verification_severity": (verification or {}).get("severity"),
        "mathematical_verification_passed": status == VERIFICATION_PASSED,
        "answer_reliability_status": reliability,
        "manual_review_required": manual_review_required,
    }


def _deterministic_override_value(
    first_verification: dict[str, Any] | None,
    retry_verification: dict[str, Any] | None,
) -> tuple[int, str] | None:
    if not isinstance(first_verification, dict) or not isinstance(retry_verification, dict):
        return None
    allowed_verifiers = {"integer_polynomial_value_gcd", "factorial_floor_exact"}
    first_name = str(first_verification.get("verifier_name") or "")
    retry_name = str(retry_verification.get("verifier_name") or "")
    if first_name != retry_name or retry_name not in allowed_verifiers:
        return None
    for verification in (first_verification, retry_verification):
        if (
            str(verification.get("status") or "") != VERIFICATION_FAILED
            or verification.get("verifier_applicable") is not True
            or verification.get("problem_parse_status") != VERIFICATION_PASSED
            or verification.get("candidate_parse_status") != VERIFICATION_PASSED
            or verification.get("deterministic_override_eligible") is not True
            or verification.get("exact_unique_answer") is not True
        ):
            return None
    first_value = first_verification.get("_deterministic_answer_value")
    retry_value = retry_verification.get("_deterministic_answer_value")
    if not isinstance(first_value, int) or first_value != retry_value:
        return None
    return first_value, retry_name


def _clean_candidate_tail(candidate: str | None, output_quality: dict[str, Any]) -> str | None:
    text = str(candidate or "").strip()
    if not output_quality.get("passed") or not text or _has_meta_reasoning_leak(text):
        return None
    compact = re.sub(r"\s+", " ", text).strip()
    tail = compact[-160:]
    safe_tail = _safe_text(tail, limit=160)
    return None if safe_tail == "[redacted]" else safe_tail


def _diagnostic_excerpt(value: Any, limit: int, *, tail: bool = False) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text or _has_meta_reasoning_leak(text):
        return None
    excerpt = text[-limit:] if tail else text[:limit]
    safe_excerpt = _safe_text(excerpt, limit=limit)
    return None if safe_excerpt == "[redacted]" else safe_excerpt


def _attempt_diagnostics(
    *,
    output_quality: dict[str, Any],
    extraction: dict[str, Any],
    candidate: str | None,
    final_answer: str | None,
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    details = output_quality.get("details") if isinstance(output_quality.get("details"), dict) else {}
    verification = verification if isinstance(verification, dict) else {}
    return {
        "raw_chars": int(details.get("raw_chars") or 0),
        "tag_status": str(details.get("tag_status") or "unknown"),
        "clean_candidate_chars": int(details.get("clean_candidate_chars") or 0),
        "quality_reason": str(output_quality.get("reason") or "unknown"),
        "quality_subreason": details.get("subreason"),
        "latex_balance_status": str(details.get("latex_balance_status") or "unknown"),
        "ending_status": str(details.get("ending_status") or "unknown"),
        "extracted_answer_type": str(extraction.get("answer_type") or "unknown"),
        "extracted_answer_summary": _diagnostic_excerpt(final_answer, 120),
        "candidate_tail": _diagnostic_excerpt(candidate, 200, tail=True),
        "verification_reason": str(verification.get("reason") or "unknown"),
        "verification_subreason": verification.get("verification_subreason") or verification.get("subreason"),
        "verifier_name": verification.get("verifier_name"),
        "verifier_applicable": verification.get("verifier_applicable"),
        "problem_parse_status": str(verification.get("problem_parse_status") or "unknown"),
        "candidate_parse_status": str(verification.get("candidate_parse_status") or "unknown"),
        "computed_value_summary": _diagnostic_excerpt(verification.get("computed_value_summary"), 120),
        "candidate_value_summary": _diagnostic_excerpt(verification.get("candidate_value_summary"), 120),
    }


def _has_clear_final_conclusion(solution: str, extracted_answer: str | None) -> bool:
    tail = solution[-800:]
    conclusion_markers = (
        "最终答案",
        "最终结论",
        "结论是",
        "因此",
        "所以",
        "故",
        "从而",
        "命题成立",
        "得证",
        "therefore",
        "thus",
        "hence",
        "we conclude",
        "final answer",
    )
    if any(marker in tail.lower() for marker in conclusion_markers):
        return True
    answer = str(extracted_answer or "").strip()
    nonempty_lines = [line.strip() for line in solution.splitlines() if line.strip()]
    if not answer or not nonempty_lines:
        return False
    last_line = nonempty_lines[-1].strip("$ ")
    compact_last = re.sub(r"[\s。.!！]", "", last_line)
    compact_answer = re.sub(r"[\s。.!！]", "", answer)
    return compact_last in {compact_answer, rf"\boxed{{{compact_answer}}}"}


def _compose_final_response(
    *,
    problem: str,
    response_mode: str,
    solution: str | None,
    extracted_answer: str | None,
    verification: dict | None,
) -> str:
    del problem  # 保留统一接口，当前合成规则不需要再次解析题面。
    verified_answer = _verification_allows_final_answer(extracted_answer, verification, response_mode)
    answer_text = str(extracted_answer or "").strip()
    quality_passed, _ = _validate_output_quality(solution, extracted_answer, response_mode)
    if response_mode == SHORT_ANSWER:
        return answer_text if verified_answer and quality_passed else FALLBACK_RESPONSE
    if verified_answer and quality_passed:
        solution_text = str(solution or "").strip()
        if answer_text and not _has_clear_final_conclusion(solution_text, extracted_answer):
            solution_text = f"{solution_text}\n\n最终结论：{answer_text}"
        return solution_text
    return FALLBACK_RESPONSE


def _domain_from_metadata(safe_metadata: dict | None) -> str | None:
    if not isinstance(safe_metadata, dict):
        return None
    for key in ("domain", "subject", "category"):
        domain = domain_from_hint(safe_metadata.get(key))
        if domain:
            return domain
    return None


def _solver_key_from_domain(domain: str | None, solver_key: str | None = None) -> str:
    if solver_key:
        return normalize_solver_key(str(solver_key), domain)
    return solver_key_for_domain(str(domain or ""))


def _problem_suggests_extremal_discrete(problem: str | None) -> bool:
    text = str(problem or "").lower()
    subset_markers = ("k-element subset", "every k-element subset", "subset")
    structure_markers = (
        "contains two distinct elements",
        "two distinct elements",
        "pair of elements",
        "divides",
        "positive integer",
        "integer",
        "{1,2,...",
        "{1, 2, ...",
        "independent set",
        "coloring",
        "tournament",
        "choose",
    )
    return any(marker in text for marker in subset_markers) and any(marker in text for marker in structure_markers)


def _apply_metadata_domain(classification: dict, safe_metadata: dict | None, problem: str | None = None) -> dict:
    updated = dict(classification or {})
    if str(updated.get("domain") or "unknown") != "unknown":
        return updated
    domain = _domain_from_metadata(safe_metadata)
    if not domain:
        return updated
    updated["domain"] = domain
    updated["solver_key"] = _solver_key_from_domain(domain)
    updated["reason"] = "题面无明确领域强信号，使用安全 metadata 的弱提示进行领域路由。"
    updated["routing_confidence"] = "low"
    updated["matched_signal_categories"] = ["metadata_hint"]
    updated["runner_up_domain"] = None
    updated["score_margin"] = 0
    return updated


def _expected_answer_type_from_metadata(safe_metadata: dict | None) -> str | None:
    if not isinstance(safe_metadata, dict):
        return None
    answer_type = str(safe_metadata.get("answer_type") or "").strip().lower()
    return answer_type if answer_type in SUPPORTED_ANSWER_TYPES else None


def _build_prompt_constraints(
    problem: str,
    solver_key: str,
    expected_answer_type: str | None,
    response_mode: str,
) -> str:
    constraints = []
    if response_mode == SHORT_ANSWER:
        constraints.extend(
            (
                "Give a concise exact derivation.",
                "Use at most 8 substantive steps and avoid exhaustive case listings unless necessary.",
                "End with one boxed final answer.",
            )
        )
        if solver_key == "number_theory":
            constraints.append(
                "Prefer a general argument, the necessary residue or valuation sets, and the final computation; "
                "do not enumerate every modular residue individually."
            )
    if str(expected_answer_type or "").lower() == "expression":
        constraints.append(
            "最终答案必须是表达式，不要用单个数字作为占位答案；"
            "除非题目中的表达式确实化简为常数，否则不要只回答裸数字。"
            "适用时请使用题目中的变量。"
        )
    if solver_key == "discrete" and _problem_suggests_extremal_discrete(problem):
        constraints.append(
            "针对该组合极值 / 图建模题：不要完整枚举所有边或邻接表；"
            "请用结构分组、参数族和极值集合证明，控制解答长度。"
        )
    if not constraints:
        return ""
    return (
        "\n\n【输出格式与长度约束】\n"
        + "\n".join(f"{index + 1}. {item}" for index, item in enumerate(constraints))
    )


def _append_prompt_constraints(
    prompt: str,
    problem: str,
    solver_key: str,
    expected_answer_type: str | None,
    response_mode: str,
) -> str:
    constraint_text = _build_prompt_constraints(problem, solver_key, expected_answer_type, response_mode)
    if not constraint_text:
        return prompt
    return f"{prompt}{constraint_text}"


def _is_meaningful_final_answer(answer: str | None) -> bool:
    if answer is None:
        return False
    text = str(answer).strip()
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


def _retry_reasons(
    final_answer: str | None,
    verification: dict | None,
    output_quality: tuple[bool, str] | dict[str, Any] | None = None,
) -> list[str]:
    del final_answer
    reasons: list[str] = []
    quality_passed = True
    quality_reason = "passed"
    if isinstance(output_quality, dict):
        quality_passed = bool(output_quality.get("passed"))
        quality_reason = str(output_quality.get("reason") or "missing_judgeable_solution")
    elif output_quality is not None:
        quality_passed, quality_reason = bool(output_quality[0]), str(output_quality[1])
    retryable_quality_reasons = {
        "meta_reasoning_leak",
        "incomplete_output",
        "unbalanced_latex",
        "worked_solution_too_short",
        "missing_judgeable_solution",
        "repetitive_output",
    }
    if not quality_passed and quality_reason in retryable_quality_reasons:
        reasons.append(quality_reason)
    if not isinstance(verification, dict):
        return reasons
    status = str(verification.get("status", "")).lower()
    if status == VERIFICATION_FAILED:
        reasons.append("mathematical_verification_failed")
    return reasons


def _primary_retry_reason(output_quality: dict[str, Any], verification: dict | None) -> str:
    if not output_quality.get("passed"):
        reason = str(output_quality.get("reason") or "missing_judgeable_solution")
        details = output_quality.get("details") if isinstance(output_quality.get("details"), dict) else {}
        subreason = str(details.get("subreason") or "")
        ending_reason = str(details.get("ending_reason") or "")
        if subreason in {"missing_closing_tag", "missing_opening_tag"}:
            return subreason
        if reason == "unbalanced_latex" or ending_reason in {"unclosed_formula", "trailing_operator"}:
            return "truncated_formula"
        if subreason == "truncated_sentence":
            return "truncated_sentence"
        return reason
    if isinstance(verification, dict) and str(verification.get("status") or "").lower() == VERIFICATION_FAILED:
        return str(verification.get("subreason") or "mathematical_verification_failed")
    return "missing_judgeable_solution"


def _should_retry(
    final_answer: str | None,
    verification: dict | None,
    output_quality: tuple[bool, str] | dict[str, Any] | None = None,
) -> bool:
    return bool(_retry_reasons(final_answer, verification, output_quality))


def _verification_allows_final_answer(
    final_answer: str | None,
    verification: dict | None,
    response_mode: str = SHORT_ANSWER,
) -> bool:
    if response_mode == SHORT_ANSWER and not _is_meaningful_final_answer(final_answer):
        return False
    if not isinstance(verification, dict):
        return False
    status = str(verification.get("status", "")).lower()
    if response_mode == SHORT_ANSWER:
        return status in {VERIFICATION_PASSED, VERIFICATION_NOT_APPLICABLE, VERIFICATION_UNKNOWN}
    return status in {VERIFICATION_PASSED, VERIFICATION_UNKNOWN, VERIFICATION_NOT_APPLICABLE}


def _normalize_verification(
    verification: dict | None,
    *,
    response_mode: str,
    expected_answer_type: str | None,
) -> dict[str, Any]:
    raw = dict(verification) if isinstance(verification, dict) else {}
    raw_issues = raw.get("issues") if isinstance(raw.get("issues"), list) else []
    issue_codes = {
        str(issue.get("code") or "").strip().lower()
        for issue in raw_issues
        if isinstance(issue, dict)
    }
    explicit_subreason = str(raw.get("subreason") or "").strip().lower()
    kind = str(expected_answer_type or "unknown").strip().lower()

    if response_mode == SHORT_ANSWER:
        type_mismatch_codes = {
            "answer_type_mismatch",
            "count_not_nonnegative_integer",
            "expression_without_math_markers",
            "equation_answer_missing_equals",
            "multiple_roots_missing",
        }
        parse_ambiguity_codes = {
            "empty_final_answer",
            "final_answer_not_meaningful",
            "empty_solution",
            "unsupported_answer_format",
        }
        if issue_codes & type_mismatch_codes:
            status, reason, severity = VERIFICATION_UNKNOWN, "answer_type_mismatch", "low"
            explicit_subreason = "answer_type_mismatch"
        elif issue_codes & parse_ambiguity_codes:
            status, reason, severity = VERIFICATION_UNKNOWN, "candidate_parse_ambiguous", "low"
            explicit_subreason = "candidate_parse_ambiguous"
        else:
            status, reason, severity = (
                VERIFICATION_NOT_APPLICABLE,
                "no_deterministic_short_answer_verifier",
                "none",
            )
            explicit_subreason = "unsupported_short_answer_problem"
        raw.update(
            {
                "verifier_name": None,
                "verifier_applicable": False,
                "problem_parse_status": VERIFICATION_NOT_APPLICABLE,
                "candidate_parse_status": (
                    VERIFICATION_UNKNOWN if status == VERIFICATION_UNKNOWN else VERIFICATION_PASSED
                ),
                "computed_value_summary": None,
                "candidate_value_summary": None,
                "verification_subreason": explicit_subreason,
                "subreason": explicit_subreason,
            }
        )
    elif kind in {"proof", "derivation", "explanation", "matrix", "vector", "numeric_derivation"}:
        status = VERIFICATION_NOT_APPLICABLE
        reason = (
            "no_deterministic_numeric_derivation_verifier"
            if kind == "numeric_derivation"
            else f"no_deterministic_{kind}_verifier"
        )
        severity = "none"
    elif kind == "unknown":
        status, reason, severity = VERIFICATION_UNKNOWN, "insufficient_deterministic_verification", "low"
    else:
        status = VERIFICATION_NOT_APPLICABLE
        reason = f"no_deterministic_{kind}_verifier"
        severity = "none"

    raw.update({"status": status, "reason": reason, "severity": severity, "issues": raw_issues})
    return raw


def _not_evaluated_verification() -> dict[str, Any]:
    return {
        "status": VERIFICATION_NOT_EVALUATED,
        "reason": "no_valid_clean_candidate",
        "severity": "none",
        "subreason": None,
        "issues": [],
        "checks": {},
        "verifier_name": None,
        "verifier_applicable": False,
        "problem_parse_status": "not_evaluated",
        "candidate_parse_status": "not_evaluated",
        "computed_value_summary": None,
        "candidate_value_summary": None,
        "verification_subreason": None,
    }


def _parse_fraction_token(token: str) -> Fraction | None:
    value = str(token or "").strip().strip("$")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = re.sub(r"\\frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}", r"\1/\2", value)
    value = value.replace(" ", "")
    if not re.fullmatch(r"-?\d+(?:/\-?\d+|\.\d+)?", value):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def _parse_matrix_body(rows: list[list[str]]) -> list[list[Fraction]] | None:
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        return None
    parsed: list[list[Fraction]] = []
    for row in rows:
        parsed_row = [_parse_fraction_token(value) for value in row]
        if any(value is None for value in parsed_row):
            return None
        parsed.append([value for value in parsed_row if value is not None])
    return parsed


def _matrix_occurrences(text: str) -> list[tuple[list[list[Fraction]], int, int]]:
    source = str(text or "")
    occurrences: list[tuple[list[list[Fraction]], int, int]] = []

    for match in re.finditer(r"\\begin\{(?:p|b|B|v|V|small)?matrix\}(.*?)\\end\{(?:p|b|B|v|V|small)?matrix\}", source, re.DOTALL):
        row_texts = re.split(r"\\\\", match.group(1))
        rows = [[cell.strip() for cell in row.split("&")] for row in row_texts if row.strip()]
        parsed = _parse_matrix_body(rows)
        if parsed is not None:
            occurrences.append((parsed, match.start(), match.end()))

    search_at = 0
    while True:
        start = source.find("[[", search_at)
        if start < 0:
            break
        depth = 0
        end = None
        for index in range(start, len(source)):
            if source[index] == "[":
                depth += 1
            elif source[index] == "]":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            break
        segment = source[start:end]
        row_matches = re.findall(r"\[([^\[\]]+)\]", segment)
        rows = [[cell.strip() for cell in row.split(",")] for row in row_matches]
        parsed = _parse_matrix_body(rows)
        if parsed is not None:
            occurrences.append((parsed, start, end))
        search_at = end

    return sorted(occurrences, key=lambda item: item[1])


def _multiply_square_matrix(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    size = len(matrix)
    return [
        [sum((matrix[row][k] * matrix[k][column] for k in range(size)), Fraction(0)) for column in range(size)]
        for row in range(size)
    ]


def _matrix_verification_result(
    status: str,
    reason: str,
    *,
    subreason: str,
    severity: str,
    details: dict[str, Any] | None = None,
    verifier_applicable: bool = True,
    problem_parse_status: str = VERIFICATION_PASSED,
    candidate_parse_status: str = VERIFICATION_PASSED,
) -> dict[str, Any]:
    issue = {"code": subreason, "severity": severity, "message": subreason}
    return {
        "status": status,
        "reason": reason,
        "subreason": subreason,
        "severity": severity,
        "issues": [issue] if status == VERIFICATION_FAILED else [],
        "checks": details or {},
        "verifier_name": "transition_matrix_exact" if verifier_applicable else None,
        "verifier_applicable": verifier_applicable,
        "problem_parse_status": problem_parse_status,
        "candidate_parse_status": candidate_parse_status,
        "computed_value_summary": None,
        "candidate_value_summary": None,
        "verification_subreason": subreason,
    }


def _verify_transition_matrix(problem: str, clean_candidate: str) -> dict[str, Any]:
    problem_lower = str(problem or "").lower()
    if not any(marker in problem_lower for marker in ("transition matrix", "转移矩阵", "markov", "马尔可夫")):
        return _matrix_verification_result(
            VERIFICATION_NOT_APPLICABLE,
            "matrix_parser_unavailable_or_ambiguous",
            subreason="unsupported_matrix_format",
            severity="none",
            verifier_applicable=False,
            problem_parse_status=VERIFICATION_NOT_APPLICABLE,
            candidate_parse_status=VERIFICATION_NOT_APPLICABLE,
        )

    problem_matrices = [item for item in _matrix_occurrences(problem) if len(item[0]) == len(item[0][0])]
    stochastic_inputs = [item for item in problem_matrices if all(sum(row, Fraction(0)) == 1 for row in item[0])]
    if len(stochastic_inputs) != 1:
        return _matrix_verification_result(
            VERIFICATION_UNKNOWN,
            "problem_parse_ambiguous",
            subreason="problem_parse_ambiguous",
            severity="low",
            problem_parse_status=VERIFICATION_UNKNOWN,
            candidate_parse_status="not_evaluated",
        )
    source_matrix = stochastic_inputs[0][0]
    expected = _multiply_square_matrix(source_matrix)
    size = len(source_matrix)

    candidates = [item for item in _matrix_occurrences(clean_candidate) if len(item[0]) == size and len(item[0][0]) == size]
    marked_candidates = []
    for item in candidates:
        prefix = clean_candidate[max(0, item[1] - 100) : item[1]].lower()
        if re.search(r"p\s*(?:\^\s*\{?2\}?|²)|two[- ]step|两步", prefix):
            marked_candidates.append(item)
    if len(marked_candidates) == 1:
        candidate = marked_candidates[0][0]
    elif len(candidates) == 1:
        candidate = candidates[0][0]
    else:
        return _matrix_verification_result(
            VERIFICATION_UNKNOWN,
            "candidate_parse_ambiguous",
            subreason="candidate_parse_ambiguous",
            severity="low",
            candidate_parse_status=VERIFICATION_UNKNOWN,
        )

    for row_index, row in enumerate(candidate):
        row_sum = sum(row, Fraction(0))
        if row_sum != 1:
            return _matrix_verification_result(
                VERIFICATION_FAILED,
                "explicit_mathematical_error",
                subreason="row_sum_mismatch",
                severity="high",
                details={"row": row_index + 1, "actual_row_sum": str(row_sum)},
            )
    for row_index in range(size):
        for column_index in range(size):
            if candidate[row_index][column_index] != expected[row_index][column_index]:
                return _matrix_verification_result(
                    VERIFICATION_FAILED,
                    "explicit_mathematical_error",
                    subreason="matrix_entry_mismatch",
                    severity="high",
                    details={
                        "row": row_index + 1,
                        "column": column_index + 1,
                        "expected": str(expected[row_index][column_index]),
                        "actual": str(candidate[row_index][column_index]),
                    },
                )
    return _matrix_verification_result(
        VERIFICATION_PASSED,
        "deterministic_matrix_verification_passed",
        subreason="matrix_entries_and_row_sums_verified",
        severity="none",
    )


def _parse_single_integer_candidate(final_answer: str | None) -> int | None:
    text = str(final_answer or "").strip()
    if not text:
        return None
    previous = None
    while text != previous:
        previous = text
        text = text.strip().strip("$` ")
        text = re.sub(r"^\\(?:boxed|fbox)\s*\{\s*([+-]?\d+)\s*\}$", r"\1", text)
        text = re.sub(r"^\\[\[(]\s*([+-]?\d+)\s*\\[\])]$", r"\1", text)
        text = re.sub(
            r"^(?:final\s+answer|answer|value|result|最终答案|答案|结果)\s*[:：=]?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.strip().strip(".。;；,， ")
    return int(text) if re.fullmatch(r"[+-]?\d+", text) else None


def _looks_like_factorial_floor_problem(problem: str) -> bool:
    text = str(problem or "")
    lowered = text.lower()
    return (
        (r"\lfloor" in text or "floor" in lowered or "greatest integer" in lowered)
        and ("!" in text or "factorial" in lowered)
        and bool(re.search(r"\b(?:find|compute|evaluate|determine)\s+(?:the\s+)?value\b", lowered))
    )


def _parse_factorial_floor_value(problem: str) -> int | None:
    text = str(problem or "")
    fraction = re.search(
        r"\\frac\s*\{\s*(\d+)\s*!\s*\}\s*\{([^{}]+)\}",
        text,
        flags=re.DOTALL,
    )
    if not fraction:
        return None
    numerator = int(fraction.group(1))
    denominator_text = re.sub(r"\s+", "", fraction.group(2))
    ellipsis = r"(?:\\cdots|\\ldots|\.\.\.)"
    if not re.fullmatch(rf"\d+!(?:\+\d+!)*\+{ellipsis}\+\d+!", denominator_text):
        return None
    denominator_terms = [int(value) for value in re.findall(r"(\d+)!", denominator_text)]
    if len(denominator_terms) < 3:
        return None
    upper, lower = denominator_terms[0], denominator_terms[-1]
    explicit_prefix = denominator_terms[:-1]
    if (
        numerator != upper + 1
        or lower < 0
        or lower >= upper
        or numerator > 5000
        or any(left - right != 1 for left, right in zip(explicit_prefix, explicit_prefix[1:]))
    ):
        return None

    factorial = 1
    denominator = 1 if lower == 0 else 0
    numerator_factorial = None
    for value in range(1, numerator + 1):
        factorial *= value
        if lower <= value <= upper:
            denominator += factorial
        if value == numerator:
            numerator_factorial = factorial
    if numerator_factorial is None or denominator <= 0:
        return None
    return numerator_factorial // denominator


def _short_verification_result(
    status: str,
    reason: str,
    *,
    verifier_name: str | None,
    verifier_applicable: bool,
    problem_parse_status: str,
    candidate_parse_status: str,
    verification_subreason: str,
    computed_value: int | None = None,
    candidate_value: int | None = None,
) -> dict[str, Any]:
    severity = "high" if status == VERIFICATION_FAILED else "low" if status == VERIFICATION_UNKNOWN else "none"
    issue = {
        "code": verification_subreason,
        "severity": severity,
        "message": verification_subreason,
    }
    result = {
        "status": status,
        "reason": reason,
        "subreason": verification_subreason,
        "verification_subreason": verification_subreason,
        "severity": severity,
        "issues": [issue] if status == VERIFICATION_FAILED else [],
        "checks": {},
        "verifier_name": verifier_name,
        "verifier_applicable": verifier_applicable,
        "problem_parse_status": problem_parse_status,
        "candidate_parse_status": candidate_parse_status,
        "computed_value_summary": None if computed_value is None else f"integer:{computed_value}",
        "candidate_value_summary": None if candidate_value is None else f"integer:{candidate_value}",
    }
    if (
        computed_value is not None
        and verifier_applicable
        and problem_parse_status == VERIFICATION_PASSED
    ):
        result["_deterministic_answer_value"] = computed_value
        result["deterministic_override_eligible"] = True
        result["exact_unique_answer"] = True
    return result


def _verify_factorial_floor_exact(problem: str, final_answer: str | None) -> dict[str, Any]:
    expected_value = _parse_factorial_floor_value(problem)
    if expected_value is None:
        return _short_verification_result(
            VERIFICATION_UNKNOWN,
            "problem_parse_ambiguous",
            verifier_name="factorial_floor_exact",
            verifier_applicable=True,
            problem_parse_status=VERIFICATION_UNKNOWN,
            candidate_parse_status="not_evaluated",
            verification_subreason="problem_parse_ambiguous",
        )
    candidate_value = _parse_single_integer_candidate(final_answer)
    if candidate_value is None:
        return _short_verification_result(
            VERIFICATION_UNKNOWN,
            "candidate_parse_ambiguous",
            verifier_name="factorial_floor_exact",
            verifier_applicable=True,
            problem_parse_status=VERIFICATION_PASSED,
            candidate_parse_status=VERIFICATION_UNKNOWN,
            verification_subreason="candidate_parse_ambiguous",
            computed_value=expected_value,
        )
    if candidate_value != expected_value:
        return _short_verification_result(
            VERIFICATION_FAILED,
            "explicit_mathematical_error",
            verifier_name="factorial_floor_exact",
            verifier_applicable=True,
            problem_parse_status=VERIFICATION_PASSED,
            candidate_parse_status=VERIFICATION_PASSED,
            verification_subreason="floor_value_mismatch",
            computed_value=expected_value,
            candidate_value=candidate_value,
        )
    return _short_verification_result(
        VERIFICATION_PASSED,
        "deterministic_factorial_floor_verification_passed",
        verifier_name="factorial_floor_exact",
        verifier_applicable=True,
        problem_parse_status=VERIFICATION_PASSED,
        candidate_parse_status=VERIFICATION_PASSED,
        verification_subreason="floor_value_verified",
        computed_value=expected_value,
        candidate_value=candidate_value,
    )


def _looks_like_integer_polynomial_value_gcd_problem(problem: str) -> bool:
    lowered = str(problem or "").lower()
    return (
        "p(n)" in lowered
        and "largest positive integer" in lowered
        and "divides" in lowered
        and "for every integer" in lowered
        and "prime" in lowered
        and (
            "not necessarily distinct" in lowered
            or "counted with multiplicity" in lowered
            or "counting multiplicity" in lowered
        )
    )


def _parse_integer_polynomial_value_gcd_problem(problem: str) -> dict[str, int] | None:
    text = str(problem or "")
    lowered = text.lower().replace("−", "-").replace("–", "-")
    if not _looks_like_integer_polynomial_value_gcd_problem(text):
        return None

    threshold_match = re.search(r"for\s+every\s+integer\s+\$?n\$?\s*>\s*(\d+)", lowered)
    if not threshold_match:
        return None
    threshold = int(threshold_match.group(1))

    compact = lowered.replace(r"\left", "").replace(r"\right", "")
    compact = re.sub(r"\s+", "", compact)
    product_match = re.search(
        r"(?:\\prod|∏)_\{?([a-z])=1\}?\^\{?(\d+)\}?\(n-\1\^\{?(\d+)\}?\)",
        compact,
    )
    if product_match:
        upper = int(product_match.group(2))
        exponent = int(product_match.group(3))
    else:
        assignment = re.search(r"p\(n\)=([^$.;]+)", compact)
        if not assignment:
            return None
        product_text = assignment.group(1)
        if not any(marker in product_text for marker in (r"\ldots", r"\cdots", "...")):
            return None
        terms = [
            (int(base), int(power))
            for base, power in re.findall(r"\(n-(\d+)\^\{?(\d+)\}?\)", product_text)
        ]
        if len(terms) < 3 or terms[0][0] != 1 or terms[1][0] != 2:
            return None
        upper, exponent = terms[-1]
        if upper <= 2 or any(power != exponent for _, power in terms):
            return None

    if not (1 <= upper <= 100 and 1 <= exponent <= 8 and 0 <= threshold <= 1_000_000):
        return None
    return {"K": upper, "e": exponent, "N": threshold}


def _integer_polynomial_value_gcd(
    upper: int,
    exponent: int,
    threshold: int,
) -> tuple[int, int]:
    """Use K+1 consecutive values: finite differences generate every later value."""
    common_divisor = 0
    sample_count = upper + 1
    for value in range(threshold + 1, threshold + sample_count + 1):
        polynomial_value = 1
        for base in range(1, upper + 1):
            polynomial_value *= value - base**exponent
        common_divisor = math.gcd(common_divisor, abs(polynomial_value))
    return common_divisor, sample_count


def _primes_up_to(limit: int) -> list[int]:
    primes: list[int] = []
    for candidate in range(2, limit + 1):
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
    return primes


def _prime_factor_multiplicity_up_to_k(value: int, limit: int) -> tuple[int, int, dict[int, int]]:
    # A common prime divisor p requires {1^e, ..., K^e} to cover every class mod p,
    # hence p <= K; any residual after these primes means verification is incomplete.
    residual = abs(int(value))
    multiplicity = 0
    factors: dict[int, int] = {}
    for prime in _primes_up_to(limit):
        while residual > 1 and residual % prime == 0:
            residual //= prime
            multiplicity += 1
            factors[prime] = factors.get(prime, 0) + 1
    return multiplicity, residual, factors


def _verify_integer_polynomial_value_gcd(problem: str, final_answer: str | None) -> dict[str, Any]:
    parsed = _parse_integer_polynomial_value_gcd_problem(problem)
    if parsed is None:
        return _short_verification_result(
            VERIFICATION_UNKNOWN,
            "problem_parse_ambiguous",
            verifier_name="integer_polynomial_value_gcd",
            verifier_applicable=True,
            problem_parse_status=VERIFICATION_UNKNOWN,
            candidate_parse_status="not_evaluated",
            verification_subreason="problem_parse_ambiguous",
        )

    candidate_value = _parse_single_integer_candidate(final_answer)
    common_divisor, sample_count = _integer_polynomial_value_gcd(
        parsed["K"], parsed["e"], parsed["N"]
    )
    multiplicity, residual, factors = _prime_factor_multiplicity_up_to_k(
        common_divisor, parsed["K"]
    )
    checks = {
        "sample_count": sample_count,
        "expected_sample_count": parsed["K"] + 1,
        "factor_count": len(factors),
    }
    if residual != 1:
        result = _short_verification_result(
            VERIFICATION_UNKNOWN,
            "unexpected_large_prime_factor",
            verifier_name="integer_polynomial_value_gcd",
            verifier_applicable=True,
            problem_parse_status=VERIFICATION_PASSED,
            candidate_parse_status=(
                VERIFICATION_PASSED if candidate_value is not None else VERIFICATION_UNKNOWN
            ),
            verification_subreason="unexpected_large_prime_factor",
            candidate_value=candidate_value,
        )
        result["checks"] = checks
        return result
    if candidate_value is None:
        result = _short_verification_result(
            VERIFICATION_UNKNOWN,
            "candidate_parse_ambiguous",
            verifier_name="integer_polynomial_value_gcd",
            verifier_applicable=True,
            problem_parse_status=VERIFICATION_PASSED,
            candidate_parse_status=VERIFICATION_UNKNOWN,
            verification_subreason="candidate_parse_ambiguous",
            computed_value=multiplicity,
        )
        result["checks"] = checks
        return result
    if candidate_value != multiplicity:
        result = _short_verification_result(
            VERIFICATION_FAILED,
            "short_answer_verification_failed",
            verifier_name="integer_polynomial_value_gcd",
            verifier_applicable=True,
            problem_parse_status=VERIFICATION_PASSED,
            candidate_parse_status=VERIFICATION_PASSED,
            verification_subreason="prime_factor_multiplicity_mismatch",
            computed_value=multiplicity,
            candidate_value=candidate_value,
        )
        result["checks"] = checks
        return result
    result = _short_verification_result(
        VERIFICATION_PASSED,
        "deterministic_integer_polynomial_gcd_verification_passed",
        verifier_name="integer_polynomial_value_gcd",
        verifier_applicable=True,
        problem_parse_status=VERIFICATION_PASSED,
        candidate_parse_status=VERIFICATION_PASSED,
        verification_subreason="prime_factor_multiplicity_verified",
        computed_value=multiplicity,
        candidate_value=candidate_value,
    )
    result["checks"] = checks
    return result


def _looks_like_polynomial_factor_count(problem: str) -> bool:
    lowered = str(problem or "").lower()
    return (
        "monic irreducible polynomial" in lowered
        and ("factorization" in lowered or "factorisation" in lowered)
        and ("how many" in lowered or "number of" in lowered)
    )


def _verify_clean_candidate(
    problem: str,
    clean_candidate: str | None,
    final_answer: str | None,
    *,
    output_quality: dict[str, Any],
    response_mode: str,
    expected_answer_type: str | None,
    domain: str,
    solver_key: str,
) -> dict[str, Any]:
    candidate = str(clean_candidate or "").strip()
    if not output_quality.get("passed") or not candidate:
        return _not_evaluated_verification()

    kind = str(expected_answer_type or "unknown").strip().lower()
    if response_mode == WORKED_SOLUTION and kind == "matrix":
        return _verify_transition_matrix(problem, candidate)
    if response_mode == WORKED_SOLUTION and kind == "numeric_derivation":
        return {
            "status": VERIFICATION_NOT_APPLICABLE,
            "reason": "no_deterministic_numeric_derivation_verifier",
            "subreason": "unsupported_verification",
            "severity": "none",
            "issues": [],
            "checks": {},
        }

    if response_mode == SHORT_ANSWER and _looks_like_factorial_floor_problem(problem):
        return _verify_factorial_floor_exact(problem, final_answer)
    if response_mode == SHORT_ANSWER and _looks_like_integer_polynomial_value_gcd_problem(problem):
        return _verify_integer_polynomial_value_gcd(problem, final_answer)
    if response_mode == SHORT_ANSWER and _looks_like_polynomial_factor_count(problem):
        candidate_value = _parse_single_integer_candidate(final_answer)
        return _short_verification_result(
            VERIFICATION_NOT_APPLICABLE,
            "no_deterministic_polynomial_factorization_verifier",
            verifier_name=None,
            verifier_applicable=False,
            problem_parse_status=VERIFICATION_NOT_APPLICABLE,
            candidate_parse_status=(
                VERIFICATION_PASSED if candidate_value is not None else VERIFICATION_UNKNOWN
            ),
            verification_subreason="unsupported_algebraic_factorization_problem",
            candidate_value=candidate_value,
        )

    return _normalize_verification(
        verify_solution(
            problem,
            candidate,
            final_answer,
            answer_type=kind,
            domain=domain,
            solver_key=solver_key,
        ),
        response_mode=response_mode,
        expected_answer_type=kind,
    )


def _verification_trace_content(verification: dict[str, Any], expected_answer_type: str | None) -> str:
    return (
        f"status={verification.get('status')}, "
        f"reason={verification.get('reason')}, "
        f"subreason={verification.get('subreason')}, "
        f"severity={verification.get('severity')}, "
        f"verifier_name={verification.get('verifier_name')}, "
        f"verifier_applicable={verification.get('verifier_applicable')}, "
        f"problem_parse_status={verification.get('problem_parse_status')}, "
        f"candidate_parse_status={verification.get('candidate_parse_status')}, "
        f"computed_value_summary={verification.get('computed_value_summary')}, "
        f"candidate_value_summary={verification.get('candidate_value_summary')}, "
        f"verification_subreason={verification.get('verification_subreason') or verification.get('subreason')}, "
        f"expected_answer_type={expected_answer_type or 'unknown'}"
    )


def _build_correction_prompt(
    problem: str,
    metadata: dict | None,
    first_solution: str,
    first_final_answer: str | None,
    verification: dict | None,
    solver_key: str | None = None,
    domain: str | None = None,
    expected_answer_type: str | None = None,
    response_mode: str = SHORT_ANSWER,
    output_quality_reason: str | None = None,
) -> str:
    del metadata, first_solution, first_final_answer, solver_key, domain, expected_answer_type
    del verification, output_quality_reason
    if response_mode == SHORT_ANSWER:
        return (
            f"Problem:\n{str(problem or '').strip()}\n\n"
            "Recompute independently. Give at most 8 concise exact steps. "
            "Do not repeat the conclusion. End once with one boxed final answer."
        )
    return f"Problem:\n{str(problem or '').strip()}\n\nProduce the final solution now."


def _proof_risk_signals(solution: str | None) -> list[str]:
    """Flag a few high-signal proof transitions without judging or rejecting the proof."""
    text = str(solution or "")
    if not text:
        return []
    displays = list(
        re.finditer(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", text, flags=re.DOTALL)
    )
    connector = re.compile(
        r"\b(?:it\s+follows|therefore|thus|hence|consequently|so)\b|因此|从而|故|推出",
        flags=re.IGNORECASE,
    )

    def direction(value: str) -> str | None:
        has_upper = bool(re.search(r"\\leq?\b|<=|≤", value))
        has_lower = bool(re.search(r"\\geq?\b|>=|≥", value))
        if has_upper == has_lower:
            return None
        return "upper" if has_upper else "lower"

    signals: list[str] = []
    for previous, current in zip(displays, displays[1:]):
        previous_direction = direction(previous.group(0))
        current_direction = direction(current.group(0))
        bridge = text[previous.end() : current.start()]
        if (
            previous_direction
            and current_direction
            and previous_direction != current_direction
            and connector.search(bridge)
        ):
            signals.extend(
                (
                    "unjustified_inequality_direction_change",
                    "conclusion_not_supported_by_previous_display",
                )
            )
            break

    for match in re.finditer(r"\bupper\s+bound\b|上界", text, flags=re.IGNORECASE):
        following = text[match.end() : match.end() + 900]
        lower_bound_position = re.search(r"\\geq?\b|>=|≥|\bat\s+least\b|下界", following, flags=re.IGNORECASE)
        connector_position = connector.search(following)
        if lower_bound_position and connector_position and connector_position.start() <= lower_bound_position.start():
            signals.append("upper_bound_used_as_lower_bound")
            break
    return list(dict.fromkeys(signals))


class ReasoningAgent:
    def __init__(self, client, *args, **kwargs):
        self.client = client

    def _call_model(
        self,
        *,
        messages,
        temperature,
        max_tokens,
        thinking_mode,
    ):
        kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        thinking_mode_applied = False
        if (
            thinking_mode is not None
            and _supports_public_kwarg(self.client.chat, "thinking_mode")
        ):
            kwargs["thinking_mode"] = thinking_mode
            thinking_mode_applied = True
        response = self.client.chat(**kwargs)
        return response, thinking_mode_applied

    def _chat(self, prompt: str, *, response_mode: str, retry: bool = False) -> tuple[str, bool]:
        short_answer = response_mode == SHORT_ANSWER
        if short_answer:
            system_message = SHORT_ANSWER_RETRY_SYSTEM_MESSAGE if retry else SHORT_ANSWER_SYSTEM_MESSAGE
        else:
            system_message = RETRY_SYSTEM_MESSAGE if retry else PRIMARY_SYSTEM_MESSAGE
        response, thinking_mode_applied = self._call_model(
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0 if retry or short_answer else 0.2,
            max_tokens=4096,
            thinking_mode=False if short_answer else not retry,
        )
        return _response_content(response).strip(), thinking_mode_applied

    def _extract_and_verify(
        self,
        problem: str,
        raw_solution: str,
        domain: str,
        solver_key: str,
        expected_answer_type: str | None = None,
        response_mode: str = SHORT_ANSWER,
    ) -> tuple[str | None, dict, dict, dict[str, Any]]:
        judgeable_solution, extraction_details = _extract_judgeable_solution_with_diagnostics(
            raw_solution,
            response_mode=response_mode,
        )
        compact_judgeable = str(judgeable_solution or "").strip()
        protected_worked_types = {"proof", "derivation", "explanation", "matrix", "vector", "numeric_derivation"}
        if response_mode == WORKED_SOLUTION and str(expected_answer_type or "unknown").lower() in protected_worked_types:
            extraction = {
                "status": "not_applicable",
                "final_answer": None,
                "answer_type": str(expected_answer_type or "unknown").lower(),
            }
        else:
            extraction = extract_final_answer(problem, compact_judgeable, domain)
            if response_mode == SHORT_ANSWER and re.fullmatch(r"[A-Za-z]\s*=\s*[^\n]+", compact_judgeable):
                extraction = dict(extraction)
                extraction["final_answer"] = compact_judgeable
                extraction["answer_type"] = "set" if "," in compact_judgeable else "expression"
                extraction["status"] = "passed"
            elif (
                response_mode == SHORT_ANSWER
                and str(expected_answer_type or "").lower() == "expression"
                and "\n" not in compact_judgeable
                and re.search(r"[A-Za-z\\]", compact_judgeable)
                and re.search(r"[=+\-*/^_\\]", compact_judgeable)
            ):
                extraction = dict(extraction)
                extraction["final_answer"] = compact_judgeable
                extraction["answer_type"] = "expression"
                extraction["status"] = "passed"
        final_answer = extraction.get("final_answer")
        verifier_answer_type = (
            expected_answer_type
            if str(expected_answer_type or "unknown").lower() != "unknown"
            else extraction.get("answer_type") or "unknown"
        )
        output_quality = _assess_output_quality(
            raw_solution,
            judgeable_solution,
            final_answer,
            response_mode,
            extraction_details,
        )
        verification = _verify_clean_candidate(
            problem,
            judgeable_solution,
            final_answer,
            output_quality=output_quality,
            response_mode=response_mode,
            expected_answer_type=verifier_answer_type,
            domain=domain,
            solver_key=solver_key,
        )
        return judgeable_solution, extraction, verification, output_quality

    def _try_local_tools(
        self,
        problem: str,
        expected_answer_type: str | None,
        domain: str,
        solver_key: str,
        response_mode: str,
        trace: list[dict],
    ) -> dict | None:
        local_tools = [
            solve_divisibility_subset_problem,
            solve_finite_field_problem,
            solve_combinatorics_counting_problem,
            solve_number_theory_problem,
            solve_recurrence_sequence_problem,
            solve_elementary_algebra_problem,
        ]
        for tool_fn in local_tools:
            tool_result = tool_fn(problem)
            if tool_result is None:
                continue

            details = tool_result.get("details", {})
            trace.append(_trace("local_tool_detect", f"tool_name={tool_result.get('tool_name')}, details={details}"))
            final_answer = str(tool_result.get("final_answer") or "").strip()
            solution = str(tool_result.get("solution") or "")
            verifier_answer_type = expected_answer_type or "number"
            trace.append(
                _trace(
                    "local_tool_solve",
                    (
                        f"tool_name={tool_result.get('tool_name')}, "
                        f"final_answer={final_answer!r}, "
                        f"details={details}"
                    ),
                )
            )
            output_quality = _assess_output_quality(
                solution,
                solution,
                final_answer,
                response_mode,
                {"tag_status": "not_applicable", "compatibility_recovered": False},
            )
            trace.append(
                _trace(
                    "output_quality_check",
                    _quality_trace_content(output_quality),
                )
            )
            verification = _verify_clean_candidate(
                problem,
                solution,
                final_answer,
                output_quality=output_quality,
                response_mode=response_mode,
                expected_answer_type=verifier_answer_type,
                domain=domain,
                solver_key=solver_key,
            )
            trace.append(
                _trace(
                    "verify",
                    _verification_trace_content(verification, verifier_answer_type),
                )
            )
            final_response = _compose_final_response(
                problem=problem,
                response_mode=response_mode,
                solution=solution,
                extracted_answer=final_answer,
                verification=verification,
            )
            pipeline_acceptance_passed = (
                _final_acceptance_passed_for_mode(output_quality, verification, response_mode)
                and final_response != FALLBACK_RESPONSE
            )
            reliability = _reliability_fields(
                verification,
                pipeline_acceptance_passed,
                bool(output_quality.get("passed")),
            )
            trace.append(
                _trace(
                    "finalize",
                    (
                        f"final_response_chars={len(final_response)}, "
                        f"final_acceptance_passed={pipeline_acceptance_passed}, "
                        f"pipeline_acceptance_passed={pipeline_acceptance_passed}, "
                        f"fallback_used={final_response == FALLBACK_RESPONSE}"
                    ),
                )
            )
            return {
                "final_response": final_response,
                "trace": trace,
                "clean_candidate_tail": _clean_candidate_tail(solution, output_quality),
                "proof_risk_signals": [],
                "proof_review_required": False,
                "deterministic_answer_override": False,
                "override_verifier_name": None,
                **reliability,
            }
        return None

    def solve(self, problem: str, metadata: dict | None) -> dict:
        problem_text = str(problem or "")
        metadata = metadata if isinstance(metadata, dict) else {}
        safe_metadata = _safe_metadata(metadata)
        trace = []

        try:
            classification = _apply_metadata_domain(classify_problem(problem_text), safe_metadata, problem_text)
            domain = classification.get("domain", "unknown")
            solver_key = classification.get("solver_key", "general")
            inferred_answer_type = _expected_answer_type_from_problem(problem_text)
            expected_answer_type = inferred_answer_type
            response_mode = _determine_response_mode(problem_text, domain, expected_answer_type)
            # Re-run once with the response mode available for otherwise ambiguous prompts.
            inferred_answer_type = _expected_answer_type_from_problem(problem_text, response_mode)
            if inferred_answer_type != "unknown":
                expected_answer_type = inferred_answer_type
            matched_categories = classification.get("matched_signal_categories")
            matched_summary = "|".join(str(item) for item in matched_categories) if isinstance(matched_categories, list) else ""
            trace.append(
                _trace(
                    "classify",
                    (
                        f"domain={domain}, solver_key={solver_key}, "
                        f"routing_confidence={classification.get('routing_confidence') or 'unknown'}, "
                        f"matched_signal_categories={matched_summary or 'none'}, "
                        f"runner_up_domain={classification.get('runner_up_domain') or 'none'}, "
                        f"score_margin={classification.get('score_margin', 0)}"
                    ),
                )
            )
            trace.append(
                _trace(
                    "response_mode",
                    f"response_mode={response_mode}, expected_answer_type={expected_answer_type}",
                )
            )

            local_result = self._try_local_tools(
                problem_text,
                expected_answer_type,
                domain,
                solver_key,
                response_mode,
                trace,
            )
            if local_result is not None:
                return local_result

            plan = make_plan(problem_text, domain)
            trace.append(_trace("plan", "; ".join(plan)))

            prompt = build_solver_prompt(
                problem_text,
                domain,
                plan,
                retry_context=None,
                solver_key=solver_key,
            )
            prompt = _append_prompt_constraints(
                prompt,
                problem_text,
                solver_key,
                expected_answer_type,
                response_mode,
            )
            trace.append(_trace("solver_prompt", f"solver_key={solver_key}, prompt_chars={len(prompt)}"))

            first_thinking_mode_requested = response_mode == WORKED_SOLUTION
            solution, thinking_mode_applied = self._chat(prompt, response_mode=response_mode)
            trace.append(
                _model_call_event(
                    "model_call",
                    "solution",
                    solution,
                    thinking_mode_requested=first_thinking_mode_requested,
                    thinking_mode_applied=thinking_mode_applied,
                )
            )

            judgeable_solution, extraction, verification, output_quality = self._extract_and_verify(
                problem_text,
                solution,
                domain,
                solver_key,
                expected_answer_type,
                response_mode,
            )
            final_answer = extraction.get("final_answer")
            first_attempt_diagnostics = _attempt_diagnostics(
                output_quality=output_quality,
                extraction=extraction,
                candidate=judgeable_solution,
                final_answer=final_answer,
                verification=verification,
            )
            extracted_answer_type = extraction.get("answer_type")
            verifier_answer_type = expected_answer_type or extracted_answer_type
            trace.append(
                _trace(
                    "extract",
                    _final_answer_trace(extraction, final_answer, extracted_answer_type, verifier_answer_type),
                )
            )
            trace.append(
                _trace(
                    "output_quality_check",
                    _quality_trace_content(output_quality),
                )
            )
            trace.append(
                _trace(
                    "verify",
                    _verification_trace_content(verification, verifier_answer_type),
                )
            )

            retry_reasons = _retry_reasons(final_answer, verification, output_quality)
            retry_used = bool(retry_reasons)
            trace.append(
                _trace(
                    "retry_decision",
                    (
                        f"retry_used={retry_used}, "
                        f"reasons={retry_reasons}, "
                        f"issues_count={len(verification.get('issues', [])) if isinstance(verification, dict) else 0}"
                    ),
                )
            )

            retry_final_answer = None
            retry_verification = None
            retry_solution = None
            retry_judgeable_solution = None
            retry_output_quality = None
            retry_attempt_diagnostics = None
            if retry_used and MAX_SOLVE_ATTEMPTS > 1:
                correction_prompt = _build_correction_prompt(
                    problem_text,
                    safe_metadata,
                    solution,
                    final_answer,
                    verification,
                    solver_key=solver_key,
                    domain=domain,
                    expected_answer_type=expected_answer_type,
                    response_mode=response_mode,
                    output_quality_reason=_primary_retry_reason(output_quality, verification),
                )
                trace.append(_trace("correction_prompt", f"correction_prompt_chars={len(correction_prompt)}"))
                try:
                    retry_solution, retry_thinking_mode_applied = self._chat(
                        correction_prompt,
                        response_mode=response_mode,
                        retry=True,
                    )
                    trace.append(
                        _model_call_event(
                            "retry_model_call",
                            "retry_solution",
                            retry_solution,
                            thinking_mode_requested=False,
                            thinking_mode_applied=retry_thinking_mode_applied,
                        )
                    )
                    (
                        retry_judgeable_solution,
                        retry_extraction,
                        retry_verification,
                        retry_output_quality,
                    ) = self._extract_and_verify(
                        problem_text,
                        retry_solution,
                        domain,
                        solver_key,
                        expected_answer_type,
                        response_mode,
                    )
                    retry_final_answer = retry_extraction.get("final_answer")
                    retry_attempt_diagnostics = _attempt_diagnostics(
                        output_quality=retry_output_quality,
                        extraction=retry_extraction,
                        candidate=retry_judgeable_solution,
                        final_answer=retry_final_answer,
                        verification=retry_verification,
                    )
                    retry_extracted_answer_type = retry_extraction.get("answer_type")
                    retry_verifier_answer_type = expected_answer_type or retry_extracted_answer_type
                    trace.append(
                        _trace(
                            "retry_extract",
                            _final_answer_trace(
                                retry_extraction,
                                retry_final_answer,
                                retry_extracted_answer_type,
                                retry_verifier_answer_type,
                            ),
                        )
                    )
                    trace.append(
                        _trace(
                            "retry_output_quality_check",
                            _quality_trace_content(retry_output_quality),
                        )
                    )
                    trace.append(
                        _trace(
                            "retry_verify",
                            _verification_trace_content(retry_verification, retry_verifier_answer_type),
                        )
                    )
                except Exception as exc:
                    trace.append(_trace("retry_model_call", f"error: {type(exc).__name__}"))

            selected_quality = output_quality
            selected_verification = verification
            diagnostic_quality = output_quality
            diagnostic_candidate = judgeable_solution
            deterministic_answer_override = False
            override_verifier_name = None
            if retry_output_quality is not None and retry_output_quality.get("passed"):
                diagnostic_quality = retry_output_quality
                diagnostic_candidate = retry_judgeable_solution
            override = _deterministic_override_value(verification, retry_verification)
            if retry_output_quality is not None and retry_output_quality.get("passed") and override is not None:
                override_value, override_verifier_name = override
                deterministic_answer_override = True
                final_response = str(override_value)
                selected_quality = retry_output_quality
                selected_verification = dict(retry_verification or {})
                selected_verification.update(
                    {
                        "status": VERIFICATION_PASSED,
                        "reason": "deterministic_answer_override_verified",
                        "subreason": "deterministic_exact_value_override",
                        "verification_subreason": "deterministic_exact_value_override",
                        "severity": "none",
                        "issues": [],
                        "candidate_parse_status": VERIFICATION_PASSED,
                        "candidate_value_summary": f"integer:{override_value}",
                    }
                )
                diagnostic_quality = retry_output_quality
                diagnostic_candidate = str(override_value)
                trace.append(
                    _trace(
                        "deterministic_override",
                        (
                            "deterministic_answer_override=True, "
                            f"override_verifier_name={override_verifier_name}"
                        ),
                    )
                )
            elif retry_output_quality is not None and _final_acceptance_passed_for_mode(
                retry_output_quality, retry_verification, response_mode
            ):
                final_response = _compose_final_response(
                    problem=problem_text,
                    response_mode=response_mode,
                    solution=retry_judgeable_solution,
                    extracted_answer=retry_final_answer,
                    verification=retry_verification,
                )
                selected_quality = retry_output_quality
                selected_verification = retry_verification
            elif _final_acceptance_passed_for_mode(output_quality, verification, response_mode):
                final_response = _compose_final_response(
                    problem=problem_text,
                    response_mode=response_mode,
                    solution=judgeable_solution,
                    extracted_answer=final_answer,
                    verification=verification,
                )
            else:
                final_response = FALLBACK_RESPONSE
            proof_risk_signals = (
                _proof_risk_signals(diagnostic_candidate)
                if str(expected_answer_type or "").lower() == "proof" and diagnostic_quality.get("passed")
                else []
            )
            pipeline_acceptance_passed = (
                _final_acceptance_passed_for_mode(
                    selected_quality, selected_verification, response_mode
                )
                and final_response != FALLBACK_RESPONSE
            )
            reliability = _reliability_fields(
                selected_verification,
                pipeline_acceptance_passed,
                bool(selected_quality.get("passed")),
            )
            trace.append(
                _trace(
                    "finalize",
                    (
                        f"final_response_chars={len(final_response)}, "
                        f"final_acceptance_passed={pipeline_acceptance_passed}, "
                        f"pipeline_acceptance_passed={pipeline_acceptance_passed}, "
                        f"deterministic_answer_override={deterministic_answer_override}, "
                        f"override_verifier_name={override_verifier_name}, "
                        f"fallback_used={final_response == FALLBACK_RESPONSE}"
                    ),
                )
            )
            return {
                "final_response": final_response,
                "trace": trace,
                "clean_candidate_tail": _clean_candidate_tail(diagnostic_candidate, diagnostic_quality),
                "first_attempt_diagnostics": first_attempt_diagnostics,
                "retry_attempt_diagnostics": retry_attempt_diagnostics,
                "proof_risk_signals": proof_risk_signals,
                "proof_review_required": bool(proof_risk_signals),
                "deterministic_answer_override": deterministic_answer_override,
                "override_verifier_name": override_verifier_name,
                **reliability,
            }
        except Exception as exc:
            trace.append(_trace("model_call", f"error: {type(exc).__name__}"))
            trace.append(_trace("finalize", "fallback_response"))
            return {
                "final_response": FALLBACK_RESPONSE,
                "trace": trace,
                "clean_candidate_tail": None,
                "first_attempt_diagnostics": None,
                "retry_attempt_diagnostics": None,
                "proof_risk_signals": [],
                "proof_review_required": False,
                "deterministic_answer_override": False,
                "override_verifier_name": None,
                **_reliability_fields(None, False, False),
            }
