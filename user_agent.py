from __future__ import annotations

from typing import Any

from agents.answer_extractor_agent import extract_final_answer
import re

from agents.classifier_agent import classify_problem, solver_key_for_domain
from agents.planner_agent import make_plan
from agents.solver_agent import build_solver_prompt, normalize_solver_key
from agents.tools.combinatorics_counting_tool import solve_combinatorics_counting_problem
from agents.tools.combinatorics_graph_tool import solve_divisibility_subset_problem
from agents.tools.finite_field_tool import solve_finite_field_problem
from agents.tools.number_theory_tool import solve_number_theory_problem
from agents.verifier_agent import verify_solution


FALLBACK_RESPONSE = "未能得到可靠答案"
MAX_SOLVE_ATTEMPTS = 2
SOLUTION_HEAD_LIMIT = 300
SOLUTION_TAIL_LIMIT = 500
CORRECTION_HEAD_LIMIT = 800
CORRECTION_TAIL_LIMIT = 1200
SENSITIVE_MARKERS = ("authorization", "bearer", "api_key", "token")
METADATA_DENYLIST = {"answer", "expected_answer", "gold_answer", "reference_answer", "solution"}
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


def _head_tail(text: str | None, head_limit: int, tail_limit: int) -> tuple[str, str]:
    raw = str(text or "")
    head = raw[:head_limit]
    tail = raw[-tail_limit:] if len(raw) > tail_limit else raw
    return _safe_snippet(head, head_limit), _safe_snippet(tail, tail_limit)


def _model_output_trace(prefix: str, solution: str | None) -> str:
    text = str(solution or "")
    head, tail = _head_tail(text, SOLUTION_HEAD_LIMIT, SOLUTION_TAIL_LIMIT)
    return (
        "status=success, "
        f"{prefix}_chars={len(text)}, "
        f"{prefix}_head={head!r}, "
        f"{prefix}_tail={tail!r}"
    )


def _final_answer_trace(
    extraction: dict,
    final_answer: str | None,
    extracted_answer_type: str | None,
    expected_answer_type: str | None,
) -> str:
    answer_text = str(final_answer or "")
    return (
        f"status={extraction.get('status')}, "
        f"extracted_final_answer={_safe_snippet(answer_text, 200)!r}, "
        f"extracted_answer_type={extracted_answer_type}, "
        f"expected_answer_type={expected_answer_type}, "
        f"meaningful_final={_is_meaningful_final_answer(final_answer)}, "
        f"final_answer_chars={len(answer_text)}, "
        f"has_final={bool(final_answer)}"
    )


def _response_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if response.get("content") is not None:
            return str(response["content"])
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            return _response_content(choices[0])
        message = response.get("message")
        if message is not None:
            return _response_content(message)
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    choices = getattr(response, "choices", None)
    if choices:
        return _response_content(choices[0])
    message = getattr(response, "message", None)
    if message is not None:
        return _response_content(message)
    return str(response or "")


def _safe_metadata(metadata: dict | None) -> dict:
    if not isinstance(metadata, dict):
        return {}
    safe = {}
    for key, value in metadata.items():
        key_text = str(key)
        if key_text.lower() in METADATA_DENYLIST:
            continue
        safe[key_text] = value
    return safe


def _domain_from_metadata(safe_metadata: dict | None) -> str | None:
    if not isinstance(safe_metadata, dict):
        return None
    domain = str(safe_metadata.get("domain") or "").strip()
    if not domain or domain.lower() == "unknown":
        return None
    return domain


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
    domain = _domain_from_metadata(safe_metadata)
    if not domain:
        return updated
    if domain == "optimization" and _problem_suggests_extremal_discrete(problem):
        updated["domain"] = "combinatorics"
        updated["solver_key"] = "discrete"
        updated["reason"] = "题面是极值集合/组合图论结构，覆盖 metadata optimization 路由。"
        return updated
    metadata_solver_key = None
    if isinstance(safe_metadata, dict) and safe_metadata.get("solver_key"):
        metadata_solver_key = str(safe_metadata.get("solver_key"))
    updated["domain"] = domain
    updated["solver_key"] = _solver_key_from_domain(domain, metadata_solver_key)
    updated["reason"] = "使用安全 metadata.domain 进行领域路由。"
    return updated


def _expected_answer_type_from_metadata(safe_metadata: dict | None) -> str | None:
    if not isinstance(safe_metadata, dict):
        return None
    answer_type = str(safe_metadata.get("answer_type") or "").strip().lower()
    return answer_type or None


def _problem_expects_number_first(problem: str | None, expected_answer_type: str | None) -> bool:
    text = str(problem or "").lower()
    if str(expected_answer_type or "").lower() == "number":
        return True
    return any(marker in text for marker in ("smallest positive integer", "minimum integer", "number of"))


def _build_prompt_constraints(problem: str, solver_key: str, expected_answer_type: str | None) -> str:
    constraints = []
    if _problem_expects_number_first(problem, expected_answer_type):
        constraints.append(
            "请先输出一行：\n"
            "最终答案：[本题计算结果]\n"
            "其中方括号内容只是格式说明，实际作答时必须替换为本题计算得到的答案；"
            "不要原样输出方括号、占位符或“答案”二字作为答案内容。"
            "然后再给简洁推理。如果推理较长，也必须先给最终答案，避免最终答案因截断丢失。"
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
    return "\n\n【输出格式与长度约束】\n" + "\n".join(f"{index + 1}. {item}" for index, item in enumerate(constraints))


def _append_prompt_constraints(prompt: str, problem: str, solver_key: str, expected_answer_type: str | None) -> str:
    constraint_text = _build_prompt_constraints(problem, solver_key, expected_answer_type)
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


def _retry_reasons(final_answer: str | None, verification: dict | None) -> list[str]:
    reasons = []
    final_text = str(final_answer or "").strip()
    if not _is_meaningful_final_answer(final_answer):
        reasons.append("not_meaningful_final_answer")
    elif not final_text or final_text == FALLBACK_RESPONSE:
        reasons.append("empty_or_fallback_final_answer")
    if not isinstance(verification, dict):
        return reasons
    status = str(verification.get("status", "")).lower()
    severity = str(verification.get("severity", "")).lower()
    issues = verification.get("issues", [])
    if status in {"failed", "uncertain"}:
        reasons.append(f"verification_status_{status}")
    if severity in {"high", "critical"}:
        reasons.append(f"verification_severity_{severity}")
    if issues and severity != "none":
        reasons.append("verification_issues")
    return reasons


def _should_retry(final_answer: str | None, verification: dict | None) -> bool:
    return bool(_retry_reasons(final_answer, verification))


def _verification_allows_final_answer(final_answer: str | None, verification: dict | None) -> bool:
    if not _is_meaningful_final_answer(final_answer):
        return False
    if not isinstance(verification, dict):
        return False
    status = str(verification.get("status", "")).lower()
    severity = str(verification.get("severity", "")).lower()
    issues = verification.get("issues", [])
    issue_codes = {
        str(issue.get("code", ""))
        for issue in issues
        if isinstance(issue, dict)
    }
    if issue_codes & {"answer_type_mismatch", "number_without_digits", "final_answer_not_meaningful"}:
        return False
    if status in {"failed", "uncertain"} or severity in {"high", "critical", "medium"}:
        return False
    return True


def _build_correction_prompt(
    problem: str,
    metadata: dict | None,
    first_solution: str,
    first_final_answer: str | None,
    verification: dict | None,
    solver_key: str | None = None,
    domain: str | None = None,
    expected_answer_type: str | None = None,
) -> str:
    verification = verification if isinstance(verification, dict) else {}
    safe_metadata = _safe_metadata(metadata)
    first_solution = str(first_solution or "")
    if len(first_solution) <= CORRECTION_HEAD_LIMIT + CORRECTION_TAIL_LIMIT:
        solution_block = _safe_snippet(first_solution, CORRECTION_HEAD_LIMIT + CORRECTION_TAIL_LIMIT)
    else:
        solution_head, solution_tail = _head_tail(first_solution, CORRECTION_HEAD_LIMIT, CORRECTION_TAIL_LIMIT)
        solution_block = (
            f"first_solution_head:\n{solution_head}\n\n"
            f"first_solution_tail:\n{solution_tail}"
        )
    answer_type_instruction = ""
    if str(expected_answer_type or "").lower() == "number":
        answer_type_instruction = (
            "\n6. 最终答案必须是一个单独的数值或数值表达式。\n"
            "7. 不要输出变量赋值列表，例如 x=2, x=3。\n"
            "8. 不要输出多个候选答案。\n"
            "9. 如果题目问 smallest/minimum/number/integer，最终答案应为单个整数或数值表达式。\n"
            "10. 先输出一行“最终答案：[本题计算结果]”，再用不超过 1200 字说明关键证明；"
            "方括号内容只是格式说明，实际答案必须替换为本题计算结果，不要原样输出方括号或占位符。"
        )
    if str(expected_answer_type or "").lower() == "expression":
        answer_type_instruction += (
            "\n6. 上一次答案类型不符合要求：本题需要 expression 类型最终答案。\n"
            "7. 不要再次返回纯数字或占位数字；除非表达式确实化简为常数，否则最终答案必须包含变量、运算符或函数形式。\n"
            "8. 适用时请使用题目中的变量，并输出清晰的表达式。"
        )
    if _problem_suggests_extremal_discrete(problem):
        answer_type_instruction += (
            "\n不要继续列边或邻接表；请先给最终答案，再用结构分组、参数族、下界构造和上界证明说明。"
        )
    return (
        "你正在修正一道数学题的解答。\n\n"
        "【题目】\n"
        f"{_safe_snippet(problem, 900)}\n\n"
        "【路由与答案类型】\n"
        f"domain={domain or safe_metadata.get('domain') or 'unknown'}\n"
        f"solver_key={solver_key or 'unknown'}\n"
        f"expected_answer_type={expected_answer_type or safe_metadata.get('answer_type') or 'unknown'}\n\n"
        "【第一次解答摘要】\n"
        f"{solution_block}\n\n"
        "【第一次提取的最终答案】\n"
        f"{_safe_snippet(first_final_answer or '', 200)}\n\n"
        "【本地验证器反馈】\n"
        f"status: {verification.get('status')}\n"
        f"severity: {verification.get('severity')}\n"
        f"issues: {_safe_snippet(verification.get('issues', []), 700)}\n"
        f"suggestion: {_safe_snippet(verification.get('suggestion') or verification.get('feedback'), 400)}\n\n"
        "请根据题目重新检查推理，修正可能的错误。\n"
        "要求：\n"
        "1. 给出简洁但完整的推理。\n"
        "2. 最终答案必须明确。\n"
        "3. 如果原答案正确，请说明并保持答案。\n"
        "4. 不要引用标准答案或隐藏评测信息。\n"
        "5. 不要输出 JSON，直接给出可读解答即可。"
        f"{answer_type_instruction}"
    )


class ReasoningAgent:
    def __init__(self, client, *args, **kwargs):
        self.client = client

    def _chat(self, prompt: str) -> str:
        response = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4096,
        )
        return _response_content(response).strip()

    def _extract_and_verify(
        self,
        problem: str,
        solution: str,
        domain: str,
        solver_key: str,
        expected_answer_type: str | None = None,
    ) -> tuple[dict, dict]:
        extraction = extract_final_answer(problem, solution, domain)
        final_answer = extraction.get("final_answer")
        verifier_answer_type = expected_answer_type or extraction.get("answer_type")
        verification = verify_solution(
            problem,
            solution,
            final_answer,
            answer_type=verifier_answer_type,
            domain=domain,
            solver_key=solver_key,
        )
        return extraction, verification

    def _try_local_tools(
        self,
        problem: str,
        expected_answer_type: str | None,
        domain: str,
        solver_key: str,
        trace: list[dict],
    ) -> dict | None:
        local_tools = [
            solve_divisibility_subset_problem,
            solve_finite_field_problem,
            solve_combinatorics_counting_problem,
            solve_number_theory_problem,
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
            verification = verify_solution(
                problem,
                solution,
                final_answer,
                answer_type=verifier_answer_type,
                domain=domain,
                solver_key=solver_key,
            )
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
            trace.append(
                _trace(
                    "verify",
                    (
                        f"status={verification.get('status')}, "
                        f"severity={verification.get('severity')}, "
                        f"expected_answer_type={verifier_answer_type}"
                    ),
                )
            )
            final_response = final_answer if _verification_allows_final_answer(final_answer, verification) else FALLBACK_RESPONSE
            trace.append(_trace("finalize", f"final_response_chars={len(final_response)}"))
            return {"final_response": final_response, "trace": trace}
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
            expected_answer_type = _expected_answer_type_from_metadata(safe_metadata)
            trace.append(_trace("classify", f"domain={domain}, solver_key={solver_key}"))

            local_result = self._try_local_tools(problem_text, expected_answer_type, domain, solver_key, trace)
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
            prompt = _append_prompt_constraints(prompt, problem_text, solver_key, expected_answer_type)
            trace.append(_trace("solver_prompt", f"solver_key={solver_key}, prompt_chars={len(prompt)}"))

            solution = self._chat(prompt)
            trace.append(_trace("model_call", _model_output_trace("solution", solution)))

            extraction, verification = self._extract_and_verify(
                problem_text,
                solution,
                domain,
                solver_key,
                expected_answer_type,
            )
            final_answer = extraction.get("final_answer")
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
                    "verify",
                    (
                        f"status={verification.get('status')}, "
                        f"severity={verification.get('severity')}, "
                        f"expected_answer_type={verifier_answer_type}"
                    ),
                )
            )

            retry_reasons = _retry_reasons(final_answer, verification)
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
                )
                trace.append(_trace("correction_prompt", f"correction_prompt_chars={len(correction_prompt)}"))
                try:
                    retry_solution = self._chat(correction_prompt)
                    trace.append(_trace("retry_model_call", _model_output_trace("retry_solution", retry_solution)))
                    retry_extraction, retry_verification = self._extract_and_verify(
                        problem_text,
                        retry_solution,
                        domain,
                        solver_key,
                        expected_answer_type,
                    )
                    retry_final_answer = retry_extraction.get("final_answer")
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
                            "retry_verify",
                            (
                                f"status={retry_verification.get('status')}, "
                                f"severity={retry_verification.get('severity')}, "
                                f"expected_answer_type={retry_verifier_answer_type}"
                            ),
                        )
                    )
                except Exception as exc:
                    trace.append(_trace("retry_model_call", f"error: {type(exc).__name__}"))

            final_response = ""
            if _verification_allows_final_answer(retry_final_answer, retry_verification):
                final_response = str(retry_final_answer).strip()
            elif _verification_allows_final_answer(final_answer, verification):
                final_response = str(final_answer or "").strip()
            if not final_response:
                final_response = FALLBACK_RESPONSE
            trace.append(_trace("finalize", f"final_response_chars={len(final_response)}"))
            return {"final_response": final_response, "trace": trace}
        except Exception as exc:
            trace.append(_trace("model_call", f"error: {type(exc).__name__}"))
            trace.append(_trace("finalize", "fallback_response"))
            return {"final_response": FALLBACK_RESPONSE, "trace": trace}
