from __future__ import annotations

from typing import Any

from agents.answer_extractor_agent import extract_final_answer
from agents.classifier_agent import classify_problem
from agents.planner_agent import make_plan
from agents.solver_agent import build_solver_prompt
from agents.verifier_agent import verify_solution


FALLBACK_RESPONSE = "未能得到可靠答案"
MAX_SOLVE_ATTEMPTS = 2
SENSITIVE_MARKERS = ("authorization", "bearer", "api_key", "token")
METADATA_DENYLIST = {"answer", "expected_answer", "gold_answer", "reference_answer"}


def _safe_text(value: Any, limit: int = 500) -> str:
    text = str(value or "")
    lowered = text.lower()
    for marker in SENSITIVE_MARKERS:
        if marker in lowered:
            return "[redacted]"
    return text[:limit]


def _trace(step: str, content: Any) -> dict:
    return {"step": step, "content": _safe_text(content)}


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


def _should_retry(final_answer: str | None, verification: dict | None) -> bool:
    final_text = str(final_answer or "").strip()
    if not final_text or final_text == FALLBACK_RESPONSE:
        return True
    if not isinstance(verification, dict):
        return False
    status = str(verification.get("status", "")).lower()
    severity = str(verification.get("severity", "")).lower()
    issues = verification.get("issues", [])
    if status in {"failed", "uncertain"}:
        return True
    if severity in {"high", "critical"}:
        return True
    if issues and severity != "none":
        return True
    return False


def _build_correction_prompt(
    problem: str,
    metadata: dict | None,
    first_solution: str,
    first_final_answer: str | None,
    verification: dict | None,
    solver_key: str | None = None,
) -> str:
    verification = verification if isinstance(verification, dict) else {}
    safe_metadata = _safe_metadata(metadata)
    return (
        "你正在修正一道数学题的解答。\n\n"
        "【题目】\n"
        f"{problem}\n\n"
        "【非答案元数据】\n"
        f"{safe_metadata}\n\n"
        "【第一次解题计划或 solver_key】\n"
        f"solver_key={solver_key or 'unknown'}\n\n"
        "【第一次解答】\n"
        f"{first_solution}\n\n"
        "【第一次提取的最终答案】\n"
        f"{first_final_answer or ''}\n\n"
        "【本地验证器反馈】\n"
        f"status: {verification.get('status')}\n"
        f"issues: {verification.get('issues', [])}\n"
        f"suggestion: {verification.get('suggestion') or verification.get('feedback')}\n\n"
        "请根据题目重新检查推理，修正可能的错误。\n"
        "要求：\n"
        "1. 给出简洁但完整的推理。\n"
        "2. 最终答案必须明确。\n"
        "3. 如果原答案正确，请说明并保持答案。\n"
        "4. 不要引用标准答案或隐藏评测信息。\n"
        "5. 不要输出 JSON，直接给出可读解答即可。"
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

    def _extract_and_verify(self, problem: str, solution: str, domain: str, solver_key: str) -> tuple[dict, dict]:
        extraction = extract_final_answer(problem, solution, domain)
        final_answer = extraction.get("final_answer")
        verification = verify_solution(
            problem,
            solution,
            final_answer,
            answer_type=extraction.get("answer_type"),
            domain=domain,
            solver_key=solver_key,
        )
        return extraction, verification

    def solve(self, problem: str, metadata: dict | None) -> dict:
        problem_text = str(problem or "")
        metadata = metadata if isinstance(metadata, dict) else {}
        trace = []

        try:
            classification = classify_problem(problem_text)
            domain = classification.get("domain", "unknown")
            solver_key = classification.get("solver_key", "general")
            trace.append(_trace("classify", f"domain={domain}, solver_key={solver_key}"))

            plan = make_plan(problem_text, domain)
            trace.append(_trace("plan", "; ".join(plan)))

            prompt = build_solver_prompt(
                problem_text,
                domain,
                plan,
                retry_context=None,
                solver_key=solver_key,
            )
            trace.append(_trace("solver_prompt", f"solver_key={solver_key}, prompt_chars={len(prompt)}"))

            solution = self._chat(prompt)
            trace.append(_trace("model_call", "success"))

            extraction, verification = self._extract_and_verify(problem_text, solution, domain, solver_key)
            final_answer = extraction.get("final_answer")
            answer_type = extraction.get("answer_type")
            trace.append(
                _trace(
                    "extract",
                    f"status={extraction.get('status')}, answer_type={answer_type}, has_final={bool(final_answer)}",
                )
            )
            trace.append(
                _trace(
                    "verify",
                    f"status={verification.get('status')}, severity={verification.get('severity')}",
                )
            )

            retry_used = _should_retry(final_answer, verification)
            trace.append(
                _trace(
                    "retry_decision",
                    f"retry_used={retry_used}, issues_count={len(verification.get('issues', [])) if isinstance(verification, dict) else 0}",
                )
            )

            retry_final_answer = None
            if retry_used and MAX_SOLVE_ATTEMPTS > 1:
                correction_prompt = _build_correction_prompt(
                    problem_text,
                    metadata,
                    solution,
                    final_answer,
                    verification,
                    solver_key=solver_key,
                )
                trace.append(_trace("correction_prompt", f"correction_prompt_chars={len(correction_prompt)}"))
                try:
                    retry_solution = self._chat(correction_prompt)
                    trace.append(_trace("retry_model_call", "success"))
                    retry_extraction, retry_verification = self._extract_and_verify(
                        problem_text,
                        retry_solution,
                        domain,
                        solver_key,
                    )
                    retry_final_answer = retry_extraction.get("final_answer")
                    trace.append(
                        _trace(
                            "retry_extract",
                            (
                                f"status={retry_extraction.get('status')}, "
                                f"answer_type={retry_extraction.get('answer_type')}, "
                                f"has_final={bool(retry_final_answer)}"
                            ),
                        )
                    )
                    trace.append(
                        _trace(
                            "retry_verify",
                            (
                                f"status={retry_verification.get('status')}, "
                                f"severity={retry_verification.get('severity')}"
                            ),
                        )
                    )
                except Exception as exc:
                    trace.append(_trace("retry_model_call", f"error: {type(exc).__name__}"))

            final_response = str(final_answer or "").strip()
            if retry_final_answer:
                final_response = str(retry_final_answer).strip()
            if not final_response:
                final_response = FALLBACK_RESPONSE
            trace.append(_trace("finalize", f"final_response_chars={len(final_response)}"))
            return {"final_response": final_response, "trace": trace}
        except Exception as exc:
            trace.append(_trace("model_call", f"error: {type(exc).__name__}"))
            trace.append(_trace("finalize", "fallback_response"))
            return {"final_response": FALLBACK_RESPONSE, "trace": trace}
