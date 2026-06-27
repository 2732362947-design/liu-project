from __future__ import annotations

from typing import Any

from agents.answer_extractor_agent import extract_final_answer
from agents.classifier_agent import classify_problem
from agents.planner_agent import make_plan
from agents.solver_agent import build_solver_prompt
from agents.verifier_agent import verify_solution


FALLBACK_RESPONSE = "未能得到可靠答案"
SENSITIVE_MARKERS = ("authorization", "bearer", "api_key", "token")


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


class ReasoningAgent:
    def __init__(self, client, *args, **kwargs):
        self.client = client

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

            response = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4096,
            )
            solution = _response_content(response).strip()
            trace.append(_trace("model_call", "success"))

            extraction = extract_final_answer(problem_text, solution, domain)
            final_answer = extraction.get("final_answer")
            answer_type = extraction.get("answer_type")
            trace.append(
                _trace(
                    "extract",
                    f"status={extraction.get('status')}, answer_type={answer_type}, has_final={bool(final_answer)}",
                )
            )

            verification = verify_solution(
                problem_text,
                solution,
                final_answer,
                answer_type=answer_type,
                domain=domain,
                solver_key=solver_key,
            )
            trace.append(
                _trace(
                    "verify",
                    f"status={verification.get('status')}, severity={verification.get('severity')}",
                )
            )

            final_response = str(final_answer or "").strip()
            if not final_response:
                final_response = FALLBACK_RESPONSE
            trace.append(_trace("finalize", f"final_response_chars={len(final_response)}"))
            return {"final_response": final_response, "trace": trace}
        except Exception as exc:
            trace.append(_trace("model_call", f"error: {type(exc).__name__}"))
            trace.append(_trace("finalize", "fallback_response"))
            return {"final_response": FALLBACK_RESPONSE, "trace": trace}
