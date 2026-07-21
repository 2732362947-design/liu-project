"""Sequential, one-question-per-domain advanced real Intern API sanity runner."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from dev_tools.run_user_agent_real_smoke import (
        _build_client_from_env,
        _load_input_item,
        _load_input_items,
        run_smoke,
    )
    from user_agent import FALLBACK_RESPONSE
except ModuleNotFoundError:
    from run_user_agent_real_smoke import _build_client_from_env, _load_input_item, _load_input_items, run_smoke
    from user_agent import FALLBACK_RESPONSE


ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = ROOT / "data" / "real_api_sanity_advanced.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "real_api_sanity_advanced_results.json"
DEFAULT_MODEL = "intern-s2-preview"
CLASSIFY_RE = re.compile(r"domain=([^,]+),\s*solver_key=([^,\s]+)")
MODE_RE = re.compile(r"\b(short_answer|worked_solution)\b")
EXPECTED_TYPE_RE = re.compile(r"\bexpected_answer_type=([^,\s]+)")
STATUS_RE = re.compile(r"\bstatus\s*=\s*([^,\s]+)", re.I)
REASON_RE = re.compile(r"\breason\s*=\s*([^,\s]+)", re.I)
SUBREASON_RE = re.compile(r"\bsubreason\s*=\s*([^,\s]+)", re.I)
SEVERITY_RE = re.compile(r"\bseverity\s*=\s*([^,\s]+)", re.I)
TOOL_RE = re.compile(r"tool_name=([^,\s]+)")
RETRY_TRUE_RE = re.compile(
    r"\b(?:retry_used|retry_triggered|needs_retry|retry_required)\s*=\s*(?:true|1|yes)\b",
    re.I,
)
VERIFY_TRUE_RE = re.compile(r"\b(?:status\s*=\s*)?(?:passed|success|verified)\b|\bpassed\s*=\s*true\b", re.I)
VERIFY_FALSE_RE = re.compile(
    r"\b(?:status\s*=\s*)?(?:failed|failure|error|rejected|unverified)\b|\bpassed\s*=\s*false\b",
    re.I,
)
TRACE_STEPS = {
    "classify",
    "response_mode",
    "local_tool_detect",
    "local_tool_solve",
    "model_call",
    "verify",
    "retry_decision",
    "retry_model_call",
    "retry_extract",
    "retry_verify",
    "output_quality_check",
    "retry_output_quality_check",
    "finalize",
    "fallback",
    "final_response_fallback",
}


def _redact_text(value: Any, max_chars: int = 300) -> str:
    text = str(value or "")
    text = re.sub(
        r"(?i)\bauthorization\s*[:=]\s*[^\s,;]+(?:\s+[^\s,;]+)?",
        "[redacted credential]",
        text,
    )
    text = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "[redacted credential]", text)
    text = re.sub(
        r"(?i)\b(?:api[_ -]?key|token|secret)\s*[:=]\s*[^\s,;&]+",
        "credential=[redacted]",
        text,
    )
    text = re.sub(
        r"(?i)([?&](?:api[_-]?key|token|access_token|secret)=)[^&#\s]+",
        r"\1[redacted]",
        text,
    )
    return text[:max_chars]


def _verification_value(content: str) -> bool | None:
    if VERIFY_FALSE_RE.search(content):
        return False
    if VERIFY_TRUE_RE.search(content):
        return True
    return None


def _boolean_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "passed", "success", "verified"}:
        return True
    if normalized in {"false", "0", "no", "failed", "failure", "error", "rejected", "unverified"}:
        return False
    return None


def _verification_from_event(event: dict[str, Any], content: str) -> bool | None:
    for key in ("passed", "verification_passed", "verified", "status"):
        if key in event:
            parsed = _boolean_value(event[key])
            if parsed is not None:
                return parsed
    return _verification_value(content)


def _status_from_event(event: dict[str, Any], content: str) -> str | None:
    direct = str(event.get("status") or "").strip().lower()
    match = STATUS_RE.search(content)
    status = direct or (match.group(1).lower() if match else "")
    aliases = {"verified": "passed", "success": "passed", "failure": "failed", "error": "failed"}
    status = aliases.get(status, status)
    if status in {"passed", "failed", "unknown", "not_applicable", "not_evaluated"}:
        return status
    passed = _verification_from_event(event, content)
    return "passed" if passed is True else "failed" if passed is False else None


def _summarize_trace(trace: object, final_response: object = "") -> dict[str, Any]:
    """Extract serializable behavior fields without retaining full model traces."""
    predicted_domain = "unknown"
    solver_key = "general"
    response_mode = "unknown"
    expected_answer_type = "unknown"
    api_call_count = 0
    retry_triggered = False
    mathematical_verification_status: str | None = None
    mathematical_verification_reason: str | None = None
    mathematical_verification_subreason: str | None = None
    mathematical_verification_severity: str | None = None
    output_quality_passed: bool | None = None
    output_quality_reason: str | None = None
    output_quality_subreason: str | None = None
    first_thinking_mode_requested: bool | None = None
    first_thinking_mode_applied: bool | None = None
    retry_thinking_mode_requested: bool | None = None
    retry_thinking_mode_applied: bool | None = None
    fallback_used = str(final_response or "").strip() == FALLBACK_RESPONSE.strip()
    local_tool_name = None
    trace_summary: list[dict[str, str]] = []

    events = trace if isinstance(trace, list) else []
    for event in events:
        if not isinstance(event, dict):
            continue
        step = str(event.get("step") or "")
        content = str(event.get("content") or "")
        lower_step = step.lower()
        lower_content = content.lower()

        if step == "classify":
            match = CLASSIFY_RE.search(content)
            if match:
                predicted_domain = match.group(1).strip()
                solver_key = match.group(2).strip()
        mode_match = MODE_RE.search(content)
        direct_mode = str(event.get("response_mode") or "")
        if step in {"response_mode", "finalize"} and direct_mode in {"short_answer", "worked_solution"}:
            response_mode = direct_mode
        elif step in {"response_mode", "finalize"} and mode_match:
            response_mode = mode_match.group(1)
        type_match = EXPECTED_TYPE_RE.search(content)
        if step in {"response_mode", "verify", "retry_verify"} and type_match:
            expected_answer_type = type_match.group(1).strip()

        if step in {"model_call", "retry_model_call"}:
            api_call_count += 1
            requested = _boolean_value(event.get("thinking_mode_requested"))
            applied = _boolean_value(event.get("thinking_mode_applied"))
            if step == "model_call":
                first_thinking_mode_requested = requested
                first_thinking_mode_applied = applied
            else:
                retry_thinking_mode_requested = requested
                retry_thinking_mode_applied = applied
        direct_retry = next(
            (
                parsed
                for key in ("retry_used", "retry_triggered", "needs_retry", "retry_required")
                if key in event and (parsed := _boolean_value(event[key])) is not None
            ),
            None,
        )
        if step in {"retry_model_call", "retry_extract", "retry_verify"} or (
            step == "retry_decision" and (direct_retry is True or RETRY_TRUE_RE.search(content))
        ):
            retry_triggered = True
        if step in {"verify", "retry_verify"}:
            mathematical_verification_status = _status_from_event(event, content)
            reason_match = REASON_RE.search(content)
            subreason_match = SUBREASON_RE.search(content)
            severity_match = SEVERITY_RE.search(content)
            mathematical_verification_reason = (
                str(event.get("reason") or "").strip()
                or (reason_match.group(1).strip() if reason_match else None)
            )
            mathematical_verification_subreason = (
                str(event.get("subreason") or "").strip()
                or (subreason_match.group(1).strip() if subreason_match else None)
            )
            mathematical_verification_severity = (
                str(event.get("severity") or "").strip()
                or (severity_match.group(1).strip() if severity_match else None)
            )
            if mathematical_verification_subreason in {"None", "none", "null", ""}:
                mathematical_verification_subreason = None
        if step in {"output_quality_check", "retry_output_quality_check"}:
            output_quality_passed = _verification_from_event(event, content)
            reason_match = REASON_RE.search(content)
            subreason_match = SUBREASON_RE.search(content)
            output_quality_reason = (
                str(event.get("reason") or "").strip()
                or (reason_match.group(1).strip() if reason_match else None)
            )
            output_quality_subreason = (
                str(event.get("subreason") or "").strip()
                or (subreason_match.group(1).strip() if subreason_match else None)
            )
            if output_quality_subreason in {"None", "none", "null", ""}:
                output_quality_subreason = None
        if step in {"local_tool_detect", "local_tool_solve"}:
            direct_tool = event.get("tool_name")
            tool_match = TOOL_RE.search(content)
            if direct_tool:
                local_tool_name = str(direct_tool)
            elif tool_match:
                local_tool_name = tool_match.group(1).strip()
        direct_fallback = _boolean_value(event.get("fallback_used")) if "fallback_used" in event else None
        if (
            lower_step in {"fallback", "final_response_fallback"}
            or (step == "finalize" and re.search(r"\bfallback(?:_response)?\b", lower_content))
            or re.search(r"\bfallback_(?:used|triggered)\s*=\s*true\b", lower_content)
            or direct_fallback is True
        ):
            fallback_used = True
        if step in TRACE_STEPS:
            summary_limit = 420 if step in {"verify", "retry_verify", "output_quality_check", "retry_output_quality_check"} else 180
            trace_summary.append({"step": step, "content": _redact_text(content, summary_limit)})

    mathematical_verification_passed = (
        mathematical_verification_status == "passed" if mathematical_verification_status is not None else None
    )
    final_acceptance_passed = (
        output_quality_passed and mathematical_verification_status != "failed"
        if mathematical_verification_status is not None and output_quality_passed is not None
        else None
    )
    return {
        "predicted_domain": predicted_domain,
        "solver_key": solver_key,
        "response_mode": response_mode,
        "expected_answer_type": expected_answer_type,
        "api_call_count": api_call_count,
        "retry_triggered": retry_triggered,
        "verification_passed": final_acceptance_passed,
        "final_acceptance_passed": final_acceptance_passed,
        "mathematical_verification_status": mathematical_verification_status,
        "mathematical_verification_reason": mathematical_verification_reason,
        "mathematical_verification_subreason": mathematical_verification_subreason,
        "mathematical_verification_severity": mathematical_verification_severity,
        "mathematical_verification_passed": mathematical_verification_passed,
        "output_quality_passed": output_quality_passed,
        "output_quality_reason": output_quality_reason,
        "output_quality_subreason": output_quality_subreason,
        "first_thinking_mode_requested": first_thinking_mode_requested,
        "first_thinking_mode_applied": first_thinking_mode_applied,
        "retry_thinking_mode_requested": retry_thinking_mode_requested,
        "retry_thinking_mode_applied": retry_thinking_mode_applied,
        "fallback_used": fallback_used,
        "local_tool_name": local_tool_name,
        "trace_summary": trace_summary,
    }


def _classification_from_trace(trace: list[dict]) -> tuple[str, str]:
    summary = _summarize_trace(trace)
    return summary["predicted_domain"], summary["solver_key"]


def _destination(path: str | Path) -> Path:
    destination = Path(path)
    return destination if destination.is_absolute() else ROOT / destination


def _atomic_write_summaries(destination: Path, summaries: list[dict[str, Any]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_name = stream.name
            json.dump(summaries, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _success_summary(
    *,
    idx: str,
    declared_domain: str,
    model: str,
    result: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    final_response = str(result.get("final_response") or "")
    trace_fields = _summarize_trace(result.get("trace"), final_response)
    clean_candidate_tail = re.sub(r"\s+", " ", str(result.get("clean_candidate_tail") or "")).strip()[-160:]
    if any(
        marker in clean_candidate_tail.lower()
        for marker in ("thinking process", "system instruction", "user prompt", "model identity", "api_key", "bearer")
    ):
        clean_candidate_tail = ""
    return {
        "idx": idx,
        "declared_domain": declared_domain,
        **trace_fields,
        "model": model,
        "status": "success",
        "final_response": final_response,
        "clean_candidate_tail": clean_candidate_tail or None,
        "final_response_nonempty": bool(final_response.strip()),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "error": None,
    }


def _error_summary(
    *,
    idx: str,
    declared_domain: str,
    model: str,
    elapsed_seconds: float,
    error: Exception,
) -> dict[str, Any]:
    return {
        "idx": idx,
        "declared_domain": declared_domain,
        "predicted_domain": "unknown",
        "solver_key": "general",
        "model": model,
        "status": "error",
        "final_response": "",
        "clean_candidate_tail": None,
        "final_response_nonempty": False,
        "response_mode": "unknown",
        "expected_answer_type": "unknown",
        "api_call_count": 0,
        "retry_triggered": False,
        "verification_passed": None,
        "final_acceptance_passed": None,
        "mathematical_verification_status": None,
        "mathematical_verification_reason": None,
        "mathematical_verification_subreason": None,
        "mathematical_verification_severity": None,
        "mathematical_verification_passed": None,
        "output_quality_passed": None,
        "output_quality_reason": None,
        "output_quality_subreason": None,
        "first_thinking_mode_requested": None,
        "first_thinking_mode_applied": None,
        "retry_thinking_mode_requested": None,
        "retry_thinking_mode_applied": None,
        "fallback_used": False,
        "local_tool_name": None,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "trace_summary": [],
        "error": {"type": type(error).__name__, "message": _redact_text(error, 300)},
    }


def _terminal_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"final_response", "trace_summary", "error"}
    } | {
        "final_response_summary": _redact_text(summary.get("final_response"), 300),
        "error": summary.get("error"),
    }


def run_advanced_sanity(
    *,
    input_path: str | Path,
    output_path: str | Path,
    timeout: float,
    sleep_seconds: float,
    model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    items = _load_input_items(input_path)
    os.environ["INTERN_S1_MODEL"] = model
    print(f"requested_model={model}", flush=True)
    client = _build_client_from_env(timeout)
    destination = _destination(output_path)
    seen_domains: set[str] = set()
    summaries: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        nested_metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        idx = str(item.get("problem_id") or item.get("idx") or f"advanced_{index:06d}")
        declared_domain = str(
            item.get("domain")
            or item.get("subject")
            or nested_metadata.get("domain")
            or nested_metadata.get("subject")
            or "unknown"
        )
        if declared_domain in seen_domains:
            continue
        seen_domains.add(declared_domain)
        started = time.perf_counter()
        try:
            problem, idx, _ = _load_input_item(input_path, index)
            safe_metadata = {"idx": idx}
            result = run_smoke(client, problem=problem, idx=idx, metadata=safe_metadata)
            if not isinstance(result, dict):
                raise TypeError("run_smoke returned a non-object result")
            summary = _success_summary(
                idx=idx,
                declared_domain=declared_domain,
                model=model,
                result=result,
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as exc:
            summary = _error_summary(
                idx=idx,
                declared_domain=declared_domain,
                model=model,
                elapsed_seconds=time.perf_counter() - started,
                error=exc,
            )
        summaries.append(summary)
        _atomic_write_summaries(destination, summaries)
        print(json.dumps(_terminal_summary(summary), ensure_ascii=False), flush=True)
        if sleep_seconds > 0 and index + 1 < len(items):
            time.sleep(sleep_seconds)
    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT.relative_to(ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(ROOT)))
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="The requested Intern model ID. Defaults to intern-s2-preview.",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--sleep", type=float, default=0)
    parser.add_argument("--concurrency", type=int, choices=(1,), default=1)
    parser.add_argument("--limit-per-domain", type=int, choices=(1,), default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_advanced_sanity(
        input_path=args.input,
        output_path=args.output,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
        model=args.model,
    )


if __name__ == "__main__":
    main()
