"""Run Omni-MATH through the formal ReasoningAgent and a real Intern-S API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev_tools.run_user_agent_real_smoke import (
    DEFAULT_BASE_URL,
    RealInternClient,
    _load_env_file,
    safe_text,
)
from user_agent import FALLBACK_RESPONSE, ReasoningAgent


DEFAULT_MODEL = "intern-s2-preview"
CLASSIFY_RE = re.compile(r"domain=([^,]+),\s*solver_key=([^,\s]+)")
MODE_RE = re.compile(r"response_mode=([^,\s]+)")
TOOL_RE = re.compile(r"tool_name=([^,\s]+)")
STATUS_RE = re.compile(r"status=([^,\s]+)")
EXPECTED_TYPE_RE = re.compile(r"expected_answer_type=([^,\s]+)")
REASON_RE = re.compile(r"reason=([^,\s]+)")
SUBREASON_RE = re.compile(r"subreason=([^,\s]+)")
SEVERITY_RE = re.compile(r"severity=([^,\s]+)")
ROUTING_CONFIDENCE_RE = re.compile(r"routing_confidence=([^,\s]+)")
MATCHED_SIGNALS_RE = re.compile(r"matched_signal_categories=([^,\s]+)")
RUNNER_UP_RE = re.compile(r"runner_up_domain=([^,\s]+)")
SCORE_MARGIN_RE = re.compile(r"score_margin=(-?\d+)")
TRACE_SUMMARY_STEPS = {
    "classify",
    "response_mode",
    "model_call",
    "extract",
    "output_quality_check",
    "verify",
    "retry_decision",
    "retry_model_call",
    "retry_extract",
    "retry_output_quality_check",
    "retry_verify",
    "deterministic_override",
    "finalize",
}
DIAGNOSTIC_TOKEN_FIELDS = (
    "tag_status",
    "quality_reason",
    "quality_subreason",
    "latex_balance_status",
    "ending_status",
    "extracted_answer_type",
    "verification_reason",
    "verification_subreason",
    "verifier_name",
    "problem_parse_status",
    "candidate_parse_status",
)
META_OR_SENSITIVE_MARKERS = (
    "thinking process",
    "analyze the request",
    "system instruction",
    "user prompt",
    "model identity",
    "authorization",
    "bearer",
    "api_key",
    "token",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def build_solve_metadata(item: dict[str, Any], use_subject_hint: bool = False) -> dict[str, Any]:
    """Construct the complete and deliberately tiny solve metadata object."""
    del use_subject_hint
    return {"idx": item.get("idx", "")}


def _boolean_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "passed"}:
        return True
    if normalized in {"false", "0", "no", "failed"}:
        return False
    return None


def _answer_reliability(
    mathematical_status: str | None,
    output_quality_passed: bool | None,
) -> tuple[str, bool]:
    status = str(mathematical_status or "not_evaluated").lower()
    if status == "passed":
        return "verified", False
    if status == "failed":
        return "rejected", True
    if output_quality_passed is True and status in {"not_applicable", "unknown"}:
        return "unverified", True
    return "unavailable", True


def _content_value(content: str, name: str) -> str | None:
    match = re.search(rf"\b{re.escape(name)}=([^,\s]+)", content)
    if not match:
        return None
    value = match.group(1).strip()
    return None if value.lower() in {"none", "null", ""} else value


def _safe_diagnostic_text(value: Any, limit: int, *, tail: bool = False) -> str | None:
    compact = re.sub(r"\s+", " ", str(value or "")).strip()
    lowered = compact.lower()
    if not compact or any(marker in lowered for marker in META_OR_SENSITIVE_MARKERS):
        return None
    excerpt = compact[-limit:] if tail else compact[:limit]
    return safe_text(excerpt, limit)


def _empty_attempt_diagnostics() -> dict[str, Any]:
    return {
        "raw_chars": 0,
        "tag_status": "unknown",
        "clean_candidate_chars": 0,
        "quality_reason": "unknown",
        "quality_subreason": None,
        "latex_balance_status": "unknown",
        "ending_status": "unknown",
        "extracted_answer_type": "unknown",
        "extracted_answer_summary": None,
        "candidate_tail": None,
        "verification_reason": "unknown",
        "verification_subreason": None,
        "verifier_name": None,
        "verifier_applicable": None,
        "problem_parse_status": "unknown",
        "candidate_parse_status": "unknown",
        "computed_value_summary": None,
        "candidate_value_summary": None,
    }


def _sanitize_attempt_diagnostics(value: Any, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    source = value if isinstance(value, dict) else fallback if isinstance(fallback, dict) else {}
    sanitized = _empty_attempt_diagnostics()
    for name in ("raw_chars", "clean_candidate_chars"):
        try:
            sanitized[name] = max(0, int(source.get(name) or 0))
        except (TypeError, ValueError):
            sanitized[name] = 0
    for name in DIAGNOSTIC_TOKEN_FIELDS:
        token = re.sub(r"[^A-Za-z0-9_.-]", "", str(source.get(name) or ""))[:80]
        if name == "verifier_name":
            sanitized[name] = token or None
        else:
            sanitized[name] = token or (None if name.endswith("subreason") else "unknown")
    sanitized["verifier_applicable"] = _boolean_value(source.get("verifier_applicable"))
    sanitized["extracted_answer_summary"] = _safe_diagnostic_text(
        source.get("extracted_answer_summary"), 120
    )
    sanitized["candidate_tail"] = _safe_diagnostic_text(source.get("candidate_tail"), 200, tail=True)
    sanitized["computed_value_summary"] = _safe_diagnostic_text(
        source.get("computed_value_summary"), 120
    )
    sanitized["candidate_value_summary"] = _safe_diagnostic_text(
        source.get("candidate_value_summary"), 120
    )
    return sanitized


def _safe_trace_summary(trace: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for event in trace if isinstance(trace, list) else []:
        if not isinstance(event, dict):
            continue
        step = str(event.get("step") or "")
        if step not in TRACE_SUMMARY_STEPS:
            continue
        content = str(event.get("content") or "")
        summary: dict[str, Any] = {"step": step}
        allowed_fields = {
            "classify": ("domain", "solver_key", "routing_confidence", "matched_signal_categories", "runner_up_domain", "score_margin"),
            "response_mode": ("response_mode", "expected_answer_type"),
            "model_call": ("status", "solution_chars"),
            "retry_model_call": ("status", "retry_solution_chars"),
            "extract": ("status", "extracted_answer_type", "expected_answer_type", "meaningful_final", "final_answer_chars", "has_final"),
            "retry_extract": ("status", "extracted_answer_type", "expected_answer_type", "meaningful_final", "final_answer_chars", "has_final"),
            "output_quality_check": ("status", "reason", "subreason", "raw_chars", "clean_candidate_chars", "tag_status", "latex_balance_status", "ending_status", "ending_reason"),
            "retry_output_quality_check": ("status", "reason", "subreason", "raw_chars", "clean_candidate_chars", "tag_status", "latex_balance_status", "ending_status", "ending_reason"),
            "verify": (
                "status", "reason", "subreason", "severity", "expected_answer_type",
                "verifier_name", "verifier_applicable", "problem_parse_status",
                "candidate_parse_status", "computed_value_summary", "candidate_value_summary",
                "verification_subreason",
            ),
            "retry_verify": (
                "status", "reason", "subreason", "severity", "expected_answer_type",
                "verifier_name", "verifier_applicable", "problem_parse_status",
                "candidate_parse_status", "computed_value_summary", "candidate_value_summary",
                "verification_subreason",
            ),
            "retry_decision": ("retry_used", "issues_count"),
            "deterministic_override": (
                "deterministic_answer_override", "override_verifier_name",
            ),
            "finalize": (
                "final_response_chars", "final_acceptance_passed",
                "pipeline_acceptance_passed", "deterministic_answer_override",
                "override_verifier_name", "fallback_used",
            ),
        }.get(step, ())
        for name in allowed_fields:
            parsed = _content_value(content, name)
            if parsed is not None:
                summary[name] = parsed[:120]
        if step in {"model_call", "retry_model_call"}:
            summary["thinking_mode_requested"] = _boolean_value(event.get("thinking_mode_requested"))
            summary["thinking_mode_applied"] = _boolean_value(event.get("thinking_mode_applied"))
        summaries.append(summary)
    return summaries


def _trace_parts(trace: Any) -> dict[str, Any]:
    predicted_domain = "unknown"
    solver_key = "general"
    routing_confidence = "unknown"
    matched_signal_categories: list[str] = []
    runner_up_domain = None
    score_margin = None
    response_mode = "unknown"
    expected_answer_type = "unknown"
    local_tool_name = None
    api_call_count = 0
    retry_triggered = False
    mathematical_verification_statuses: list[str] = []
    output_quality_statuses: list[str] = []
    mathematical_verification_reasons: list[str] = []
    mathematical_verification_subreasons: list[str | None] = []
    mathematical_verification_severities: list[str] = []
    output_quality_reasons: list[str] = []
    output_quality_subreasons: list[str | None] = []
    first_thinking_mode_requested = None
    first_thinking_mode_applied = None
    retry_thinking_mode_requested = None
    retry_thinking_mode_applied = None
    retry_reason_codes: list[str] = []
    first_attempt_diagnostics = _empty_attempt_diagnostics()
    retry_attempt_diagnostics = _empty_attempt_diagnostics()
    traced_final_acceptance = None
    model_call_error = None
    if not isinstance(trace, list):
        trace = []
    trace_summary = _safe_trace_summary(trace)
    for event in trace:
        if not isinstance(event, dict):
            continue
        step = str(event.get("step") or "")
        content = str(event.get("content") or "")
        if step == "classify":
            match = CLASSIFY_RE.search(content)
            if match:
                predicted_domain, solver_key = match.group(1).strip(), match.group(2).strip()
            confidence_match = ROUTING_CONFIDENCE_RE.search(content)
            signals_match = MATCHED_SIGNALS_RE.search(content)
            runner_up_match = RUNNER_UP_RE.search(content)
            margin_match = SCORE_MARGIN_RE.search(content)
            if confidence_match:
                routing_confidence = confidence_match.group(1).strip()
            if signals_match and signals_match.group(1).lower() != "none":
                matched_signal_categories = [part for part in signals_match.group(1).split("|") if part]
            if runner_up_match and runner_up_match.group(1).lower() != "none":
                runner_up_domain = runner_up_match.group(1).strip()
            if margin_match:
                score_margin = int(margin_match.group(1))
        elif step == "response_mode":
            match = MODE_RE.search(content)
            if match:
                response_mode = match.group(1).strip()
        if step in {"response_mode", "verify", "retry_verify"}:
            type_match = EXPECTED_TYPE_RE.search(content)
            if type_match:
                expected_answer_type = type_match.group(1).strip()
        elif step in {"local_tool_detect", "local_tool_solve"}:
            match = TOOL_RE.search(content)
            if match:
                local_tool_name = match.group(1).strip()
        if step in {"model_call", "retry_model_call"}:
            api_call_count += 1
            if step == "model_call":
                first_thinking_mode_requested = _boolean_value(event.get("thinking_mode_requested"))
                first_thinking_mode_applied = _boolean_value(event.get("thinking_mode_applied"))
            else:
                retry_thinking_mode_requested = _boolean_value(event.get("thinking_mode_requested"))
                retry_thinking_mode_applied = _boolean_value(event.get("thinking_mode_applied"))
        if step == "model_call" and content.lower().startswith("error:"):
            model_call_error = content[:200]
        if step == "retry_model_call":
            retry_triggered = True
        if step == "retry_decision":
            reasons_match = re.search(r"reasons=\[([^\]]*)\]", content)
            if reasons_match:
                retry_reason_codes = re.findall(r"[A-Za-z][A-Za-z0-9_]*", reasons_match.group(1))
        if step in {"verify", "retry_verify"}:
            diagnostics = first_attempt_diagnostics if step == "verify" else retry_attempt_diagnostics
            match = STATUS_RE.search(content)
            if match:
                mathematical_verification_statuses.append(match.group(1).lower())
            reason_match = REASON_RE.search(content)
            if reason_match:
                mathematical_verification_reasons.append(reason_match.group(1))
            subreason_match = SUBREASON_RE.search(content)
            if subreason_match:
                value = subreason_match.group(1)
                mathematical_verification_subreasons.append(None if value.lower() in {"none", "null"} else value)
            severity_match = SEVERITY_RE.search(content)
            if severity_match:
                mathematical_verification_severities.append(severity_match.group(1))
            for name in (
                "verification_reason",
                "verification_subreason",
                "verifier_name",
                "problem_parse_status",
                "candidate_parse_status",
                "computed_value_summary",
                "candidate_value_summary",
            ):
                source_name = "reason" if name == "verification_reason" else name
                diagnostics[name] = _content_value(content, source_name)
            diagnostics["verifier_applicable"] = _boolean_value(
                _content_value(content, "verifier_applicable")
            )
        if step in {"extract", "retry_extract"}:
            diagnostics = first_attempt_diagnostics if step == "extract" else retry_attempt_diagnostics
            extracted_type = _content_value(content, "extracted_answer_type")
            if extracted_type:
                diagnostics["extracted_answer_type"] = extracted_type
        if step in {"output_quality_check", "retry_output_quality_check"}:
            diagnostics = first_attempt_diagnostics if step == "output_quality_check" else retry_attempt_diagnostics
            match = STATUS_RE.search(content)
            if match:
                output_quality_statuses.append(match.group(1).lower())
            reason_match = REASON_RE.search(content)
            if reason_match:
                output_quality_reasons.append(reason_match.group(1))
            subreason_match = SUBREASON_RE.search(content)
            if subreason_match:
                value = subreason_match.group(1)
                output_quality_subreasons.append(None if value.lower() in {"none", "null"} else value)
            diagnostics["quality_reason"] = _content_value(content, "reason") or "unknown"
            diagnostics["quality_subreason"] = _content_value(content, "subreason")
            for name in ("tag_status", "latex_balance_status", "ending_status"):
                diagnostics[name] = _content_value(content, name) or "unknown"
            for name in ("raw_chars", "clean_candidate_chars"):
                try:
                    diagnostics[name] = int(_content_value(content, name) or 0)
                except ValueError:
                    diagnostics[name] = 0
        if step == "finalize":
            traced_final_acceptance = _boolean_value(_content_value(content, "final_acceptance_passed"))
    mathematical_verification_status = (
        mathematical_verification_statuses[-1]
        if mathematical_verification_statuses
        else "not_evaluated"
    )
    mathematical_verification_reason = (
        mathematical_verification_reasons[-1] if mathematical_verification_reasons else None
    )
    mathematical_verification_subreason = (
        mathematical_verification_subreasons[-1] if mathematical_verification_subreasons else None
    )
    mathematical_verification_severity = (
        mathematical_verification_severities[-1] if mathematical_verification_severities else None
    )
    mathematical_verification_passed = mathematical_verification_status == "passed"
    output_quality_passed = output_quality_statuses[-1] == "passed" if output_quality_statuses else None
    output_quality_reason = output_quality_reasons[-1] if output_quality_reasons else None
    output_quality_subreason = output_quality_subreasons[-1] if output_quality_subreasons else None
    accepted_statuses = {"passed", "unknown", "not_applicable"}
    inferred_acceptance = (
        output_quality_passed and mathematical_verification_status in accepted_statuses
        if output_quality_passed is not None
        else None
    )
    final_acceptance_passed = (
        traced_final_acceptance if traced_final_acceptance is not None else inferred_acceptance
    )
    answer_reliability_status, manual_review_required = _answer_reliability(
        mathematical_verification_status,
        output_quality_passed,
    )
    return {
        "predicted_domain": predicted_domain,
        "solver_key": solver_key,
        "routing_confidence": routing_confidence,
        "matched_signal_categories": matched_signal_categories,
        "runner_up_domain": runner_up_domain,
        "score_margin": score_margin,
        "response_mode": response_mode,
        "expected_answer_type": expected_answer_type,
        "local_tool_name": local_tool_name,
        "api_call_count": api_call_count,
        "retry_triggered": retry_triggered,
        "verification_passed": final_acceptance_passed,
        "final_acceptance_passed": final_acceptance_passed,
        "pipeline_acceptance": final_acceptance_passed,
        "pipeline_acceptance_passed": final_acceptance_passed,
        "mathematical_verification_status": mathematical_verification_status,
        "mathematical_verification_reason": mathematical_verification_reason,
        "mathematical_verification_subreason": mathematical_verification_subreason,
        "mathematical_verification_severity": mathematical_verification_severity,
        "mathematical_verification_passed": mathematical_verification_passed,
        "answer_reliability_status": answer_reliability_status,
        "manual_review_required": manual_review_required,
        "output_quality_passed": output_quality_passed,
        "output_quality_reason": output_quality_reason,
        "output_quality_subreason": output_quality_subreason,
        "first_thinking_mode_requested": first_thinking_mode_requested,
        "first_thinking_mode_applied": first_thinking_mode_applied,
        "retry_thinking_mode_requested": retry_thinking_mode_requested,
        "retry_thinking_mode_applied": retry_thinking_mode_applied,
        "retry_reason_codes": retry_reason_codes,
        "first_attempt_diagnostics": first_attempt_diagnostics,
        "retry_attempt_diagnostics": retry_attempt_diagnostics,
        "trace_summary": trace_summary,
        "model_call_error": model_call_error,
    }


def _base_result(item: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "idx": str(item.get("idx") or ""),
        "source_idx": str(item.get("source_idx") or ""),
        "problem": str(item.get("problem") or ""),
        "expected_answer": str(item.get("expected_answer") or ""),
        "label_status": str(item.get("label_status") or "unreviewed"),
        "review_note": str(item.get("review_note") or ""),
        "expected_domain": str(item.get("expected_domain") or "unknown"),
        "model": model,
    }


def evaluate_item(
    item: dict[str, Any],
    client: Any,
    use_subject_hint: bool = False,
    agent_factory: Callable[..., Any] = ReasoningAgent,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Evaluate one item; no gold or review field crosses the solve boundary."""
    started = time.perf_counter()
    base = _base_result(item, model)
    try:
        agent = agent_factory(client=client)
        result = agent.solve(str(item.get("problem") or ""), build_solve_metadata(item, use_subject_hint))
        if not isinstance(result, dict):
            raise TypeError("ReasoningAgent.solve returned a non-object result")
        final_response = str(result.get("final_response") or "")
        clean_candidate_tail = re.sub(r"\s+", " ", str(result.get("clean_candidate_tail") or "")).strip()[-160:]
        if any(
            marker in clean_candidate_tail.lower()
            for marker in ("thinking process", "system instruction", "user prompt", "model identity", "api_key", "bearer")
        ):
            clean_candidate_tail = ""
        summary = _trace_parts(result.get("trace"))
        model_call_error = summary.pop("model_call_error", None)
        trace_first_diagnostics = summary.get("first_attempt_diagnostics")
        trace_retry_diagnostics = summary.get("retry_attempt_diagnostics")
        first_attempt_diagnostics = _sanitize_attempt_diagnostics(
            result.get("first_attempt_diagnostics"), trace_first_diagnostics
        )
        retry_attempt_diagnostics = _sanitize_attempt_diagnostics(
            result.get("retry_attempt_diagnostics"), trace_retry_diagnostics
        )
        summary["first_attempt_diagnostics"] = first_attempt_diagnostics
        summary["retry_attempt_diagnostics"] = retry_attempt_diagnostics
        base.update(summary)
        structured_status = str(result.get("mathematical_verification_status") or "").lower()
        if structured_status in {"passed", "failed", "unknown", "not_applicable", "not_evaluated"}:
            base["mathematical_verification_status"] = structured_status
            base["mathematical_verification_passed"] = bool(
                result.get("mathematical_verification_passed")
            )
            base["mathematical_verification_reason"] = result.get(
                "mathematical_verification_reason"
            )
            base["mathematical_verification_subreason"] = result.get(
                "mathematical_verification_subreason"
            )
            base["mathematical_verification_severity"] = result.get(
                "mathematical_verification_severity"
            )
        if "pipeline_acceptance_passed" in result:
            structured_acceptance = bool(result.get("pipeline_acceptance_passed"))
            base["pipeline_acceptance_passed"] = structured_acceptance
            base["pipeline_acceptance"] = structured_acceptance
            base["final_acceptance_passed"] = structured_acceptance
            base["verification_passed"] = structured_acceptance
        reliability, manual_review_required = _answer_reliability(
            base.get("mathematical_verification_status"),
            base.get("output_quality_passed"),
        )
        base["answer_reliability_status"] = reliability
        base["manual_review_required"] = manual_review_required
        if base["response_mode"] == "unknown":
            base["response_mode"] = str(item.get("response_mode") or "unknown")
        proof_risk_signals = [
            str(signal)
            for signal in result.get("proof_risk_signals", [])
            if str(signal) in {
                "unjustified_inequality_direction_change",
                "conclusion_not_supported_by_previous_display",
                "upper_bound_used_as_lower_bound",
            }
        ]
        base.update(
            {
                "final_response": final_response,
                "final_response_nonempty": bool(final_response.strip()),
                "clean_candidate_tail": clean_candidate_tail or None,
                "fallback_used": final_response.strip() == FALLBACK_RESPONSE.strip(),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "status": "runtime_error" if model_call_error else "success",
                "error": "ReasoningAgent trace reported a model_call error" if model_call_error else None,
                "proof_risk_signals": proof_risk_signals,
                "proof_review_required": bool(proof_risk_signals),
                "deterministic_answer_override": bool(
                    result.get("deterministic_answer_override")
                ),
                "override_verifier_name": (
                    str(result.get("override_verifier_name"))[:80]
                    if result.get("override_verifier_name")
                    else None
                ),
            }
        )
        rejected_diagnostics = (
            retry_attempt_diagnostics
            if base.get("retry_triggered") and retry_attempt_diagnostics.get("raw_chars", 0) > 0
            else first_attempt_diagnostics
        )
        rejected_candidate = (
            base.get("output_quality_passed") is True
            and base.get("mathematical_verification_status") == "failed"
        )
        base.update(
            {
                "rejected_candidate_chars": (
                    rejected_diagnostics.get("clean_candidate_chars", 0) if rejected_candidate else 0
                ),
                "rejected_candidate_tail": (
                    rejected_diagnostics.get("candidate_tail") if rejected_candidate else None
                ),
                "extracted_short_answer": (
                    rejected_diagnostics.get("extracted_answer_summary")
                    if base.get("response_mode") == "short_answer"
                    else None
                ),
                "short_answer_verification_reason": (
                    base.get("mathematical_verification_reason") if rejected_candidate else None
                ),
                "short_answer_verification_subreason": (
                    base.get("mathematical_verification_subreason") if rejected_candidate else None
                ),
            }
        )
    except Exception as exc:
        base.update(
            {
                "predicted_domain": "unknown",
                "solver_key": "unknown",
                "response_mode": str(item.get("response_mode") or "unknown"),
                "expected_answer_type": "unknown",
                "final_response": "",
                "final_response_nonempty": False,
                "clean_candidate_tail": None,
                "local_tool_name": None,
                "api_call_count": 0,
                "retry_triggered": False,
                "verification_passed": False,
                "final_acceptance_passed": False,
                "pipeline_acceptance": False,
                "pipeline_acceptance_passed": False,
                "mathematical_verification_status": "not_evaluated",
                "mathematical_verification_reason": "runtime_error",
                "mathematical_verification_subreason": "runtime_error",
                "mathematical_verification_severity": "high",
                "mathematical_verification_passed": False,
                "answer_reliability_status": "unavailable",
                "manual_review_required": True,
                "output_quality_passed": False,
                "output_quality_reason": "runtime_error",
                "output_quality_subreason": None,
                "routing_confidence": "unknown",
                "matched_signal_categories": [],
                "runner_up_domain": None,
                "score_margin": None,
                "trace_summary": [],
                "first_thinking_mode_requested": None,
                "first_thinking_mode_applied": None,
                "retry_thinking_mode_requested": None,
                "retry_thinking_mode_applied": None,
                "retry_reason_codes": [],
                "first_attempt_diagnostics": _empty_attempt_diagnostics(),
                "retry_attempt_diagnostics": _empty_attempt_diagnostics(),
                "proof_risk_signals": [],
                "proof_review_required": False,
                "deterministic_answer_override": False,
                "override_verifier_name": None,
                "rejected_candidate_chars": 0,
                "rejected_candidate_tail": None,
                "extracted_short_answer": None,
                "short_answer_verification_reason": None,
                "short_answer_verification_subreason": None,
                "fallback_used": False,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "status": "runtime_error",
                "error": f"{type(exc).__name__}: {safe_text(exc, 300)}",
            }
        )
    return base


def _successful_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    for row in load_jsonl(path):
        if row.get("status") == "success" and row.get("idx") is not None:
            completed.add(str(row["idx"]))
    return completed


def _selected_items(
    rows: list[dict[str, Any]],
    start_index: int,
    limit: int | None,
    domain: str | None,
    completed: set[str],
    idxs: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected = rows[max(0, start_index) :]
    if domain:
        selected = [row for row in selected if str(row.get("expected_domain")) == domain]
    if idxs:
        selected = [row for row in selected if str(row.get("idx")) in idxs]
    selected = [row for row in selected if str(row.get("idx")) not in completed]
    return selected[:limit] if limit is not None else selected


def _print_progress(result: dict[str, Any], done: int, total: int) -> None:
    answer = safe_text(result.get("final_response"), 160).replace("\n", " ")
    print(
        f"[{done}/{total}] idx={result.get('idx')} domain={result.get('predicted_domain')} "
        f"solver_key={result.get('solver_key')} local_tool={result.get('local_tool_name')} "
        f"api_calls={result.get('api_call_count')} retry={result.get('retry_triggered')} "
        f"elapsed={result.get('elapsed_seconds')}s status={result.get('status')} final={answer!r}",
        flush=True,
    )


def run_evaluation(
    input_path: str | Path,
    output_path: str | Path,
    client: Any,
    *,
    concurrency: int = 1,
    limit: int | None = None,
    start_index: int = 0,
    domain: str | None = None,
    resume: bool = True,
    use_subject_hint: bool = False,
    agent_factory: Callable[..., Any] = ReasoningAgent,
    model: str = DEFAULT_MODEL,
    idxs: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    output = _resolve(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _successful_ids(output) if resume else set()
    rows = load_jsonl(input_path)
    selected = _selected_items(rows, start_index, limit, domain, completed, set(idxs or ()))
    mode = "a" if resume else "w"
    written: list[dict[str, Any]] = []
    with output.open(mode, encoding="utf-8") as stream:
        def record(result: dict[str, Any]) -> None:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            written.append(result)
            _print_progress(result, len(written), len(selected))

        if concurrency == 1:
            for item in selected:
                record(evaluate_item(item, client, use_subject_hint, agent_factory, model))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(evaluate_item, item, client, use_subject_hint, agent_factory, model): item
                    for item in selected
                }
                for future in as_completed(futures):
                    record(future.result())
    return written


def build_client(model: str, timeout: float) -> RealInternClient:
    # Keep the historical environment name synchronized for helpers and logs;
    # the public client constructor below also receives the requested ID.
    os.environ["INTERN_S1_MODEL"] = model
    _load_env_file()
    api_key = os.getenv("INTERN_S1_API_KEY", "")
    if not api_key:
        raise RuntimeError("INTERN_S1_API_KEY is not configured")
    return RealInternClient(
        api_key=api_key,
        base_url=os.getenv("INTERN_S1_BASE_URL", DEFAULT_BASE_URL),
        model=model,
        timeout=timeout,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--idx", action="append", default=None, help="Only evaluate the selected idx; repeatable.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--use-subject-hint", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"requested_model={args.model}", flush=True)
    client = build_client(args.model, args.timeout)
    run_evaluation(
        args.input,
        args.output,
        client,
        concurrency=args.concurrency,
        limit=args.limit,
        start_index=args.start_index,
        domain=args.domain,
        resume=args.resume,
        use_subject_hint=args.use_subject_hint,
        model=args.model,
        idxs=getattr(args, "idx", None),
    )


if __name__ == "__main__":
    main()
