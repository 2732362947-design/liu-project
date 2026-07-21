"""Conservatively score Omni-MATH run results without symbolic algebra."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
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
NO_SOLUTION = {
    "no solution",
    "no solutions",
    "does not exist",
    "nonexistent",
    "impossible",
    "无解",
    "不存在",
    "没有解",
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


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("−", "-").replace("–", "-").replace("，", ",")
    text = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", r"\1/\2", text)
    text = re.sub(r"\\pmod\{([+-]?\d+)\}", r"(mod \1)", text)
    text = text.replace("\\equiv", "≡").replace("\\%", "%")
    text = text.replace("$", "").replace("`", "")
    text = re.sub(r"^(?:final\s+answer|answer|最终答案|答案)\s*[:：]\s*", "", text, flags=re.I)
    return text.strip().strip(".。;；").strip()


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


def _number(value: str) -> Fraction | None:
    text = value.strip()
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", text):
        text = text.replace(",", "")
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


def parse_simple_answer(value: Any) -> tuple[str, Any] | None:
    text = _clean(value)
    lower = re.sub(r"\s+", " ", text.lower()).strip()
    compact = re.sub(r"\s+", "", lower)
    if lower in NO_SOLUTION or compact in {item.replace(" ", "") for item in NO_SOLUTION}:
        return ("no_solution", True)

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

    pair = re.fullmatch(rf"\(\s*({NUMBER}(?:\s*/\s*{NUMBER})?)\s*,\s*({NUMBER}(?:\s*/\s*{NUMBER})?)\s*\)", lower)
    if pair:
        first, second = _number(pair.group(1)), _number(pair.group(2))
        if first is not None and second is not None:
            return ("ordered_pair", (first, second))

    assignment = re.fullmatch(rf"[a-z]\s*=\s*({NUMBER}(?:\s*/\s*{NUMBER})?%?)", lower)
    if assignment:
        lower = assignment.group(1)

    if lower.endswith("%"):
        numeric = _number(lower[:-1].strip())
        return ("number", numeric / 100) if numeric is not None else None
    numeric = _number(lower)
    return ("number", numeric) if numeric is not None else None


def compare_answers(expected: Any, predicted: Any) -> bool | None:
    """Return True/False for safely decidable answers and None for manual review."""
    expected_parsed = parse_simple_answer(expected)
    predicted_parsed = parse_simple_answer(predicted)
    if expected_parsed is None or predicted_parsed is None:
        return None
    if expected_parsed[0] == predicted_parsed[0]:
        return expected_parsed[1] == predicted_parsed[1]
    if expected_parsed[0] == predicted_parsed[0] == "number":  # pragma: no cover - explicit clarity
        return expected_parsed[1] == predicted_parsed[1]
    return False


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
    comparison = compare_answers(item.get("expected_answer"), final_response)
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
