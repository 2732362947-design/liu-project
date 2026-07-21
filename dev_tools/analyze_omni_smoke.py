"""Build a field-safe offline analysis of an Omni-MATH smoke run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev_tools.score_omni_results import _result_reliability, is_solver_compatible, load_jsonl


REQUIRED_DATASET_FIELDS = ("expected_domain", "expected_answer")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def validate_dataset_fields(dataset: list[dict[str, Any]]) -> None:
    for position, item in enumerate(dataset, 1):
        missing = [field for field in REQUIRED_DATASET_FIELDS if field not in item]
        if missing:
            raise ValueError(f"dataset row {position} is missing required fields: {', '.join(missing)}")


def _distribution(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _matrix(rows: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for expected, actual in rows:
        matrix[expected][actual] += 1
    return {
        expected: dict(sorted(counts.items()))
        for expected, counts in sorted(matrix.items())
    }


def build_analysis(
    dataset: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    score_report: dict[str, Any],
) -> dict[str, Any]:
    validate_dataset_fields(dataset)
    results = {str(row.get("idx") or ""): row for row in result_rows}
    score_rows = score_report.get("items") if isinstance(score_report, dict) else None
    if not isinstance(score_rows, list):
        raise ValueError("score report must contain an items list")
    scores = {
        str(row.get("idx") or ""): row
        for row in score_rows
        if isinstance(row, dict)
    }

    expected_distribution: list[str] = []
    predicted_distribution: list[str] = []
    confusion_rows: list[tuple[str, str]] = []
    solver_rows: list[tuple[str, str]] = []
    unknown_items: list[dict[str, Any]] = []
    fallback_items: list[dict[str, Any]] = []
    quality_failures: list[dict[str, Any]] = []
    verifier_failures: list[dict[str, Any]] = []
    incorrect_items: list[dict[str, Any]] = []
    format_failures: list[dict[str, Any]] = []
    manual_review_items: list[dict[str, Any]] = []
    reliability_manual_review_items: list[dict[str, Any]] = []
    deterministic_override_items: list[dict[str, Any]] = []
    retry_reasons: Counter[str] = Counter()
    thinking_behavior: Counter[str] = Counter()

    for item in dataset:
        idx = str(item.get("idx") or "")
        expected_domain = str(item["expected_domain"] or "unknown")
        # Accessing expected_answer here is intentional: validation prevents the
        # old silent answer=None diagnostic caused by reading a nonexistent alias.
        expected_answer_present = bool(str(item["expected_answer"] or "").strip())
        result = results.get(idx, {})
        score = scores.get(idx, {})
        predicted_domain = str(result.get("predicted_domain") or "unknown")
        solver_key = str(result.get("solver_key") or "general")
        compatible = is_solver_compatible(expected_domain, predicted_domain, solver_key)

        expected_distribution.append(expected_domain)
        predicted_distribution.append(predicted_domain)
        confusion_rows.append((expected_domain, predicted_domain))
        solver_rows.append((expected_domain, f"{solver_key}:{'compatible' if compatible else 'incompatible'}"))

        base = {"idx": idx, "expected_domain": expected_domain, "predicted_domain": predicted_domain}
        if predicted_domain == "unknown":
            unknown_items.append(base)
        if result.get("fallback_used"):
            fallback_items.append({**base, "solver_key": solver_key, "solver_compatible": compatible})
        if result.get("output_quality_passed") is False:
            quality_failures.append(
                {
                    **base,
                    "reason": result.get("output_quality_reason"),
                    "subreason": result.get("output_quality_subreason"),
                }
            )
        if result.get("mathematical_verification_status") == "failed":
            verifier_failures.append(
                {
                    **base,
                    "reason": result.get("mathematical_verification_reason"),
                    "subreason": result.get("mathematical_verification_subreason"),
                    "extracted_short_answer": result.get("extracted_short_answer"),
                }
            )
        score_status = str(score.get("score_status") or "missing")
        if score_status == "incorrect":
            incorrect_items.append({**base, "score_reason": score.get("score_reason")})
        if score_status == "format_fail":
            format_failures.append({**base, "score_reason": score.get("score_reason")})
        if score_status in {"manual_review", "label_suspected"}:
            manual_review_items.append(
                {
                    **base,
                    "score_status": score_status,
                    "expected_answer_present": expected_answer_present,
                    "pipeline_acceptance": score.get("pipeline_acceptance"),
                    "mathematical_correctness": score.get("mathematical_correctness"),
                    "proof_review_required": score.get("proof_review_required", False),
                    "proof_risk_signals": score.get("proof_risk_signals", []),
                }
            )
        reliability = _result_reliability(result)
        manual_review_required = reliability != "verified"
        if manual_review_required:
            reliability_manual_review_items.append(
                {
                    **base,
                    "answer_reliability_status": reliability,
                    "mathematical_verification_status": result.get(
                        "mathematical_verification_status"
                    ),
                }
            )
        if result.get("deterministic_answer_override"):
            deterministic_override_items.append(
                {**base, "override_verifier_name": result.get("override_verifier_name")}
            )
        if result.get("retry_triggered"):
            reason_codes = result.get("retry_reason_codes")
            if isinstance(reason_codes, list) and reason_codes:
                retry_reasons.update(str(reason) for reason in reason_codes)
            else:
                first_diagnostics = result.get("first_attempt_diagnostics")
                first_quality_reason = (
                    first_diagnostics.get("quality_reason")
                    if isinstance(first_diagnostics, dict)
                    else None
                )
                retry_reason = str(
                    first_quality_reason
                    if first_quality_reason not in {None, "", "unknown", "passed"}
                    else result.get("mathematical_verification_reason")
                    or result.get("output_quality_reason")
                    or "unknown"
                )
                retry_reasons[retry_reason] += 1
        thinking_key = (
            f"first_requested={result.get('first_thinking_mode_requested')},"
            f"first_applied={result.get('first_thinking_mode_applied')},"
            f"retry_requested={result.get('retry_thinking_mode_requested')},"
            f"retry_applied={result.get('retry_thinking_mode_applied')}"
        )
        thinking_behavior[thinking_key] += 1

    attempted_results = [results.get(str(item.get("idx") or ""), {}) for item in dataset]
    pipeline_acceptance_count = sum(
        bool(result.get("pipeline_acceptance_passed", result.get("final_acceptance_passed")))
        for result in attempted_results
    )
    deterministically_verified_count = sum(
        result.get("mathematical_verification_status") == "passed"
        for result in attempted_results
    )
    unverified_accepted_count = sum(
        bool(result.get("pipeline_acceptance_passed", result.get("final_acceptance_passed")))
        and result.get("mathematical_verification_status") in {"not_applicable", "unknown"}
        for result in attempted_results
    )
    deterministically_rejected_count = sum(
        result.get("mathematical_verification_status") == "failed"
        for result in attempted_results
    )
    reliability_counts = Counter(_result_reliability(result) for result in attempted_results)
    return {
        "expected_domain_distribution": _distribution(expected_distribution),
        "predicted_domain_distribution": _distribution(predicted_distribution),
        "exact_confusion_matrix": _matrix(confusion_rows),
        "compatible_solver_matrix": _matrix(solver_rows),
        "unknown_items": unknown_items,
        "fallback_items": fallback_items,
        "quality_failures": quality_failures,
        "verifier_failures": verifier_failures,
        "incorrect_items": incorrect_items,
        "format_failures": format_failures,
        "manual_review_items": manual_review_items,
        "reliability_manual_review_items": reliability_manual_review_items,
        "deterministic_override_items": deterministic_override_items,
        "pipeline_acceptance_count": pipeline_acceptance_count,
        "deterministically_verified_count": deterministically_verified_count,
        "unverified_accepted_count": unverified_accepted_count,
        "deterministically_rejected_count": deterministically_rejected_count,
        "verified_count": reliability_counts["verified"],
        "unverified_count": reliability_counts["unverified"],
        "rejected_count": reliability_counts["rejected"],
        "unavailable_count": reliability_counts["unavailable"],
        "deterministic_override_count": len(deterministic_override_items),
        "manual_review_required_count": len(reliability_manual_review_items),
        "retry_reason_distribution": dict(sorted(retry_reasons.items())),
        "thinking_mode_behavior": dict(sorted(thinking_behavior.items())),
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = ["# Omni-MATH Smoke Analysis", ""]
    for title, key in (
        ("Expected domain distribution", "expected_domain_distribution"),
        ("Predicted domain distribution", "predicted_domain_distribution"),
        ("Retry reason distribution", "retry_reason_distribution"),
        ("Thinking-mode behavior", "thinking_mode_behavior"),
    ):
        lines.extend([f"## {title}", "", "```json", json.dumps(analysis[key], ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## Acceptance and reliability", "", "| Metric | Count |", "| --- | ---: |"])
    for key in (
        "pipeline_acceptance_count",
        "deterministically_verified_count",
        "unverified_accepted_count",
        "deterministically_rejected_count",
        "verified_count",
        "unverified_count",
        "rejected_count",
        "unavailable_count",
        "deterministic_override_count",
        "manual_review_required_count",
    ):
        lines.append(f"| {key} | {analysis[key]} |")
    lines.extend(["", "## Diagnostic item counts", ""])
    for key in (
        "unknown_items",
        "fallback_items",
        "quality_failures",
        "verifier_failures",
        "incorrect_items",
        "format_failures",
        "manual_review_items",
        "reliability_manual_review_items",
        "deterministic_override_items",
    ):
        lines.append(f"- {key}: {len(analysis[key])}")
    lines.extend(["", "## Exact confusion matrix", "", "```json", json.dumps(analysis["exact_confusion_matrix"], ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## Compatible solver matrix", "", "```json", json.dumps(analysis["compatible_solver_matrix"], ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def analyze_files(
    dataset_path: str | Path,
    results_path: str | Path,
    score_report_path: str | Path,
    output_json: str | Path,
    output_md: str | Path,
) -> dict[str, Any]:
    score_report = json.loads(_resolve(score_report_path).read_text(encoding="utf-8"))
    analysis = build_analysis(load_jsonl(dataset_path), load_jsonl(results_path), score_report)
    json_path = _resolve(output_json)
    md_path = _resolve(output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(analysis), encoding="utf-8")
    return analysis


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--score-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analysis = analyze_files(
        args.dataset,
        args.results,
        args.score_report,
        args.output_json,
        args.output_md,
    )
    print(json.dumps({key: len(value) for key, value in analysis.items() if isinstance(value, list)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
