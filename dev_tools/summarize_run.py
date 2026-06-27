import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_FILE = ROOT / "outputs" / "dev_results.json"
DEFAULT_REVIEW_FILE = ROOT / "outputs" / "dev_review_summary.json"
DEFAULT_OUTPUT_FILE = ROOT / "outputs" / "dev_report.md"
OVERVIEW_FIELDS = (
    "total",
    "model_failed_count",
    "api_failed_count",
    "effective_evaluated_count",
    "math_passed_count",
    "math_failed_count",
    "math_unknown_count",
    "estimated_accuracy",
    "suggested_action",
)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_json(path: str | Path) -> Any:
    return json.loads(_resolve_path(path).read_text(encoding="utf-8"))


def index_review_by_problem_id(review: dict | None) -> dict:
    if not isinstance(review, dict):
        return {}
    reviews = review.get("reviews", [])
    if not isinstance(reviews, list):
        return {}
    return {
        item.get("problem_id"): item
        for item in reviews
        if isinstance(item, dict) and item.get("problem_id") is not None
    }


def _get_path(item: dict, key_path: str) -> Any:
    value: Any = item
    for key in key_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def summarize_counts(results: list[dict], key_path_or_func: str | Callable[[dict], Any]) -> Counter:
    counts: Counter = Counter()
    for item in results:
        if callable(key_path_or_func):
            value = key_path_or_func(item)
        else:
            value = _get_path(item, key_path_or_func)
        counts[str(value or "unknown")] += 1
    return counts


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value)


def _escape_table(value: Any) -> str:
    text = _format_value(value)
    return text.replace("\n", " ").replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_escape_table(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def _count_table(counts: Counter, first_header: str) -> str:
    rows = [[key, counts[key]] for key in sorted(counts)]
    return _markdown_table([first_header, "count"], rows)


def _runtime_value(item: dict) -> float | None:
    value = item.get("time_cost_seconds")
    if isinstance(value, (int, float)):
        return float(value)
    attempt_values = [
        attempt.get("time_cost_seconds")
        for attempt in item.get("attempts", [])
        if isinstance(attempt, dict) and isinstance(attempt.get("time_cost_seconds"), (int, float))
    ]
    if attempt_values:
        return float(sum(attempt_values))
    return None


def _runtime_summary(results: list[dict]) -> list[tuple[str, str]]:
    values = [value for value in (_runtime_value(item) for item in results) if value is not None]
    if not values:
        return [
            ("total_time_seconds", "N/A"),
            ("average_time_seconds", "N/A"),
            ("max_time_seconds", "N/A"),
            ("min_time_seconds", "N/A"),
        ]
    return [
        ("total_time_seconds", f"{sum(values):.4g}"),
        ("average_time_seconds", f"{sum(values) / len(values):.4g}"),
        ("max_time_seconds", f"{max(values):.4g}"),
        ("min_time_seconds", f"{min(values):.4g}"),
    ]


def _safe_text(text: Any, limit: int | None = None) -> str:
    value = str(text or "")
    value = value.replace("Authorization", "[redacted-header]")
    value = value.replace("Bearer", "[redacted-token]")
    if limit is not None and len(value) > limit:
        return value[:limit] + "..."
    return value


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe_json_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(child) for child in value]
    if isinstance(value, str):
        return _safe_text(value)
    return value


def _sample_structured_output(results: list[dict]) -> str:
    if not results:
        return "N/A"
    item = results[0]
    sample = {
        "problem_id": item.get("problem_id"),
        "domain": item.get("domain"),
        "solver_key": item.get("solver_key"),
        "model_call_status": item.get("model_call_status"),
        "final_answer": item.get("final_answer"),
        "verification": item.get("verification"),
        "confidence": item.get("confidence"),
        "time_cost_seconds": item.get("time_cost_seconds"),
        "solution": _safe_text(item.get("solution"), limit=300),
    }
    return json.dumps(_safe_json_value(sample), ensure_ascii=False, indent=2)


def _failed_or_uncertain_rows(results: list[dict], review_index: dict) -> list[list[Any]]:
    rows = []
    for item in results:
        problem_id = item.get("problem_id")
        review_item = review_index.get(problem_id, {})
        math_check = review_item.get("math_check", {}) if isinstance(review_item, dict) else {}
        math_status = math_check.get("status") if isinstance(math_check, dict) else None
        verification = item.get("verification", {})
        verification_status = verification.get("status") if isinstance(verification, dict) else None
        model_status = item.get("model_call_status")
        is_problem = (
            model_status != "success"
            or verification_status != "passed"
            or math_status in {"failed", "unknown"}
        )
        if not is_problem:
            continue
        issue = item.get("issue")
        if not issue and isinstance(verification, dict):
            issue = verification.get("suggestion") or verification.get("feedback")
        if not issue and isinstance(math_check, dict):
            issue = math_check.get("reason")
        rows.append(
            [
                problem_id,
                model_status or "unknown",
                verification_status or "unknown",
                math_status or "N/A",
                item.get("solver_key") or "unknown",
                issue or "N/A",
            ]
        )
    return rows


def build_markdown_report(results: list[dict], review: dict | None = None) -> str:
    review_index = index_review_by_problem_id(review)
    overview_rows = [[field, review.get(field) if isinstance(review, dict) else None] for field in OVERVIEW_FIELDS]
    verification_counts = summarize_counts(results, "verification.status")
    severity_counts = summarize_counts(results, "verification.severity")
    failed_rows = _failed_or_uncertain_rows(results, review_index)

    sections = [
        "# Intern-S1 Math Agent Run Report",
        "## 1. Overview",
    ]
    if review is None:
        sections.append("Review summary missing; overview fields from review are shown as N/A.")
    sections.append(_markdown_table(["metric", "value"], overview_rows))
    sections.extend(
        [
            "## 2. Pipeline Summary",
            (
                "Classifier -> Planner -> Solver Router -> Intern-S1 Solver -> Answer Extractor "
                "-> Local Verifier -> Review Summary. The pipeline records solver_key routing, "
                "uses specialized solver prompt templates, applies local verifier checks, and "
                "writes structured JSON outputs for later review."
            ),
            "## 3. Solver Routing Distribution",
            _count_table(summarize_counts(results, "solver_key"), "solver_key"),
            "## 4. Domain Distribution",
            _count_table(summarize_counts(results, "domain"), "domain"),
            "## 5. Verification Summary",
            _count_table(verification_counts, "verification_status"),
            _count_table(severity_counts, "severity"),
            "## 6. Runtime Summary",
            _markdown_table(["metric", "value"], _runtime_summary(results)),
            "## 7. Failed or Uncertain Items",
        ]
    )
    if failed_rows:
        sections.append(
            _markdown_table(
                [
                    "problem_id",
                    "model_call_status",
                    "verification_status",
                    "math_check_status",
                    "solver_key",
                    "issue",
                ],
                failed_rows,
            )
        )
    else:
        sections.append("No failed or uncertain items found.")
    sections.extend(
        [
            "## 8. Sample Structured Output",
            "```json",
            _sample_structured_output(results),
            "```",
        ]
    )
    return "\n\n".join(sections) + "\n"


def write_markdown_report(results_path: str | Path, review_path: str | Path, output_path: str | Path) -> str:
    results = load_json(results_path)
    review_file = _resolve_path(review_path)
    review = load_json(review_file) if review_file.exists() else None
    report = build_markdown_report(results if isinstance(results, list) else [], review)
    output_file = _resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an Intern-S1 math agent run as Markdown.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_FILE.relative_to(ROOT)))
    parser.add_argument("--review", default=str(DEFAULT_REVIEW_FILE.relative_to(ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE.relative_to(ROOT)))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_file = _resolve_path(args.output)
    write_markdown_report(args.results, args.review, output_file)
    print(f"Run report written to: {output_file}")


if __name__ == "__main__":
    main()
