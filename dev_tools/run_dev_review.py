import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev_tools.intern_math_checker import check_math_result

DEFAULT_RESULTS_FILE = ROOT / "outputs" / "results.json"
DEFAULT_SUMMARY_FILE = ROOT / "outputs" / "review_summary.json"


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_question_refs(questions_path: str | Path | None) -> dict:
    path = _resolve_path(questions_path)
    if path is None or not path.exists():
        return {}
    questions = json.loads(path.read_text(encoding="utf-8"))
    return _question_refs_from_list(questions)


def _question_refs_from_list(questions: list[dict] | None) -> dict:
    if not questions:
        return {}
    return {item["problem_id"]: item for item in questions}


def _with_reference(item: dict, question_refs: dict) -> dict:
    merged = dict(item)
    question = question_refs.get(item.get("problem_id"), {})
    if "expected_answer" in question:
        merged["expected_answer"] = question["expected_answer"]
    if "answer_type" in question:
        merged["expected_answer_type"] = question["answer_type"]
    return merged


def _suggested_action(
    model_failed_count: int,
    math_failed_count: int,
    math_unknown_count: int,
) -> str:
    if model_failed_count > 0:
        return "check_network_or_api"
    if math_failed_count > 0:
        return "improve_solver_or_prompt"
    if math_unknown_count > 0:
        return "review_unknown"
    return "continue"


def _has_final_answer(item: dict) -> bool:
    return bool(item.get("final_answer"))


def _skipped_math_check(reason: str) -> dict:
    return {
        "status": "unknown",
        "uses_fallback": False,
        "method": "skipped",
        "reason": reason,
    }


def build_review_summary(results: list[dict], questions: list[dict] | None = None) -> dict:
    question_refs = _question_refs_from_list(questions)
    reviews = []

    model_failed_count = sum(1 for item in results if item.get("model_call_status") == "failed")
    api_failed_count = model_failed_count
    uncertain_count = sum(
        1
        for item in results
        if item.get("verification", {}).get("status") == "uncertain"
        or any(attempt.get("extract_status") == "uncertain" for attempt in item.get("attempts", []))
    )
    fallback_count = sum(1 for item in results if item.get("fallback_final_answer"))
    math_passed_count = 0
    math_failed_count = 0
    math_unknown_count = 0
    effective_evaluated_count = 0

    for item in results:
        is_effective = item.get("model_call_status") == "success" and _has_final_answer(item)
        if is_effective:
            effective_evaluated_count += 1
            item_with_ref = _with_reference(item, question_refs)
            math_check = check_math_result(item_with_ref)
            if math_check["status"] == "passed":
                math_passed_count += 1
            elif math_check["status"] == "failed":
                math_failed_count += 1
            else:
                math_unknown_count += 1
        elif item.get("model_call_status") == "failed":
            math_check = _skipped_math_check("model call failed")
        else:
            math_check = _skipped_math_check("final_answer is empty")

        reviews.append(
            {
                "problem_id": item.get("problem_id"),
                "model_call_status": item.get("model_call_status"),
                "verification_status": item.get("verification", {}).get("status"),
                "attempt_count": len(item.get("attempts", [])),
                "math_check": math_check,
            }
        )

    total = len(results)
    estimated_accuracy = (
        math_passed_count / effective_evaluated_count if effective_evaluated_count else None
    )
    return {
        "suggested_action": _suggested_action(
            model_failed_count,
            math_failed_count,
            math_unknown_count,
        ),
        "total": total,
        "model_failed_count": model_failed_count,
        "api_failed_count": api_failed_count,
        "uncertain_count": uncertain_count,
        "fallback_count": fallback_count,
        "effective_evaluated_count": effective_evaluated_count,
        "math_passed_count": math_passed_count,
        "math_failed_count": math_failed_count,
        "math_unknown_count": math_unknown_count,
        "estimated_accuracy": estimated_accuracy,
        "reviews": reviews,
    }


def run_review(
    results_path: str | Path = DEFAULT_RESULTS_FILE,
    output_path: str | Path = DEFAULT_SUMMARY_FILE,
    questions_path: str | Path | None = None,
) -> dict:
    results_file = _resolve_path(results_path)
    output_file = _resolve_path(output_path)
    results = json.loads(results_file.read_text(encoding="utf-8"))
    questions = list(_load_question_refs(questions_path).values())
    summary = build_review_summary(results, questions)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review math agent result files.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_FILE.relative_to(ROOT)))
    parser.add_argument("--questions", default=None)
    parser.add_argument("--output", default=str(DEFAULT_SUMMARY_FILE.relative_to(ROOT)))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_file = _resolve_path(args.output)
    run_review(args.results, output_file, args.questions)
    print(f"Review summary written to: {output_file}")


if __name__ == "__main__":
    main()
