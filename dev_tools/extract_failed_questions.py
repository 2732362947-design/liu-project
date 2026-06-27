import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev_tools.intern_math_checker import check_math_result


DEFAULT_RESULTS_FILE = ROOT / "outputs" / "official_results.json"
DEFAULT_QUESTIONS_FILE = ROOT / "data" / "official_questions.json"
DEFAULT_OUTPUT_FILE = ROOT / "outputs" / "official_failed_questions.json"


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: str | Path) -> list[dict]:
    return json.loads(_resolve_path(path).read_text(encoding="utf-8"))


def _question_refs(questions: list[dict]) -> dict:
    return {item["problem_id"]: item for item in questions}


def _with_reference(result: dict, question: dict | None) -> dict:
    merged = dict(result)
    if not question:
        return merged
    for source_key, target_key in (
        ("expected_answer", "expected_answer"),
        ("answer_type", "expected_answer_type"),
        ("problem", "problem"),
    ):
        if source_key in question:
            merged[target_key] = question[source_key]
    return merged


def _math_check_status(result: dict, question: dict | None) -> str | None:
    embedded = result.get("math_check")
    if isinstance(embedded, dict) and embedded.get("status"):
        return embedded["status"]
    if question and question.get("expected_answer"):
        return check_math_result(_with_reference(result, question)).get("status")
    return None


def should_extract_failed_question(result: dict, question: dict | None = None) -> bool:
    if result.get("model_call_status") != "success":
        return True
    if result.get("verification", {}).get("status") != "passed":
        return True
    return _math_check_status(result, question) == "failed"


def extract_failed_questions(results: list[dict], questions: list[dict]) -> list[dict]:
    questions_by_id = _question_refs(questions)
    failed_questions = []
    seen_problem_ids = set()

    for result in results:
        problem_id = result.get("problem_id")
        question = questions_by_id.get(problem_id)
        if not should_extract_failed_question(result, question):
            continue
        if problem_id in seen_problem_ids:
            continue
        seen_problem_ids.add(problem_id)
        failed_questions.append(dict(question or {"problem_id": problem_id}))

    return failed_questions


def run_extract_failed_questions(
    results_path: str | Path = DEFAULT_RESULTS_FILE,
    questions_path: str | Path = DEFAULT_QUESTIONS_FILE,
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
) -> list[dict]:
    failed_questions = extract_failed_questions(
        _load_json(results_path),
        _load_json(questions_path),
    )
    output_file = _resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(failed_questions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return failed_questions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract failed questions for reruns.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_FILE.relative_to(ROOT)))
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_FILE.relative_to(ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE.relative_to(ROOT)))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_file = _resolve_path(args.output)
    failed_questions = run_extract_failed_questions(args.results, args.questions, output_file)
    print(f"Wrote {len(failed_questions)} failed questions to: {output_file}")


if __name__ == "__main__":
    main()
