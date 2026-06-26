import argparse
import json
import time
from pathlib import Path

from config import PIPELINE_MAX_ATTEMPTS
from agents.answer_extractor_agent import extract_fallback_final_answer, extract_final_answer
from agents.classifier_agent import classify_problem
from agents.explainer_agent import explain_solution
from agents.planner_agent import make_plan
from agents.solver_agent import build_solver_prompt, solve_problem
from agents.verifier_agent import verify_solution
from intern_s1_client import classify_intern_s1_error


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "sample_questions.json"
OUTPUT_FILE = ROOT / "outputs" / "results.json"
LOG_FILE = ROOT / "logs" / "run_log.jsonl"
SOLVER_NAME = "intern-s1"


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _model_call_status(solution: str) -> str:
    lower_solution = solution.strip().lower()
    if lower_solution.startswith("[intern-s1 error]") or lower_solution.startswith("[mock intern-s1]"):
        return "failed"
    return "success"


def _confidence(model_call_status: str, extract_status: str, verification: dict, fallback_used: bool) -> float:
    if model_call_status == "failed" or fallback_used:
        return 0.2
    if verification.get("status") == "passed" and extract_status == "passed":
        return 0.85
    if extract_status == "uncertain":
        return 0.55
    return 0.2


def _issue(model_call_status: str, extraction: dict, verification: dict, solution: str) -> str | None:
    issues = []
    if model_call_status == "failed":
        issues.append("model_call_status failed")
    if extraction.get("status") != "passed":
        issues.append(f"final_answer extraction {extraction.get('status')}: {extraction.get('reason')}")
    if verification.get("status") != "passed":
        issues.append(f"verification {verification.get('status')}: {verification.get('feedback')}")
    if len(solution.strip()) < 20:
        issues.append("solution too short")
    return "; ".join(issues) if issues else None


def _safe_summary(text: str, limit: int = 300) -> str:
    return text.replace("Authorization", "[redacted]").replace("Bearer", "[redacted]")[:limit]


def _error_type(solution: str) -> str | None:
    if not solution.lower().startswith("[intern-s1 error]"):
        return None
    return classify_intern_s1_error(solution)


def _run_step(name: str, func, *args):
    started = time.perf_counter()
    try:
        value = func(*args)
        status = value.get("status", "ok") if isinstance(value, dict) else "ok"
        error = ""
    except Exception as exc:
        value = None
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    return value, {
        "step": name,
        "status": status,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "error": error,
    }


def _run_attempt(
    problem: str,
    domain: str,
    plan: list[str],
    round_number: int,
    max_attempts: int,
    retry_context: str | None,
):
    attempt_started = time.perf_counter()
    steps = []
    prompt_chars = len(build_solver_prompt(problem, domain, plan, retry_context))

    print(f"attempt {round_number}/{max_attempts}", flush=True)
    print("calling Intern-S1 ...", flush=True)
    solution, step_log = _run_step("solve", solve_problem, problem, domain, plan, retry_context)
    steps.append(step_log)
    solution = solution or ""
    model_call_status = _model_call_status(solution)
    print(f"Intern-S1 returned status={model_call_status}", flush=True)

    extraction, step_log = _run_step("extract_final_answer", extract_final_answer, problem, solution, domain)
    steps.append(step_log)
    extraction = extraction or {
        "final_answer": None,
        "answer_type": "unknown",
        "status": "failed",
        "reason": "答案抽取步骤失败。",
    }
    final_answer = extraction.get("final_answer") if model_call_status == "success" else None

    verification, step_log = _run_step("verify", verify_solution, problem, solution, final_answer)
    steps.append(step_log)
    verification = verification or {
        "status": "failed",
        "checks": [],
        "feedback": "验证步骤失败。",
    }
    if model_call_status == "failed":
        verification["status"] = "failed"

    issue = _issue(model_call_status, extraction, verification, solution)
    attempt = {
        "round": round_number,
        "model_call_status": model_call_status,
        "extract_status": extraction.get("status", "failed"),
        "verification_status": verification.get("status", "failed"),
        "error_type": _error_type(solution),
        "raw_error_summary": _safe_summary(solution) if model_call_status == "failed" else "",
        "prompt_chars": prompt_chars,
        "issue": issue,
        "time_cost_seconds": round(time.perf_counter() - attempt_started, 4),
    }
    return {
        "solution": solution,
        "model_call_status": model_call_status,
        "extraction": extraction,
        "verification": verification,
        "final_answer": final_answer,
        "attempt": attempt,
        "steps": steps,
    }


def run_pipeline(
    input_path: str | Path = DATA_FILE,
    output_path: str | Path = OUTPUT_FILE,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
    max_attempts: int | None = None,
) -> list[dict]:
    input_file = _resolve_path(input_path)
    output_file = _resolve_path(output_path)
    questions = json.loads(input_file.read_text(encoding="utf-8"))
    if limit is not None:
        questions = questions[:limit]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    attempt_limit = max(1, max_attempts if max_attempts is not None else PIPELINE_MAX_ATTEMPTS)

    results = []
    with LOG_FILE.open("w", encoding="utf-8") as log_file:
        for index, item in enumerate(questions):
            problem_started = time.perf_counter()
            problem_id = item["problem_id"]
            problem = item["problem"]
            step_logs = []
            print(f"[{index + 1}/{len(questions)}] Solving {problem_id} ...", flush=True)

            try:
                classification, step_log = _run_step("classify", classify_problem, problem)
                step_logs.append(step_log)
                classification = classification or {"domain": "unknown", "reason": "分类步骤失败。"}
                domain = classification["domain"]

                plan, step_log = _run_step("plan", make_plan, problem, domain)
                step_logs.append(step_log)
                plan = plan or ["提取题目条件。", "尝试求解。", "检查答案。"]

                attempts = []
                retry_context = None
                last_attempt = None
                for round_number in range(1, attempt_limit + 1):
                    last_attempt = _run_attempt(
                        problem,
                        domain,
                        plan,
                        round_number,
                        attempt_limit,
                        retry_context,
                    )
                    attempts.append(last_attempt["attempt"])
                    step_logs.extend(last_attempt["steps"])
                    if (
                        last_attempt["model_call_status"] == "success"
                        and last_attempt["extraction"].get("status") == "passed"
                        and last_attempt["verification"].get("status") == "passed"
                    ):
                        break
                    retry_context = last_attempt["attempt"]["issue"]

                solution = last_attempt["solution"] if last_attempt else ""
                model_call_status = last_attempt["model_call_status"] if last_attempt else "failed"
                extraction = last_attempt["extraction"] if last_attempt else {}
                verification = last_attempt["verification"] if last_attempt else {"status": "failed", "checks": [], "feedback": "无尝试记录。"}
                final_answer = extraction.get("final_answer") if model_call_status == "success" else None
                answer_type = extraction.get("answer_type", "unknown") if final_answer else "unknown"
                fallback_final_answer = None
                if model_call_status == "failed" or not final_answer:
                    fallback_final_answer = extract_fallback_final_answer(problem)
                fallback_used = fallback_final_answer is not None and not final_answer

                explanation, step_log = _run_step(
                    "explain",
                    explain_solution,
                    problem,
                    solution,
                    plan,
                    final_answer,
                )
                step_logs.append(step_log)
                explanation = explanation or ""
                confidence = _confidence(
                    model_call_status,
                    extraction.get("status", "failed"),
                    verification,
                    fallback_used,
                )
            except KeyboardInterrupt:
                output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                raise
            except Exception as exc:
                domain = "unknown"
                plan = []
                solution = f"[pipeline error] {type(exc).__name__}: {exc}"
                final_answer = None
                answer_type = "unknown"
                fallback_final_answer = extract_fallback_final_answer(problem)
                model_call_status = "failed"
                explanation = "该题处理过程中出现 pipeline 错误，已记录诊断信息并继续后续题目。"
                verification = {"status": "failed", "checks": [], "feedback": "pipeline 单题处理失败。"}
                confidence = 0.2
                attempts = []

            time_cost_seconds = round(time.perf_counter() - problem_started, 4)
            result = {
                "problem_id": problem_id,
                "problem": problem,
                "domain": domain,
                "solver": SOLVER_NAME,
                "plan": plan,
                "solution": solution,
                "final_answer": final_answer,
                "answer_type": answer_type,
                "fallback_final_answer": fallback_final_answer,
                "model_call_status": model_call_status,
                "explanation": explanation,
                "verification": verification,
                "confidence": confidence,
                "attempts": attempts,
                "time_cost_seconds": time_cost_seconds,
            }
            results.append(result)
            output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            log_record = {
                "problem_id": problem_id,
                "status": verification.get("status", "unknown"),
                "model_call_status": model_call_status,
                "attempts": attempts,
                "total_duration_ms": round(time_cost_seconds * 1000, 3),
                "steps": step_logs,
            }
            log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
            log_file.flush()
            print(
                f"done {problem_id} status={model_call_status} "
                f"verification={verification.get('status', 'unknown')} time={time_cost_seconds:.1f}s",
                flush=True,
            )
            if sleep_seconds > 0 and index < len(questions) - 1:
                time.sleep(sleep_seconds)

    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Intern-S1 math agent pipeline.")
    parser.add_argument("--input", default=str(DATA_FILE.relative_to(ROOT)))
    parser.add_argument("--output", default=str(OUTPUT_FILE.relative_to(ROOT)))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--attempts", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_file = _resolve_path(args.output)
    print("Running pipeline", flush=True)
    print(f"input={args.input}", flush=True)
    print(f"output={args.output}", flush=True)
    print(f"limit={args.limit}", flush=True)
    print(f"sleep={args.sleep}", flush=True)
    print(f"attempts={args.attempts if args.attempts is not None else PIPELINE_MAX_ATTEMPTS}", flush=True)
    run_pipeline(
        args.input,
        output_file,
        limit=args.limit,
        sleep_seconds=args.sleep,
        max_attempts=args.attempts,
    )
    print(f"Results written to: {output_file}")


if __name__ == "__main__":
    main()
