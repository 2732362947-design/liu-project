import json
import os
import subprocess
import sys

from runner import DATA_FILE, LOG_FILE, OUTPUT_FILE


def _run_runner(args=None):
    command = [sys.executable, "runner.py"]
    if args:
        command.extend(args)
    env = os.environ.copy()
    env["AGENT_SYSTEM_FAKE_LLM"] = "1"
    env.pop("INTERN_S1_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    return subprocess.run(
        command,
        check=True,
        timeout=10,
        capture_output=True,
        text=True,
        env=env,
    )


def test_runner_generates_results_file():
    completed = _run_runner()

    assert "calling Intern-S1" not in completed.stdout

    assert OUTPUT_FILE.exists()
    questions = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    saved_results = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    assert len(saved_results) == len(questions)
    assert all(item["problem_id"] for item in saved_results)
    assert all(item["solution"].strip() for item in saved_results)

    log_records = [
        json.loads(line)
        for line in LOG_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(log_records) == len(questions)
    for record in log_records:
        assert "total_duration_ms" in record
        assert "steps" in record
        assert all("duration_ms" in step and "status" in step for step in record["steps"])


def test_runner_accepts_input_and_output_args():
    output_file = "outputs/test_results.json"
    _run_runner(
        [
            "--input",
            "data/sample_questions.json",
            "--output",
            output_file,
        ]
    )

    saved_results = json.loads(open(output_file, encoding="utf-8").read())
    questions = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    assert len(saved_results) == len(questions)
    assert all(isinstance(item["attempts"], list) for item in saved_results)
    assert all(item["model_failed"] is False for item in saved_results)


def test_runner_accepts_limit_and_sleep_args():
    output_file = "outputs/test_results_limited.json"
    _run_runner(
        [
            "--input",
            "data/dev_questions.json",
            "--output",
            output_file,
            "--limit",
            "2",
            "--sleep",
            "0",
        ]
    )

    saved_results = json.loads(open(output_file, encoding="utf-8").read())
    assert len(saved_results) == 2


def test_runner_limit_one_generates_one_result():
    output_file = "outputs/test_results_one.json"
    _run_runner(
        [
            "--input",
            "data/sample_questions.json",
            "--output",
            output_file,
            "--limit",
            "1",
            "--sleep",
            "0",
        ]
    )

    saved_results = json.loads(open(output_file, encoding="utf-8").read())
    assert len(saved_results) == 1
    item = saved_results[0]
    for field in ("final_answer", "answer_type", "model_call_status", "attempts", "time_cost_seconds", "model_failed"):
        assert field in item
    assert item["model_failed"] is False


def test_runner_attempts_one_limits_attempt_count():
    output_file = "outputs/test_results_attempts_one.json"
    _run_runner(
        [
            "--input",
            "data/sample_questions.json",
            "--output",
            output_file,
            "--limit",
            "1",
            "--sleep",
            "0",
            "--attempts",
            "1",
        ]
    )

    saved_results = json.loads(open(output_file, encoding="utf-8").read())
    attempts = saved_results[0]["attempts"]
    assert len(attempts) <= 1
    assert "prompt_chars" in attempts[0]
    assert "error_type" in attempts[0]
    assert saved_results[0]["model_failed"] is False
