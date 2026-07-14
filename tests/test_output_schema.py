import json
import os
import subprocess
import sys

from runner import OUTPUT_FILE


def _run_runner():
    env = os.environ.copy()
    env["AGENT_SYSTEM_FAKE_LLM"] = "1"
    env.pop("INTERN_S1_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    return subprocess.run(
        [sys.executable, "runner.py"],
        check=True,
        timeout=10,
        capture_output=True,
        text=True,
        env=env,
    )


def test_output_schema_fields():
    if not OUTPUT_FILE.exists():
        _run_runner()

    results = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    if any("solver" not in item or "attempts" not in item for item in results):
        _run_runner()
        results = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))

    required_fields = {
        "problem_id",
        "problem",
        "domain",
        "solver",
        "plan",
        "solution",
        "final_answer",
        "answer_type",
        "fallback_final_answer",
        "model_call_status",
        "explanation",
        "verification",
        "confidence",
        "attempts",
        "time_cost_seconds",
    }

    for item in results:
        assert required_fields.issubset(item.keys())
        assert isinstance(item["plan"], list)
        assert item["final_answer"] is None or isinstance(item["final_answer"], str)
        assert item["model_call_status"] in {"success", "failed"}
        assert isinstance(item["attempts"], list)
        assert isinstance(item["time_cost_seconds"], float)
        assert 0 <= item["confidence"] <= 1
        assert isinstance(item["verification"], dict)
