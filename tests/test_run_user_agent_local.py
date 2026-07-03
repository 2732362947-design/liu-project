import json
import subprocess
import sys

from dev_tools.run_user_agent_local import (
    FakeClient,
    load_jsonl,
    normalize_agent_response,
    run_one,
    run_local,
)
from user_agent import ReasoningAgent


def test_load_jsonl_reads_items(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text('{"idx": 0, "problem": "1+1=?"}\n{"problem": "2+2=?"}\n', encoding="utf-8")

    items = load_jsonl(path)

    assert len(items) == 2
    assert items[0]["idx"] == 0
    assert "idx" not in items[1]


def test_fake_client_chat_records_call():
    client = FakeClient("最终答案：7")

    response = client.chat(messages=[{"role": "user", "content": "hi"}])

    assert response == "最终答案：7"
    assert len(client.calls) == 1


def test_run_one_success_with_reasoning_agent():
    agent = ReasoningAgent(FakeClient("推理过程\n最终答案：2"))

    result = run_one(agent, {"idx": 0, "problem": "1+1=?"}, 0)

    assert result["status"] == "success"
    assert result["final_response"]
    assert isinstance(result["trace"], list)


def test_run_one_exception_returns_error():
    class BadAgent:
        def solve(self, problem, metadata):
            raise RuntimeError("boom")

    result = run_one(BadAgent(), {"idx": 0, "problem": "x"}, 0)

    assert result["status"] == "error"
    assert result["final_response"] == ""
    assert result["error"]["type"] == "RuntimeError"


def test_normalize_agent_response_fills_missing_trace():
    normalized = normalize_agent_response({"final_response": "2"})

    assert normalized["final_response"] == "2"
    assert normalized["trace"] == []


def test_normalize_agent_response_handles_non_dict():
    normalized = normalize_agent_response("plain answer")

    assert normalized["final_response"] == "plain answer"
    assert normalized["trace"] == []


def test_run_local_skips_existing_output_without_overwrite(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    input_path.write_text('{"idx": 0, "problem": "1+1=?"}\n', encoding="utf-8")
    existing = output_dir / "0.json"
    existing.write_text('{"status": "old"}', encoding="utf-8")

    results = run_local(input_path, output_dir=output_dir, overwrite=False)

    assert results == []
    assert json.loads(existing.read_text(encoding="utf-8"))["status"] == "old"


def test_cli_smoke_generates_output(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    input_path.write_text('{"idx": 0, "problem": "1+1=?"}\n', encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "dev_tools/run_user_agent_local.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    result = json.loads((output_dir / "0.json").read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert result["final_response"]
