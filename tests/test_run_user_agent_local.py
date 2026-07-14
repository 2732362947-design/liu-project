import json
import inspect
import subprocess
import sys

import user_agent
from dev_tools.run_user_agent_local import (
    FakeClient,
    build_metadata,
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


def _trace_content(result, step):
    return next(item["content"] for item in result["trace"] if item["step"] == step)


def test_build_metadata_prefers_nested_metadata():
    item = {
        "idx": "row-id",
        "problem": "题目",
        "expected_answer": "999",
        "metadata": {
            "idx": "nested-id",
            "domain": "geometry",
            "difficulty": 4.5,
            "source": "china_team_selection_test",
            "answer_type": "expression",
            "raw_domain": "Mathematics -> Geometry",
            "solver_key": "geometry",
            "expected_answer": "secret",
            "solution": "hidden",
        },
    }

    metadata = build_metadata(item, 0)

    assert metadata == {
        "idx": "nested-id",
        "domain": "geometry",
        "difficulty": 4.5,
        "source": "china_team_selection_test",
        "answer_type": "expression",
        "raw_domain": "Mathematics -> Geometry",
        "solver_key": "geometry",
    }


def test_run_one_passes_nested_expression_metadata_to_agent():
    agent = ReasoningAgent(FakeClient("最终答案：x=2"))
    item = {
        "idx": "omni_000002",
        "problem": "Find x.",
        "metadata": {
            "idx": "omni_000002",
            "domain": "geometry",
            "answer_type": "expression",
            "solver_key": "geometry",
        },
    }

    result = run_one(agent, item, 0)

    assert result["status"] == "success"
    assert result["final_response"] == "x=2"
    assert "domain=geometry" in _trace_content(result, "classify")
    assert "expected_answer_type=expression" in _trace_content(result, "verify")


def test_default_fake_client_returns_expression_for_expression_metadata():
    agent = ReasoningAgent(FakeClient())
    item = {
        "idx": "expr",
        "problem": "Find an expression.",
        "metadata": {"domain": "geometry", "answer_type": "expression", "solver_key": "geometry"},
    }

    result = run_one(agent, item, 0)

    assert result["status"] == "success"
    assert result["final_response"] == "x+1"
    assert result["final_response"] != "2"
    assert "expected_answer_type=expression" in _trace_content(result, "verify")


def test_run_one_passes_legacy_top_level_expression_metadata_to_agent():
    agent = ReasoningAgent(FakeClient("最终答案：x=2"))
    item = {
        "idx": "omni_000002",
        "problem": "Find x.",
        "domain": "geometry",
        "answer_type": "expression",
        "solver_key": "geometry",
    }

    result = run_one(agent, item, 0)

    assert result["status"] == "success"
    assert result["final_response"] == "x=2"
    assert "domain=geometry" in _trace_content(result, "classify")
    assert "expected_answer_type=expression" in _trace_content(result, "verify")


def test_run_one_passes_number_metadata_to_agent():
    agent = ReasoningAgent(FakeClient("最终答案：2"))
    item = {
        "idx": "n1",
        "problem": "1+1=?",
        "metadata": {"answer_type": "number"},
    }

    result = run_one(agent, item, 0)

    assert result["status"] == "success"
    assert result["final_response"] == "2"
    assert "expected_answer_type=number" in _trace_content(result, "verify")


def test_default_fake_client_returns_number_for_number_metadata():
    agent = ReasoningAgent(FakeClient())
    item = {
        "idx": "n1",
        "problem": "1+1=?",
        "metadata": {"answer_type": "number"},
    }

    result = run_one(agent, item, 0)

    assert result["status"] == "success"
    assert result["final_response"] == "2"
    assert "expected_answer_type=number" in _trace_content(result, "verify")


def test_local_fake_client_markers_do_not_enter_user_agent_entrypoint():
    source = inspect.getsource(user_agent)

    assert "class FakeClient" not in source
    assert "DEFAULT_FAKE_ANSWER" not in source
    assert "AUTO_FAKE_ANSWER" not in source
    assert "最终答案：x+1" not in source


def test_build_metadata_top_level_fallback_filters_answer_fields():
    item = {
        "idx": "row-1",
        "problem": "1+1=?",
        "expected_answer": "2",
        "answer": "secret",
        "solution": "hidden",
        "domain": "algebra",
        "difficulty": 3,
        "source": "test",
        "answer_type": "number",
        "raw_domain": "Mathematics -> Algebra",
        "solver_key": "algebra",
    }

    metadata = build_metadata(item, 0)

    assert "expected_answer" not in metadata
    assert "answer" not in metadata
    assert "solution" not in metadata
    assert metadata["idx"] == "row-1"
    assert metadata["domain"] == "algebra"
    assert metadata["answer_type"] == "number"


def test_build_metadata_nested_filters_all_answer_like_variants():
    item = {
        "metadata": {
            "answer": "blocked-1",
            "Answer": "blocked-2",
            "expected-answer": "blocked-3",
            "Ground Truth": "blocked-4",
            "ground_truth": "blocked-5",
            "gold": "blocked-6",
            "reference": "blocked-7",
            "official_answer": "blocked-8",
            "label": "blocked-9",
            "target": "blocked-10",
            "domain": "algebra",
            "source": "local",
        }
    }

    metadata = build_metadata(item, 3)

    assert metadata == {"idx": 3, "domain": "algebra", "source": "local"}


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
        timeout=10,
        capture_output=True,
        text=True,
    )

    result = json.loads((output_dir / "0.json").read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert result["final_response"]
