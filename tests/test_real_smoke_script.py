import json

import pytest

from dev_tools import run_user_agent_real_smoke
from dev_tools.run_user_agent_real_smoke import RealInternClient, _load_input_item, run_smoke


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if self.exc:
            raise self.exc
        return self.response


class FakeOfficialClient:
    def chat(self, **kwargs):
        return "推理过程\n最终答案：2"


def test_real_client_extracts_openai_like_content():
    session = FakeSession(
        FakeResponse({"choices": [{"message": {"content": "最终答案：2"}}]})
    )
    client = RealInternClient(api_key="secret", session=session)

    result = client.chat(messages=[{"role": "user", "content": "1+1=?"}])

    assert result == "最终答案：2"
    assert session.calls


def test_real_client_error_message_redacts_sensitive_markers():
    session = FakeSession(exc=RuntimeError("Authorization Bearer token leaked"))
    client = RealInternClient(api_key="secret", session=session)

    with pytest.raises(RuntimeError) as exc_info:
        client.chat(messages=[])

    message = str(exc_info.value)
    assert "Authorization" not in message
    assert "Bearer" not in message
    assert "token" not in message


def test_run_smoke_with_fake_client_returns_result():
    result = run_smoke(FakeOfficialClient(), problem="1+1=?", idx="unit")

    assert result["idx"] == "unit"
    assert result["final_response"]
    assert isinstance(result["trace"], list)


def test_load_input_json_builds_safe_metadata(tmp_path):
    input_file = tmp_path / "items.json"
    input_file.write_text(
        json.dumps(
            [
                {
                    "problem_id": "omni_000001",
                    "problem": "Find the minimum.",
                    "domain": "optimization",
                    "difficulty": 5.0,
                    "source": "omni",
                    "answer_type": "number",
                    "raw_domain": "Mathematics -> Optimization",
                    "expected_answer": "26",
                    "solution": "hidden solution",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    problem, idx, metadata = _load_input_item(input_file, 0)

    assert problem == "Find the minimum."
    assert idx == "omni_000001"
    assert metadata["idx"] == "omni_000001"
    assert metadata["domain"] == "optimization"
    assert metadata["difficulty"] == 5.0
    assert metadata["source"] == "omni"
    assert metadata["answer_type"] == "number"
    assert metadata["raw_domain"] == "Mathematics -> Optimization"
    assert "expected_answer" not in metadata
    assert "solution" not in metadata


def test_run_smoke_passes_safe_input_metadata(monkeypatch):
    captured = {}

    class RecordingAgent:
        def __init__(self, client):
            self.client = client

        def solve(self, problem, metadata):
            captured["problem"] = problem
            captured["metadata"] = metadata
            return {"final_response": "26", "trace": [{"step": "classify", "content": "domain=optimization"}]}

    monkeypatch.setattr("user_agent.ReasoningAgent", RecordingAgent)

    result = run_user_agent_real_smoke.run_smoke(
        FakeOfficialClient(),
        problem="Find the minimum.",
        idx="omni_000001",
        metadata={
            "idx": "omni_000001",
            "domain": "optimization",
            "difficulty": 5.0,
            "source": "omni",
            "answer_type": "number",
        },
    )

    assert result["final_response"] == "26"
    assert captured["metadata"]["domain"] == "optimization"
    assert captured["metadata"]["difficulty"] == 5.0
    assert captured["metadata"]["source"] == "omni"
    assert captured["metadata"]["answer_type"] == "number"
    assert "expected_answer" not in captured["metadata"]
    assert "solution" not in captured["metadata"]
