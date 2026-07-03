import json
from types import SimpleNamespace

import user_agent
from user_agent import ReasoningAgent, _build_correction_prompt, _safe_metadata


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, list):
            index = min(len(self.calls) - 1, len(self.response) - 1)
            return self.response[index]
        return self.response


class ErrorClient:
    def chat(self, **kwargs):
        raise RuntimeError("network token should not leak")


def _steps(result):
    return [item["step"] for item in result["trace"]]


def test_can_import_and_initialize_reasoning_agent():
    agent = ReasoningAgent(client=FakeClient("最终答案：2"))

    assert agent.client is not None


def test_solve_calls_chat_and_returns_dict_with_trace():
    client = FakeClient("解：1+1=2\n最终答案：2")
    agent = ReasoningAgent(client=client)

    result = agent.solve("1+1=?", {})

    assert client.calls
    assert isinstance(result, dict)
    assert isinstance(result["final_response"], str)
    assert result["final_response"]
    assert isinstance(result["trace"], list)


def test_trace_contains_required_steps():
    result = ReasoningAgent(FakeClient("解得 x=6。\n最终答案：x=6")).solve("解方程 2x+5=17", {})

    steps = _steps(result)

    for step in ("classify", "plan", "model_call", "extract", "verify", "finalize"):
        assert step in steps


def test_string_response_is_supported():
    result = ReasoningAgent(FakeClient("最终答案：2")).solve("1+1=?", {})

    assert "2" in result["final_response"]


def test_dict_response_is_supported():
    result = ReasoningAgent(FakeClient({"content": "最终答案：3"})).solve("1+2=?", {})

    assert "3" in result["final_response"]


def test_object_content_response_is_supported():
    result = ReasoningAgent(FakeClient(SimpleNamespace(content="最终答案：4"))).solve("2+2=?", {})

    assert "4" in result["final_response"]


def test_openai_like_response_is_supported():
    response = {"choices": [{"message": {"content": "最终答案：5"}}]}
    result = ReasoningAgent(FakeClient(response)).solve("2+3=?", {})

    assert "5" in result["final_response"]


def test_metadata_answer_is_not_used_as_final_response():
    client = FakeClient("解：1+1=2\n最终答案：2")
    result = ReasoningAgent(client).solve("1+1=?", {"answer": "999"})

    assert "2" in result["final_response"]
    assert result["final_response"] != "999"


def test_model_call_exception_returns_nonempty_response_and_error_trace():
    result = ReasoningAgent(ErrorClient()).solve("1+1=?", {})

    assert isinstance(result, dict)
    assert result["final_response"]
    assert any(item["step"] == "model_call" and "error" in item["content"] for item in result["trace"])


def test_no_retry_when_first_verification_passes():
    client = FakeClient("解：1+1=2\n最终答案：2")
    result = ReasoningAgent(client).solve("1+1=?", {})

    assert len(client.calls) == 1
    assert result["final_response"]
    assert any(item["step"] == "retry_decision" and "retry_used=False" in item["content"] for item in result["trace"])


def test_empty_final_answer_triggers_retry():
    client = FakeClient(["这是一段没有明确答案的说明。", "修正后：1+1=2\n最终答案：2"])
    result = ReasoningAgent(client).solve("1+1=?", {})

    assert len(client.calls) == 2
    assert "2" in result["final_response"]
    assert "retry_model_call" in _steps(result)


def test_verifier_failed_triggers_retry(monkeypatch):
    calls = {"count": 0}

    def fake_verify(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "failed", "severity": "high", "issues": [{"code": "bad"}]}
        return {"status": "passed", "severity": "none", "issues": []}

    monkeypatch.setattr(user_agent, "verify_solution", fake_verify)
    client = FakeClient(["最终答案：1", "最终答案：2"])

    result = ReasoningAgent(client).solve("1+1=?", {})

    assert len(client.calls) == 2
    assert "2" in result["final_response"]


def test_second_attempt_without_answer_falls_back_to_nonempty_response():
    client = FakeClient(["没有明确答案。", "仍然没有明确答案。"])

    result = ReasoningAgent(client).solve("1+1=?", {})

    assert len(client.calls) == 2
    assert result["final_response"]


def test_correction_prompt_filters_metadata_answer():
    prompt = _build_correction_prompt(
        "1+1=?",
        {"idx": 0, "answer": "999", "expected_answer": "888", "subject": "math"},
        "第一次解答",
        "",
        {"status": "failed", "issues": [], "suggestion": "retry"},
        solver_key="algebra",
    )

    assert "999" not in prompt
    assert "888" not in prompt
    assert "subject" in prompt


def test_metadata_answer_not_leaked_to_retry_prompt():
    client = FakeClient(["没有明确答案。", "最终答案：2"])
    result = ReasoningAgent(client).solve("1+1=?", {"idx": 0, "answer": "999"})

    assert len(client.calls) == 2
    retry_prompt = client.calls[1]["messages"][0]["content"]
    assert "999" not in retry_prompt
    assert result["final_response"] != "999"


def test_trace_is_json_serializable_with_retry():
    result = ReasoningAgent(FakeClient(["没有明确答案。", "最终答案：2"])).solve("1+1=?", {})

    json.dumps(result, ensure_ascii=False)


def test_retry_model_call_exception_is_captured():
    class RetryErrorClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return "没有明确答案。"
            raise RuntimeError("Authorization token should not leak")

    client = RetryErrorClient()
    result = ReasoningAgent(client).solve("1+1=?", {})

    assert len(client.calls) == 2
    assert result["final_response"]
    assert any(item["step"] == "retry_model_call" and "error" in item["content"] for item in result["trace"])


def test_safe_metadata_filters_answer_fields():
    safe = _safe_metadata({"answer": "999", "gold_answer": "888", "idx": 1})

    assert safe == {"idx": 1}
