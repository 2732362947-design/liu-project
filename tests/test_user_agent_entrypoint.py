from types import SimpleNamespace

from user_agent import ReasoningAgent


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
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
