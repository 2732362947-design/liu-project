import pytest

from dev_tools.run_user_agent_real_smoke import RealInternClient, run_smoke


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
