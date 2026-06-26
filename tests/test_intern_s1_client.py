import requests

import intern_s1_client


def test_timeout_returns_error(monkeypatch):
    monkeypatch.setattr(intern_s1_client, "INTERN_S1_API_KEY", "test-key")
    monkeypatch.setattr(intern_s1_client, "INTERN_S1_MAX_RETRIES", 0)
    monkeypatch.setattr(intern_s1_client, "INTERN_S1_READ_TIMEOUT_SECONDS", 1)

    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout("read timed out")

    class FakeSession:
        trust_env = True

        def post(self, *args, **kwargs):
            return fake_post(*args, **kwargs)

    monkeypatch.setattr(requests, "Session", FakeSession)

    result = intern_s1_client.call_intern_s1("hello")

    assert result.startswith("[intern-s1 error] timeout")
    assert "test-key" not in result


def test_dns_error_classification():
    text = "[intern-s1 error] connection_error: Name or service not known"

    assert intern_s1_client.classify_intern_s1_error(text) == "dns_error"


def test_connection_error_classification_without_dns_keyword():
    text = (
        "[intern-s1 error] connection_error: ConnectionError: "
        "Failed to establish a new connection"
    )

    assert intern_s1_client.classify_intern_s1_error(text) == "connection_error"


def test_timeout_classification():
    text = "[intern-s1 error] timeout: Read timed out"

    assert intern_s1_client.classify_intern_s1_error(text) == "timeout"
