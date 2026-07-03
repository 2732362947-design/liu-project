from pathlib import Path

from dev_tools.check_submission_ready import (
    check_response_schema,
    check_user_agent_entrypoint,
    scan_sensitive_patterns,
)


def test_user_agent_entrypoint_check_passes_with_fake_client():
    result = check_user_agent_entrypoint()

    assert not result["failures"]


def test_missing_final_response_fails_schema_check():
    result = check_response_schema({"trace": []}, {"answer": "999"})

    assert any(item["code"] == "missing_final_response" for item in result["failures"])


def test_non_json_trace_fails_schema_check():
    result = check_response_schema({"final_response": "2", "trace": [{"bad": {1, 2}}]}, {})

    assert any(item["code"] == "response_not_json_serializable" for item in result["failures"])


def test_scan_detects_call_intern_s1_in_user_agent(tmp_path):
    path = tmp_path / "user_agent.py"
    path.write_text("def solve():\n    call_intern_s1('x')\n", encoding="utf-8")

    result = scan_sensitive_patterns([path])

    assert any(item["code"] == "call_intern_s1_in_user_agent" for item in result["failures"])


def test_scan_detects_load_dotenv_in_user_agent(tmp_path):
    path = tmp_path / "user_agent.py"
    path.write_text("from dotenv import load_dotenv\nload_dotenv()\n", encoding="utf-8")

    result = scan_sensitive_patterns([path])

    assert any(item["code"] == "load_dotenv_in_user_agent" for item in result["failures"])


def test_scan_detects_home_absolute_path(tmp_path):
    path = tmp_path / "tool.py"
    path.write_text("DATA = '/home/ubuntu/private.json'\n", encoding="utf-8")

    result = scan_sensitive_patterns([path])

    assert any(item["code"] == "absolute_home_path" for item in result["warnings"])


def test_scan_detects_metadata_answer_direct_use(tmp_path):
    path = tmp_path / "user_agent.py"
    path.write_text("final_response = metadata.get('answer')\n", encoding="utf-8")

    result = scan_sensitive_patterns([path])

    assert any(item["code"] == "metadata_answer_used" for item in result["failures"])
