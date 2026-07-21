import json
import os
from pathlib import Path

from agents.classifier_agent import ADVANCED_DOMAINS, classify_problem
from dev_tools import run_advanced_real_sanity
from dev_tools.run_user_agent_real_smoke import _load_input_item, _load_input_items
from dev_tools.run_advanced_real_sanity import (
    DEFAULT_MODEL,
    _classification_from_trace,
    _summarize_trace,
    run_advanced_sanity,
)
from user_agent import FALLBACK_RESPONSE


ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "real_api_sanity_advanced.jsonl"
ANSWER_FIELDS = {
    "answer",
    "expected_answer",
    "gold_answer",
    "reference_answer",
    "solution",
    "gold",
    "reference",
    "ground_truth",
    "expected",
    "expected_solution",
    "official_answer",
    "label",
    "target",
}


def test_advanced_real_sanity_jsonl_has_one_safe_item_per_domain():
    items = _load_input_items(DATA_FILE)

    assert len(items) == len(ADVANCED_DOMAINS)
    assert {item["domain"] for item in items} == set(ADVANCED_DOMAINS)
    for item in items:
        assert item["problem"].strip()
        assert item["expected_key_points"]
        assert item["expected_short_conclusion"].strip()
        assert item["response_mode"] == "worked_solution"
        assert not ANSWER_FIELDS.intersection(item)
        json.dumps(item, ensure_ascii=False)


def test_advanced_real_sanity_items_are_runner_compatible_and_route_correctly():
    for index, expected_domain in enumerate(ADVANCED_DOMAINS):
        problem, idx, metadata = _load_input_item(DATA_FILE, index)
        classification = classify_problem(problem)

        assert idx.startswith("advanced_")
        assert metadata["domain"] == expected_domain
        assert metadata["subject"]
        assert "expected_key_points" not in metadata
        assert "expected_short_conclusion" not in metadata
        assert "response_mode" not in metadata
        assert classification["domain"] == expected_domain
        assert classification["solver_key"] == expected_domain


def test_advanced_real_sanity_summary_extracts_routing_without_api_call():
    trace = [{"step": "classify", "content": "domain=statistics, solver_key=statistics"}]

    assert _classification_from_trace(trace) == ("statistics", "statistics")


def test_trace_summary_extracts_calls_retry_verification_mode_and_tool():
    trace = [
        {"step": "classify", "content": "domain=statistics, solver_key=statistics"},
        {"step": "response_mode", "content": "response_mode=worked_solution"},
        {"step": "local_tool_detect", "content": "tool_name=finite_field, details={}"},
        {
            "step": "model_call",
            "content": "solution_chars=10",
            "thinking_mode_requested": True,
            "thinking_mode_applied": True,
        },
        {"step": "retry_decision", "content": "retry_used=True, reasons=['failed']"},
        {
            "step": "retry_model_call",
            "content": "retry_solution_chars=10",
            "thinking_mode_requested": False,
            "thinking_mode_applied": True,
        },
        {"step": "retry_extract", "content": "answer=2"},
        {"step": "retry_output_quality_check", "content": "status=passed, reason=passed"},
        {"step": "retry_verify", "content": "status=verified"},
    ]

    summary = _summarize_trace(trace)

    assert summary["predicted_domain"] == "statistics"
    assert summary["solver_key"] == "statistics"
    assert summary["response_mode"] == "worked_solution"
    assert summary["api_call_count"] == 2
    assert summary["retry_triggered"] is True
    assert summary["first_thinking_mode_requested"] is True
    assert summary["first_thinking_mode_applied"] is True
    assert summary["retry_thinking_mode_requested"] is False
    assert summary["retry_thinking_mode_applied"] is True
    assert summary["verification_passed"] is True
    assert summary["local_tool_name"] == "finite_field"
    assert len(summary["trace_summary"]) == len(trace)
    json.dumps(summary, ensure_ascii=False)


def test_trace_summary_handles_local_tool_no_retry_failed_verify_and_fallback():
    no_retry = _summarize_trace(
        [
            {"step": "local_tool_solve", "content": "tool_name=number_theory"},
            {"step": "retry_decision", "content": "retry_used=False, reasons=[]"},
            {"step": "verify", "content": "passed=false"},
            {"step": "output_quality_check", "content": "status=passed, reason=passed"},
            {"step": "finalize", "content": "response_mode=short_answer, final_response_chars=1"},
        ]
    )

    assert no_retry["api_call_count"] == 0
    assert no_retry["retry_triggered"] is False
    assert no_retry["verification_passed"] is False
    assert no_retry["response_mode"] == "short_answer"
    assert no_retry["local_tool_name"] == "number_theory"
    assert no_retry["fallback_used"] is False

    fallback = _summarize_trace(
        [{"step": "finalize", "content": "fallback_response"}],
        FALLBACK_RESPONSE,
    )
    assert fallback["fallback_used"] is True

    boolean_fields = _summarize_trace(
        [
            {"step": "retry_decision", "retry_used": True},
            {"step": "retry_verify", "passed": True},
            {"step": "retry_output_quality_check", "passed": True},
            {"step": "finalize", "response_mode": "worked_solution", "fallback_used": False},
        ]
    )
    assert boolean_fields["retry_triggered"] is True
    assert boolean_fields["verification_passed"] is True
    assert boolean_fields["response_mode"] == "worked_solution"
    assert boolean_fields["fallback_used"] is False


def test_trace_summary_preserves_not_applicable_status_and_final_acceptance():
    summary = _summarize_trace(
        [
            {
                "step": "response_mode",
                "content": "response_mode=worked_solution, expected_answer_type=matrix",
            },
            {
                "step": "verify",
                "content": (
                    "status=not_applicable, reason=no_deterministic_matrix_verifier, "
                    "severity=none, expected_answer_type=matrix"
                ),
            },
            {
                "step": "output_quality_check",
                "content": "status=passed, reason=passed, subreason=None",
            },
        ]
    )

    assert summary["response_mode"] == "worked_solution"
    assert summary["expected_answer_type"] == "matrix"
    assert summary["mathematical_verification_status"] == "not_applicable"
    assert summary["mathematical_verification_reason"] == "no_deterministic_matrix_verifier"
    assert summary["mathematical_verification_passed"] is False
    assert summary["output_quality_passed"] is True
    assert summary["output_quality_subreason"] is None
    assert summary["final_acceptance_passed"] is True
    assert summary["verification_passed"] is True


def _write_runner_items(path, count=2):
    rows = [
        {
            "problem_id": f"advanced_domain_{index}",
            "problem": f"Problem {index}",
            "domain": f"domain_{index}",
            "subject": f"Subject {index}",
            "expected_answer": "SECRET ANSWER",
            "solution": "SECRET SOLUTION",
            "review_note": "SECRET NOTE",
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_advanced_runner_saves_full_response_model_and_updates_each_item(tmp_path, monkeypatch, capsys):
    source = tmp_path / "advanced.jsonl"
    output = tmp_path / "results.json"
    _write_runner_items(source)
    long_answer = "完整答案" * 150
    captured_metadata = []
    persist_lengths = []
    original_atomic_write = run_advanced_real_sanity._atomic_write_summaries

    def fake_builder(timeout):
        assert timeout == 7
        assert os.environ["INTERN_S1_MODEL"] == DEFAULT_MODEL
        return object()

    def fake_run_smoke(client, problem, idx, metadata):
        captured_metadata.append(metadata)
        return {
            "final_response": long_answer,
            "trace": [
                {"step": "classify", "content": "domain=domain_0, solver_key=domain_0"},
                {"step": "response_mode", "content": "response_mode=short_answer"},
                {"step": "model_call", "content": "solution_chars=600"},
                {"step": "verify", "content": "status=passed"},
                {"step": "output_quality_check", "content": "status=passed, reason=passed"},
            ],
        }

    def recording_write(destination, summaries):
        persist_lengths.append(len(summaries))
        original_atomic_write(destination, summaries)

    monkeypatch.setattr(run_advanced_real_sanity, "_build_client_from_env", fake_builder)
    monkeypatch.setattr(run_advanced_real_sanity, "run_smoke", fake_run_smoke)
    monkeypatch.setattr(run_advanced_real_sanity, "_atomic_write_summaries", recording_write)

    summaries = run_advanced_sanity(
        input_path=source,
        output_path=output,
        timeout=7,
        sleep_seconds=0,
        model=DEFAULT_MODEL,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    terminal = capsys.readouterr().out

    assert persist_lengths == [1, 2]
    assert saved == summaries
    assert all(row["final_response"] == long_answer for row in saved)
    assert all(row["final_response_nonempty"] is True for row in saved)
    assert all(row["model"] == DEFAULT_MODEL for row in saved)
    assert all(row["elapsed_seconds"] >= 0 for row in saved)
    assert all(row["api_call_count"] == 1 for row in saved)
    assert all(row["verification_passed"] is True for row in saved)
    assert captured_metadata == [{"idx": "advanced_domain_0"}, {"idx": "advanced_domain_1"}]
    assert "requested_model=intern-s2-preview" in terminal
    assert long_answer not in terminal
    assert "SECRET ANSWER" not in json.dumps(saved, ensure_ascii=False)
    json.dumps(saved, ensure_ascii=False)


def test_advanced_runner_records_redacted_error_and_continues(tmp_path, monkeypatch):
    source = tmp_path / "advanced.jsonl"
    output = tmp_path / "results.json"
    _write_runner_items(source)
    calls = []

    monkeypatch.setattr(run_advanced_real_sanity, "_build_client_from_env", lambda timeout: object())

    def fake_run_smoke(client, problem, idx, metadata):
        calls.append(idx)
        if len(calls) == 1:
            raise TimeoutError(
                "Authorization: Bearer TOP_SECRET token=TOKEN_SECRET "
                "https://example.test/path?api_key=QUERY_SECRET"
            )
        return {"final_response": "第二题答案", "trace": []}

    monkeypatch.setattr(run_advanced_real_sanity, "run_smoke", fake_run_smoke)

    summaries = run_advanced_sanity(
        input_path=source,
        output_path=output,
        timeout=1,
        sleep_seconds=0,
        model=DEFAULT_MODEL,
    )
    serialized = json.dumps(summaries, ensure_ascii=False)

    assert calls == ["advanced_domain_0", "advanced_domain_1"]
    assert [row["status"] for row in summaries] == ["error", "success"]
    assert summaries[0]["error"]["type"] == "TimeoutError"
    assert summaries[0]["final_response_nonempty"] is False
    assert summaries[1]["final_response"] == "第二题答案"
    assert "TOP_SECRET" not in serialized
    assert "TOKEN_SECRET" not in serialized
    assert "QUERY_SECRET" not in serialized
    assert json.loads(output.read_text(encoding="utf-8")) == summaries
