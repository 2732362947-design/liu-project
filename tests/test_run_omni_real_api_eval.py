import json
from types import SimpleNamespace

from dev_tools import run_omni_real_api_eval
from dev_tools.run_omni_real_api_eval import DEFAULT_MODEL, build_client, build_solve_metadata, evaluate_item, run_evaluation


class FakeClient:
    pass


class RecordingAgent:
    calls = []

    def __init__(self, client):
        assert isinstance(client, FakeClient)

    def solve(self, problem, metadata):
        self.calls.append((problem, metadata))
        if "explode" in problem:
            raise RuntimeError("fake failure")
        return {
            "final_response": "2",
            "trace": [
                {"step": "classify", "content": "domain=algebra, solver_key=algebra"},
                {"step": "response_mode", "content": "response_mode=short_answer"},
                {"step": "model_call", "content": "summary only"},
                {"step": "verify", "content": "status=passed, severity=none"},
                {"step": "output_quality_check", "content": "status=passed, reason=passed"},
            ],
        }


def _item(idx, problem="1+1=?"):
    return {
        "idx": idx,
        "source_idx": idx,
        "problem": problem,
        "expected_answer": "2",
        "expected_domain": "algebra",
        "solution": "SECRET SOLUTION",
        "review_note": "SECRET NOTE",
        "label_status": "unreviewed",
        "response_mode": "short_answer",
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_metadata_isolation_excludes_all_evaluation_labels():
    item = _item("one")
    metadata = build_solve_metadata(item)

    assert metadata == {"idx": "one"}
    assert not {"expected_answer", "solution", "review_note", "label_status"} & metadata.keys()


def test_subject_hint_is_only_added_explicitly():
    assert build_solve_metadata(_item("one"), True) == {"idx": "one"}


def test_evaluate_item_extracts_trace_summary_without_trace_output():
    RecordingAgent.calls = []
    result = evaluate_item(_item("one"), FakeClient(), agent_factory=RecordingAgent)

    assert RecordingAgent.calls == [("1+1=?", {"idx": "one"})]
    assert result["predicted_domain"] == "algebra"
    assert result["solver_key"] == "algebra"
    assert result["api_call_count"] == 1
    assert result["verification_passed"] is True
    assert result["final_response_nonempty"] is True
    assert isinstance(result["trace_summary"], list)
    assert result["first_attempt_diagnostics"]["quality_reason"] == "passed"
    assert result["model"] == DEFAULT_MODEL
    assert "trace" not in result
    json.dumps(result, ensure_ascii=False)


def test_runner_saves_thinking_flags_and_only_whitelisted_trace_fields():
    class DiagnosticAgent:
        def __init__(self, client):
            pass

        def solve(self, problem, metadata):
            return {
                "final_response": "2",
                "trace": [
                    {
                        "step": "model_call",
                        "content": "status=success, solution_chars=500, Thinking Process: PRIVATE_RAW",
                        "thinking_mode_requested": True,
                        "thinking_mode_applied": True,
                    },
                    {
                        "step": "retry_model_call",
                        "content": "status=success, retry_solution_chars=200, system prompt=PRIVATE",
                        "thinking_mode_requested": False,
                        "thinking_mode_applied": True,
                    },
                ],
            }

    result = evaluate_item(_item("one"), FakeClient(), agent_factory=DiagnosticAgent)
    serialized = json.dumps(result["trace_summary"], ensure_ascii=False)

    assert result["first_thinking_mode_requested"] is True
    assert result["first_thinking_mode_applied"] is True
    assert result["retry_thinking_mode_requested"] is False
    assert result["retry_thinking_mode_applied"] is True
    assert "PRIVATE_RAW" not in serialized
    assert "system prompt" not in serialized


def test_rejected_candidate_diagnostics_are_bounded_and_meta_cleaned():
    class RejectedAgent:
        def __init__(self, client):
            pass

        def solve(self, problem, metadata):
            return {
                "final_response": "未能得到可靠答案",
                "trace": [
                    {"step": "model_call", "content": "status=success, solution_chars=900"},
                    {
                        "step": "extract",
                        "content": "status=passed, extracted_answer_type=number, final_answer_chars=4",
                    },
                    {
                        "step": "output_quality_check",
                        "content": (
                            "status=passed, reason=passed, subreason=None, raw_chars=900, "
                            "clean_candidate_chars=700, tag_status=complete, "
                            "latex_balance_status=passed, ending_status=complete"
                        ),
                    },
                    {
                        "step": "verify",
                        "content": "status=failed, reason=short_answer_verification_failed, subreason=None, severity=high",
                    },
                ],
                "first_attempt_diagnostics": {
                    "raw_chars": 900,
                    "tag_status": "complete",
                    "clean_candidate_chars": 700,
                    "quality_reason": "passed",
                    "quality_subreason": None,
                    "latex_balance_status": "passed",
                    "ending_status": "complete",
                    "extracted_answer_type": "number",
                    "extracted_answer_summary": "2000",
                    "candidate_tail": "Thinking Process: PRIVATE model draft",
                },
            }

    result = evaluate_item(_item("one"), FakeClient(), agent_factory=RejectedAgent)

    assert result["rejected_candidate_chars"] == 700
    assert result["rejected_candidate_tail"] is None
    assert result["extracted_short_answer"] == "2000"
    assert result["short_answer_verification_reason"] == "short_answer_verification_failed"


def test_omni_trace_summary_preserves_unknown_math_as_accepted_not_verified():
    summary = run_omni_real_api_eval._trace_parts(
        [
            {
                "step": "response_mode",
                "content": "response_mode=worked_solution, expected_answer_type=unknown",
            },
            {
                "step": "verify",
                "content": (
                    "status=unknown, reason=insufficient_deterministic_verification, "
                    "severity=low, expected_answer_type=unknown"
                ),
            },
            {"step": "output_quality_check", "content": "status=passed, reason=passed, subreason=None"},
        ]
    )

    assert summary["mathematical_verification_status"] == "unknown"
    assert summary["mathematical_verification_passed"] is False
    assert summary["final_acceptance_passed"] is True
    assert summary["verification_passed"] is True


def test_build_client_overrides_environment_model_without_network(monkeypatch):
    captured = {}

    class FakeRealInternClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("INTERN_S1_API_KEY", "fake-key")
    monkeypatch.setenv("INTERN_S1_MODEL", "stale-model")
    monkeypatch.setattr(run_omni_real_api_eval, "_load_env_file", lambda: None)
    monkeypatch.setattr(run_omni_real_api_eval, "RealInternClient", FakeRealInternClient)

    client = build_client("intern-s2-preview", 9)

    assert isinstance(client, FakeRealInternClient)
    assert captured["model"] == "intern-s2-preview"
    assert captured["timeout"] == 9
    assert run_omni_real_api_eval.os.environ["INTERN_S1_MODEL"] == "intern-s2-preview"


def test_main_prints_and_threads_requested_model(monkeypatch, capsys):
    captured = {}
    args = SimpleNamespace(
        input="input.jsonl",
        output="output.jsonl",
        model="intern-s2-preview",
        concurrency=1,
        limit=None,
        start_index=0,
        domain=None,
        resume=True,
        use_subject_hint=False,
        timeout=12,
    )
    monkeypatch.setattr(run_omni_real_api_eval, "_parse_args", lambda: args)
    monkeypatch.setattr(
        run_omni_real_api_eval,
        "build_client",
        lambda model, timeout: captured.update(builder=(model, timeout)) or FakeClient(),
    )
    monkeypatch.setattr(
        run_omni_real_api_eval,
        "run_evaluation",
        lambda *positional, **keywords: captured.update(run=(positional, keywords)) or [],
    )

    run_omni_real_api_eval.main()

    assert captured["builder"] == ("intern-s2-preview", 12)
    assert captured["run"][1]["model"] == "intern-s2-preview"
    assert "requested_model=intern-s2-preview" in capsys.readouterr().out


def test_model_call_error_in_agent_trace_is_a_runtime_error():
    class TraceErrorAgent:
        def __init__(self, client):
            pass

        def solve(self, problem, metadata):
            return {
                "final_response": "Unable to produce a verified answer.",
                "trace": [
                    {"step": "model_call", "content": "error: RuntimeError"},
                    {"step": "finalize", "content": "fallback_response"},
                ],
            }

    result = evaluate_item(_item("one"), FakeClient(), agent_factory=TraceErrorAgent)

    assert result["status"] == "runtime_error"
    assert result["api_call_count"] == 1
    assert result["error"] == "ReasoningAgent trace reported a model_call error"
    assert "model_call_error" not in result


def test_single_item_exception_does_not_stop_incremental_jsonl(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "results.jsonl"
    _write(source, [_item("one"), _item("bad", "explode"), _item("three")])

    written = run_evaluation(source, output, FakeClient(), resume=False, agent_factory=RecordingAgent)
    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert [row["status"] for row in written] == ["success", "runtime_error", "success"]
    assert saved == written
    assert len(saved) == 3
    assert all(row["model"] == DEFAULT_MODEL for row in saved)


def test_jsonl_is_flushed_before_the_next_item_runs(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "results.jsonl"
    _write(source, [_item("one"), _item("two")])

    class ObservingAgent:
        calls = 0

        def __init__(self, client):
            pass

        def solve(self, problem, metadata):
            if self.calls:
                assert len(output.read_text(encoding="utf-8").splitlines()) == 1
            type(self).calls += 1
            return {"final_response": "2", "trace": []}

    run_evaluation(source, output, FakeClient(), resume=False, agent_factory=ObservingAgent)

    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_resume_skips_success_but_retries_failed_idx(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "results.jsonl"
    _write(source, [_item("done"), _item("retry")])
    _write(
        output,
        [
            {"idx": "done", "status": "success"},
            {"idx": "retry", "status": "runtime_error"},
        ],
    )

    written = run_evaluation(source, output, FakeClient(), resume=True, agent_factory=RecordingAgent)

    assert [row["idx"] for row in written] == ["retry"]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3


def test_limit_start_index_and_domain_filters(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "results.jsonl"
    rows = [_item("zero"), _item("one"), _item("two")]
    rows[1]["expected_domain"] = "geometry"
    rows[2]["expected_domain"] = "geometry"
    _write(source, rows)

    written = run_evaluation(
        source,
        output,
        FakeClient(),
        resume=False,
        start_index=1,
        limit=1,
        domain="geometry",
        agent_factory=RecordingAgent,
    )

    assert [row["idx"] for row in written] == ["one"]


def test_repeatable_idx_filter_selects_only_requested_items(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "results.jsonl"
    _write(source, [_item("zero"), _item("one"), _item("two")])

    written = run_evaluation(
        source,
        output,
        FakeClient(),
        resume=False,
        idxs=["zero", "two"],
        agent_factory=RecordingAgent,
    )

    assert [row["idx"] for row in written] == ["zero", "two"]
