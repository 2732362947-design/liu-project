from __future__ import annotations

import json
from pathlib import Path

import user_agent
from agents.classifier_agent import classify_problem
from dev_tools.run_omni_real_api_eval import evaluate_item
from dev_tools.score_omni_results import score_item
from user_agent import (
    FALLBACK_RESPONSE,
    SHORT_ANSWER,
    VERIFICATION_FAILED,
    VERIFICATION_NOT_APPLICABLE,
    VERIFICATION_PASSED,
    VERIFICATION_UNKNOWN,
    WORKED_SOLUTION,
    ReasoningAgent,
    _build_correction_prompt,
    _expected_answer_type_from_problem,
    _normalize_verification,
    _proof_risk_signals,
    _verify_factorial_floor_exact,
)


DATASET = Path(__file__).resolve().parents[1] / "evaluation/datasets/omni_math_smoke_30.jsonl"


def _dataset_item(idx: str) -> dict:
    return next(
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["idx"] == idx
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


class LegacyClient:
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls = []

    def chat(self, messages, temperature, max_tokens):
        self.calls.append(
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        )
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


def _tagged(body: str) -> str:
    return f"<final_solution>{body}</final_solution>"


def _proof_with_risk() -> str:
    return _tagged(
        "First derive the estimate from the hypotheses.\n"
        r"\[a \leq b.\]" "\nIt follows that\n" r"\[c \geq d.\]" "\n"
        "Therefore the stated inequality follows from these estimates."
    )


def test_how_many_polynomial_factors_infers_number():
    assert _expected_answer_type_from_problem(_dataset_item("omni_eval_000934")["problem"]) == "number"


def test_find_floor_factorial_value_infers_number():
    assert _expected_answer_type_from_problem(_dataset_item("omni_eval_000817")["problem"]) == "number"


def test_explicit_factor_polynomial_infers_expression():
    assert _expected_answer_type_from_problem("Factor the polynomial x^4-1.") == "expression"


def test_closed_formula_infers_expression():
    assert _expected_answer_type_from_problem("Find a closed formula for a_n.") == "expression"


def test_metadata_answer_type_does_not_control_formal_route():
    result = ReasoningAgent(FakeClient(_tagged(r"\boxed{4}"))).solve(
        "Compute the value of 2+2.", {"answer_type": "expression"}
    )
    mode = next(event["content"] for event in result["trace"] if event["step"] == "response_mode")
    assert "expected_answer_type=number" in mode


def test_generic_short_answer_has_no_deterministic_verifier():
    result = ReasoningAgent(FakeClient(_tagged(r"2+2=4.\ \boxed{4}"))).solve(
        "Compute the value of 2+2.", {}
    )
    verify = next(event["content"] for event in result["trace"] if event["step"] == "verify")
    assert "status=not_applicable" in verify
    assert "reason=no_deterministic_short_answer_verifier" in verify
    assert result["final_response"] == "4"


def test_applicable_verifier_problem_parse_failure_is_unknown():
    result = _verify_factorial_floor_exact(
        r"Find the value of \(\left\lfloor\frac{10!}{9!+broken}\right\rfloor\).",
        "9",
    )
    assert result["status"] == VERIFICATION_UNKNOWN
    assert result["reason"] == "problem_parse_ambiguous"


def test_answer_type_mismatch_is_unknown_not_failed():
    result = _normalize_verification(
        {
            "status": "failed",
            "severity": "high",
            "issues": [{"code": "answer_type_mismatch"}],
        },
        response_mode=SHORT_ANSWER,
        expected_answer_type="number",
    )
    assert result["status"] == VERIFICATION_UNKNOWN
    assert result["reason"] == "answer_type_mismatch"


def test_factorial_floor_2000_passes_with_diagnostics():
    result = _verify_factorial_floor_exact(_dataset_item("omni_eval_000817")["problem"], "2000")
    assert result["status"] == VERIFICATION_PASSED
    assert result["verifier_name"] == "factorial_floor_exact"
    assert result["computed_value_summary"] == "integer:2000"
    assert result["candidate_value_summary"] == "integer:2000"


def test_factorial_floor_2001_fails_with_located_mismatch():
    result = _verify_factorial_floor_exact(_dataset_item("omni_eval_000817")["problem"], "2001")
    assert result["status"] == VERIFICATION_FAILED
    assert result["subreason"] == "floor_value_mismatch"
    assert result["problem_parse_status"] == VERIFICATION_PASSED
    assert result["candidate_parse_status"] == VERIFICATION_PASSED


def test_polynomial_factor_count_five_is_accepted_without_retry():
    item = _dataset_item("omni_eval_000934")
    client = FakeClient(_tagged(r"The factor multiplicities total 2+1+1+1=5. Hence \boxed{5}."))
    result = ReasoningAgent(client).solve(item["problem"], {})
    verify = next(event["content"] for event in result["trace"] if event["step"] == "verify")
    assert len(client.calls) == 1
    assert result["final_response"] == "5"
    assert result["final_response"] != FALLBACK_RESPONSE
    assert "status=not_applicable" in verify
    assert "reason=no_deterministic_polynomial_factorization_verifier" in verify
    assert result["first_attempt_diagnostics"]["verifier_applicable"] is False


def test_factorial_floor_wrong_candidate_retries_then_uses_exact_override():
    item = _dataset_item("omni_eval_000817")
    client = FakeClient([_tagged(r"Thus \boxed{2001}."), _tagged(r"Again \boxed{2001}.")])
    result = ReasoningAgent(client).solve(item["problem"], {})
    assert len(client.calls) == 2
    assert result["final_response"] == "2000"
    assert result["deterministic_answer_override"] is True
    assert result["retry_attempt_diagnostics"]["verification_subreason"] == "floor_value_mismatch"


def test_short_answer_first_and_retry_disable_thinking_and_use_zero_temperature():
    client = FakeClient(["Thinking Process: PRIVATE", _tagged(r"\boxed{4}")])
    ReasoningAgent(client).solve("Compute the value of 2+2.", {})
    assert [call["thinking_mode"] for call in client.calls] == [False, False]
    assert [call["temperature"] for call in client.calls] == [0, 0]


def test_worked_solution_first_enables_thinking_and_retry_disables_it():
    client = FakeClient(["Thinking Process: PRIVATE", _proof_with_risk()])
    ReasoningAgent(client).solve("Prove the stated inequality.", {})
    assert [call["thinking_mode"] for call in client.calls] == [True, False]


def test_legacy_client_keeps_two_attempt_limit_without_thinking_kwarg():
    client = LegacyClient(["Thinking Process: PRIVATE", _tagged(r"\boxed{4}")])
    result = ReasoningAgent(client).solve("Compute the value of 2+2.", {})
    assert len(client.calls) == 2
    assert all("thinking_mode" not in call for call in client.calls)
    assert result["final_response"] == "4"


def test_short_answer_prompts_are_concise_bounded_and_boxed():
    client = FakeClient(_tagged(r"\boxed{4}"))
    ReasoningAgent(client).solve("Compute the value of 2+2.", {})
    system = client.calls[0]["messages"][0]["content"]
    user = client.calls[0]["messages"][1]["content"]
    joined = f"{system}\n{user}".lower()
    assert "concise exact derivation" in joined
    assert "at most 8 substantive steps" in joined
    assert "boxed final answer" in joined


def test_retry_prompt_omits_old_answer_expected_answer_and_metadata():
    prompt = _build_correction_prompt(
        "Compute the value of 2+2.",
        {"expected_answer": "4", "solution": "SECRET", "answer_type": "number"},
        "UNIQUE_OLD_OUTPUT 9001",
        "9001",
        {"status": "failed", "computed_value_summary": "integer:4"},
        response_mode=SHORT_ANSWER,
    )
    assert "Recompute independently." in prompt
    assert "Do not repeat the conclusion." in prompt
    assert "UNIQUE_OLD_OUTPUT" not in prompt
    assert "9001" not in prompt
    assert "SECRET" not in prompt
    assert "expected_answer" not in prompt


def test_factorial_floor_routes_to_compatible_number_theory_solver():
    route = classify_problem(_dataset_item("omni_eval_000817")["problem"])
    assert route["domain"] in {"number_theory", "algebra"}
    assert route["solver_key"] in {"number_theory", "algebra"}
    assert "factorial_floor_value" in route["matched_signal_categories"]


def test_proof_risk_is_reported_without_changing_pipeline_acceptance():
    client = FakeClient(_proof_with_risk())
    result = ReasoningAgent(client).solve("Prove the stated inequality.", {})
    assert result["final_response"] != FALLBACK_RESPONSE
    assert result["proof_review_required"] is True
    assert "unjustified_inequality_direction_change" in result["proof_risk_signals"]


def test_runner_and_score_separate_pipeline_acceptance_from_correctness():
    item = {
        "idx": "proof-risk",
        "source_idx": "proof-risk",
        "problem": "Prove the stated inequality.",
        "expected_answer": "proof",
        "expected_domain": "proof",
        "answer_type": "proof",
        "response_mode": "worked_solution",
        "label_status": "unreviewed",
    }
    run = evaluate_item(
        item,
        FakeClient("unused"),
        agent_factory=lambda client: ReasoningAgent(FakeClient(_proof_with_risk())),
    )
    scored = score_item(item, run)
    assert run["pipeline_acceptance"] is True
    assert run["proof_review_required"] is True
    assert scored["pipeline_acceptance"] is True
    assert scored["mathematical_correctness"] == "manual_review"
    assert scored["score_status"] == "manual_review"
