from __future__ import annotations

import user_agent
from dev_tools.run_advanced_real_sanity import _success_summary
from user_agent import (
    FALLBACK_RESPONSE,
    SHORT_ANSWER,
    VERIFICATION_FAILED,
    VERIFICATION_NOT_APPLICABLE,
    VERIFICATION_NOT_EVALUATED,
    WORKED_SOLUTION,
    ReasoningAgent,
    _assess_output_quality,
    _extract_judgeable_solution_with_diagnostics,
    _verify_transition_matrix,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


def _worked_body(ending: str = "Thus the result follows") -> str:
    return (
        "First, apply the definition to obtain the required estimate.\n"
        "Since the hypotheses hold, the intermediate identity is valid.\n"
        f"{ending}"
    )


def _tagged(body: str) -> str:
    return f"<final_solution>{body}</final_solution>"


def _quality(raw: str, mode: str = WORKED_SOLUTION):
    candidate, details = _extract_judgeable_solution_with_diagnostics(raw, response_mode=mode)
    return candidate, _assess_output_quality(raw, candidate, None, mode, details)


def _verification_event(result: dict, retry: bool = False) -> str:
    step = "retry_verify" if retry else "verify"
    return next(event["content"] for event in result["trace"] if event["step"] == step)


def test_quality_failed_does_not_call_generic_verifier(monkeypatch):
    calls = []
    monkeypatch.setattr(user_agent, "verify_solution", lambda *args, **kwargs: calls.append((args, kwargs)))
    agent = ReasoningAgent(FakeClient("unused"))

    _, _, verification, quality = agent._extract_and_verify(
        "Prove the claim.",
        "Thinking Process: Analyze the request.",
        "proof",
        "proof",
        "proof",
        WORKED_SOLUTION,
    )

    assert quality["passed"] is False
    assert calls == []
    assert verification["status"] == VERIFICATION_NOT_EVALUATED


def test_no_clean_candidate_has_not_evaluated_reason():
    agent = ReasoningAgent(FakeClient("unused"))
    _, _, verification, _ = agent._extract_and_verify(
        "Prove the claim.",
        "Thinking Process: Analyze the request.",
        "proof",
        "proof",
        "proof",
        WORKED_SOLUTION,
    )
    assert verification["status"] == "not_evaluated"
    assert verification["reason"] == "no_valid_clean_candidate"
    assert verification["candidate_parse_status"] == "not_evaluated"
    assert verification["verifier_applicable"] is False


def test_quality_passed_calls_generic_verifier_once(monkeypatch):
    calls = []

    def fake_verify(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "passed", "severity": "none", "issues": []}

    monkeypatch.setattr(user_agent, "verify_solution", fake_verify)
    agent = ReasoningAgent(FakeClient("unused"))
    _, _, verification, quality = agent._extract_and_verify(
        "Calculate 24+16.", _tagged("40"), "algebra", "algebra", "number", SHORT_ANSWER
    )
    assert quality["passed"] is True
    assert verification["status"] == "not_applicable"
    assert verification["reason"] == "no_deterministic_short_answer_verifier"
    assert len(calls) == 1
    assert calls[0][0][1] == "40"


def test_retry_polluted_then_clean_verifies_only_second_attempt(monkeypatch):
    calls = []

    def fake_verify(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "passed", "severity": "none", "issues": []}

    monkeypatch.setattr(user_agent, "verify_solution", fake_verify)
    client = FakeClient(["Thinking Process: PRIVATE", _tagged(_worked_body())])
    result = ReasoningAgent(client).solve("Prove the stated claim.", {})

    assert len(client.calls) == 2
    assert len(calls) == 1
    assert calls[0][0][1] == _worked_body()
    assert "reasons=['meta_reasoning_leak']" in next(
        event["content"] for event in result["trace"] if event["step"] == "retry_decision"
    )


def test_two_quality_failures_never_call_verifier(monkeypatch):
    calls = []
    monkeypatch.setattr(user_agent, "verify_solution", lambda *args, **kwargs: calls.append((args, kwargs)))
    client = FakeClient(["Thinking Process: PRIVATE ONE", "Thinking Process: PRIVATE TWO"])
    result = ReasoningAgent(client).solve("Prove the stated claim.", {})
    assert len(client.calls) == 2
    assert calls == []
    assert result["final_response"] == FALLBACK_RESPONSE


def test_worked_matrix_does_not_use_generic_scalar_extractor_or_verifier(monkeypatch):
    monkeypatch.setattr(user_agent, "extract_final_answer", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(user_agent, "verify_solution", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    problem = (
        "A Markov chain has transition matrix P=[[1/2,1/2],[1/4,3/4]]. "
        "Compute P^2 and verify its row sums."
    )
    answer = _tagged(
        r"Multiplication gives $P^2=\begin{pmatrix}3/8&5/8\\5/16&11/16\end{pmatrix}$. "
        "Both rows sum to 1."
    )
    _, extraction, verification, quality = ReasoningAgent(FakeClient("unused"))._extract_and_verify(
        problem, answer, "stochastic_processes", "stochastic_processes", "matrix", WORKED_SOLUTION
    )
    assert quality["passed"] is True
    assert extraction["final_answer"] is None
    assert extraction["answer_type"] == "matrix"
    assert verification["status"] == "passed"


def test_worked_derivation_does_not_use_generic_scalar_extraction(monkeypatch):
    monkeypatch.setattr(user_agent, "extract_final_answer", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    captured = []

    def fake_verify(*args, **kwargs):
        captured.append((args, kwargs))
        return {"status": "passed", "severity": "none", "issues": []}

    monkeypatch.setattr(user_agent, "verify_solution", fake_verify)
    _, extraction, verification, _ = ReasoningAgent(FakeClient("unused"))._extract_and_verify(
        "Derive the estimator.", _tagged(_worked_body(r"Thus \hat p=\bar X")), "statistics", "statistics", "derivation", WORKED_SOLUTION
    )
    assert extraction["final_answer"] is None
    assert captured[0][0][2] is None
    assert verification["status"] == VERIFICATION_NOT_APPLICABLE


def test_worked_proof_does_not_use_generic_scalar_extraction(monkeypatch):
    monkeypatch.setattr(user_agent, "extract_final_answer", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    captured = []

    def fake_verify(*args, **kwargs):
        captured.append((args, kwargs))
        return {"status": "passed", "severity": "none", "issues": []}

    monkeypatch.setattr(user_agent, "verify_solution", fake_verify)
    _, extraction, verification, _ = ReasoningAgent(FakeClient("unused"))._extract_and_verify(
        "Prove the theorem.", _tagged(_worked_body()), "proof", "proof", "proof", WORKED_SOLUTION
    )
    assert extraction["final_answer"] is None
    assert captured[0][0][2] is None
    assert verification["status"] == VERIFICATION_NOT_APPLICABLE


def test_applicable_matrix_verifier_with_unparseable_candidate_is_unknown():
    result = _verify_transition_matrix(
        "A Markov chain has P=[[1/2,1/2],[1/4,3/4]]. Compute P^2.",
        "Matrix multiplication gives the requested transition matrix, with both row sums equal to one.",
    )
    assert result["status"] == "unknown"
    assert result["reason"] == "candidate_parse_ambiguous"
    assert result["subreason"] == "candidate_parse_ambiguous"


def test_matrix_verifier_fails_only_with_located_entry_error():
    result = _verify_transition_matrix(
        "A Markov chain has P=[[1/2,1/2],[1/4,3/4]]. Compute P^2.",
        r"$P^2=\begin{pmatrix}1/2&1/2\\5/16&11/16\end{pmatrix}$.",
    )
    assert result["status"] == VERIFICATION_FAILED
    assert result["reason"] == "explicit_mathematical_error"
    assert result["subreason"] == "matrix_entry_mismatch"
    assert result["checks"]["row"] == 1
    assert result["checks"]["column"] == 1


def test_numeric_derivation_is_not_sent_to_generic_scalar_verifier(monkeypatch):
    monkeypatch.setattr(user_agent, "extract_final_answer", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(user_agent, "verify_solution", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    problem = "Use Newton's method to compute x_1 and x_2, and justify the local convergence order."
    _, extraction, verification, quality = ReasoningAgent(FakeClient("unused"))._extract_and_verify(
        problem,
        _tagged(_worked_body(r"Thus x_1=3/2 and x_2=17/12")),
        "numerical_analysis",
        "numerical_analysis",
        "numeric_derivation",
        WORKED_SOLUTION,
    )
    assert quality["passed"] is True
    assert extraction["final_answer"] is None
    assert verification["status"] == VERIFICATION_NOT_APPLICABLE
    assert verification["reason"] == "no_deterministic_numeric_derivation_verifier"


def test_complete_worked_answer_ending_in_x_i_passes():
    _, quality = _quality(_worked_body("The resulting coordinate is X_i"))
    assert quality["passed"] is True
    assert quality["details"]["ending_reason"] == "ends_with_variable"


def test_complete_worked_answer_ending_in_rank_condition_passes():
    _, quality = _quality(_worked_body(r"Thus \operatorname{rank}(X)=p"))
    assert quality["passed"] is True


def test_complete_nonuniform_convergence_conclusion_passes():
    _, quality = _quality(_worked_body("Thus the convergence is not uniform"))
    assert quality["passed"] is True


def test_trailing_because_is_a_strong_truncation_signal():
    _, quality = _quality(_worked_body("The estimate holds because"))
    assert quality["passed"] is False
    assert quality["details"]["subreason"] == "truncated_sentence"
    assert quality["details"]["ending_reason"] == "trailing_connector"


def test_unclosed_fraction_is_rejected_as_formula_truncation():
    _, quality = _quality(_worked_body(r"Thus x=\frac{"))
    assert quality["passed"] is False
    assert quality["reason"] == "unbalanced_latex"
    assert quality["details"]["ending_reason"] == "unclosed_formula"


def test_opening_only_complete_body_is_recovered():
    raw = "<final_solution>\n" + _worked_body(r"Thus X^TX\hat\beta=X^Ty")
    candidate, quality = _quality(raw)
    assert candidate
    assert quality["passed"] is True
    assert quality["details"]["tag_status"] == "opening_only_recovered"
    assert quality["details"]["compatibility_recovery"] == "opening_only"


def test_closing_only_complete_body_is_recovered():
    raw = _worked_body(r"Thus X^TX\hat\beta=X^Ty") + "\n</final_solution>"
    candidate, quality = _quality(raw)
    assert candidate
    assert quality["passed"] is True
    assert quality["details"]["tag_status"] == "closing_only_recovered"
    assert quality["details"]["compatibility_recovery"] == "closing_only"


def test_opening_only_truncated_formula_is_not_recovered():
    raw = "<final_solution>\n" + _worked_body(r"Thus x=\frac{")
    candidate, quality = _quality(raw)
    assert candidate is None
    assert quality["passed"] is False
    assert quality["details"]["tag_status"] == "opening_only"


def test_meta_response_with_complete_final_solution_section_is_rescued():
    raw = (
        "Thinking Process: Analyze the request and plan the response.\n"
        "Ready for the answer.\nFinal Solution:\n"
        + _worked_body()
    )
    candidate, quality = _quality(raw)
    assert candidate == _worked_body()
    assert quality["passed"] is True
    assert quality["details"]["compatibility_recovery"] == "final_section"


def test_meta_draft_example_is_not_rescued():
    raw = "Thinking Process: Analyze the request.\nDraft:\nFinal Solution:\n" + _worked_body()
    candidate, quality = _quality(raw)
    assert candidate is None
    assert quality["passed"] is False
    assert quality["reason"] == "meta_reasoning_leak"


def test_explicit_mathematical_error_always_has_specific_subreason():
    result = _verify_transition_matrix(
        "A Markov chain has P=[[1/2,1/2],[1/4,3/4]]. Compute P^2.",
        r"$P^2=\begin{pmatrix}1/2&1/2\\5/16&11/16\end{pmatrix}$.",
    )
    assert result["reason"] == "explicit_mathematical_error"
    assert result["subreason"] in {"matrix_entry_mismatch", "row_sum_mismatch"}


def test_runner_clean_candidate_tail_never_contains_polluted_prefix(monkeypatch):
    monkeypatch.setattr(
        user_agent,
        "verify_solution",
        lambda *args, **kwargs: {"status": "passed", "severity": "none", "issues": []},
    )
    raw = "Thinking Process: PRIVATE_INTERNAL_TEXT\nFinal Solution:\n" + _worked_body()
    result = ReasoningAgent(FakeClient(raw)).solve("Prove the claim.", {})
    summary = _success_summary(
        idx="tail",
        declared_domain="proof",
        model="fake",
        result=result,
        elapsed_seconds=0,
    )
    assert summary["clean_candidate_tail"]
    assert len(summary["clean_candidate_tail"]) <= 160
    assert "Thinking Process" not in summary["clean_candidate_tail"]
    assert "PRIVATE_INTERNAL_TEXT" not in summary["clean_candidate_tail"]


def test_short_answer_number_type_mismatch_is_unknown_not_mathematical_failure():
    _, _, verification, quality = ReasoningAgent(FakeClient("unused"))._extract_and_verify(
        "Find the number.",
        _tagged("x=2, x=3"),
        "algebra",
        "algebra",
        "number",
        SHORT_ANSWER,
    )
    assert quality["passed"] is True
    assert verification["status"] == "unknown"
    assert verification["reason"] == "answer_type_mismatch"
