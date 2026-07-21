import json

import pytest

import user_agent
from agents.solver_agent import VALID_SOLVER_KEYS, build_solver_prompt, load_solver_template
from dev_tools.run_advanced_real_sanity import _summarize_trace
from dev_tools.run_user_agent_real_smoke import RealInternClient, extract_content
from user_agent import (
    FALLBACK_RESPONSE,
    PRIMARY_SYSTEM_MESSAGE,
    RETRY_SYSTEM_MESSAGE,
    SHORT_ANSWER_SYSTEM_MESSAGE,
    SHORT_ANSWER,
    VERIFICATION_FAILED,
    VERIFICATION_NOT_APPLICABLE,
    VERIFICATION_UNKNOWN,
    WORKED_SOLUTION,
    ReasoningAgent,
    _assess_output_quality,
    _build_correction_prompt,
    _compose_final_response,
    _determine_response_mode,
    _expected_answer_type_from_problem,
    _extract_judgeable_solution,
    _extract_judgeable_solution_with_diagnostics,
    _has_meta_reasoning_leak,
    _normalize_verification,
    _validate_short_solution_quality,
    _validate_worked_solution_quality,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


def test_all_formal_solver_prompts_are_neutral_without_repeated_system_protocol():
    for solver_key in VALID_SOLVER_KEYS:
        template = load_solver_template(solver_key)
        assert "Intern-S1" not in template
        assert "Intern-S2" not in template
        prompt = user_agent._append_prompt_constraints(
            build_solver_prompt("Compute the requested quantity.", "algebra", ["Solve it."], solver_key=solver_key),
            "Compute the requested quantity.",
            solver_key,
            "number",
            SHORT_ANSWER,
        )
        assert "<final_solution>" not in prompt
        assert "Thinking Process" not in prompt
        assert "private reasoning" not in prompt


def test_correction_prompt_is_neutral_and_does_not_reinject_polluted_response():
    polluted = "Thinking Process: Analyze the Request. TOP_SECRET_DRAFT"
    prompt = _build_correction_prompt(
        "Prove the claim.",
        {"idx": "one", "expected_answer": "blocked"},
        polluted,
        "Wait,",
        {"status": "passed", "severity": "none", "issues": []},
        solver_key="proof",
        domain="proof",
        response_mode=WORKED_SOLUTION,
        output_quality_reason="meta_reasoning_leak",
    )

    assert "TOP_SECRET_DRAFT" not in prompt
    assert "Intern-S1" not in prompt
    assert "Intern-S2" not in prompt
    assert "<final_solution>" not in prompt
    assert "blocked" not in prompt
    assert prompt.count("Problem:") == 1
    assert prompt == "Problem:\nProve the claim.\n\nProduce the final solution now."
    assert "meta_reasoning_leak" not in prompt


def test_extracts_last_complete_final_solution_block():
    raw = (
        "<final_solution>first draft</final_solution>\n"
        "<final_solution>Second and judgeable.</final_solution>"
    )
    assert _extract_judgeable_solution(raw, response_mode=WORKED_SOLUTION) == "Second and judgeable."


@pytest.mark.parametrize(
    "raw",
    [
        "<final_solution>unfinished",
        "orphan</final_solution>",
        "<final_solution>outer <final_solution>inner</final_solution></final_solution>",
        "<final_solution>   </final_solution>",
    ],
)
def test_rejects_incomplete_empty_or_nested_final_solution_tags(raw):
    assert _extract_judgeable_solution(raw, response_mode=WORKED_SOLUTION) is None


def test_thinking_outside_tag_is_discarded_but_thinking_only_is_rejected():
    raw = "Thinking Process: Analyze the Request.\n<final_solution>40</final_solution>"
    assert _extract_judgeable_solution(raw, response_mode=SHORT_ANSWER) == "40"
    assert _extract_judgeable_solution("Thinking Process: Analyze the Request.", response_mode=SHORT_ANSWER) is None


def test_meta_reasoning_detection_does_not_reject_normal_proof():
    proof = "Let n=2k. Then n+n=4k=2(2k). Therefore n+n is even, and the proof is complete."
    assert _has_meta_reasoning_leak(proof) is False
    assert _validate_worked_solution_quality(proof, "n+n is even") == (True, "passed")


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Thinking Process: Analyze the Request before answering.", "meta_reasoning_leak"),
        ("First inspect the system instruction, then answer the problem.", "meta_reasoning_leak"),
        ("Thus not uniform", "worked_solution_too_short"),
        ("命题成立", "worked_solution_too_short"),
        ("First compute $x=1$. Therefore,", "incomplete_output"),
        (r"First compute $x=\frac{1}{2$ and then continue.", "unbalanced_latex"),
    ],
)
def test_worked_solution_quality_rejects_leaks_fragments_and_truncation(text, reason):
    passed, actual_reason = _validate_worked_solution_quality(text, "result")
    assert passed is False
    assert actual_reason == reason


def test_worked_solution_quality_accepts_complete_group_and_regression_derivations():
    group_proof = (
        "First, ker(phi) is a subgroup because phi(ab^{-1})=e whenever a,b are in the kernel.\n"
        "For g in G and k in ker(phi), phi(gkg^{-1})=phi(g)ephi(g)^{-1}=e.\n"
        "Therefore gkg^{-1} is in the kernel, so ker(phi) is normal in G."
    )
    regression = (
        "First compute X^T X and X^T y from the stated design matrix.\n"
        "Since X has full column rank, X^T X is invertible and the normal equations have a unique solution.\n"
        "Therefore beta-hat=(X^T X)^{-1}X^T y."
    )
    assert _validate_worked_solution_quality(group_proof, "ker(phi) is normal in G") == (True, "passed")
    assert _validate_worked_solution_quality(regression, "beta-hat=(X^T X)^{-1}X^T y") == (True, "passed")


@pytest.mark.parametrize("answer", ["40", "x=6"])
def test_short_answers_remain_valid(answer):
    assert _validate_short_solution_quality(answer, answer) == (True, "passed")


def test_meta_leak_triggers_one_retry_and_returns_only_clean_second_answer():
    client = FakeClient(
        [
            "Thinking Process: Analyze the Request. Final Answer: 39",
            "<final_solution>40</final_solution>",
        ]
    )
    result = ReasoningAgent(client).solve("Calculate 24+16.", {"answer_type": "number"})

    assert len(client.calls) == 2
    assert result["final_response"] == "40"
    assert "Thinking Process" not in json.dumps(result, ensure_ascii=False)
    assert "retry_model_call" in [event["step"] for event in result["trace"]]
    assert any(
        event["step"] == "output_quality_check" and "meta_reasoning_leak" in event["content"]
        for event in result["trace"]
    )


def test_truncated_first_worked_solution_uses_complete_second_solution():
    complete = (
        "<final_solution>First, let n=2k for an integer k.\n"
        "Then n+n=4k=2(2k), so it is divisible by 2.\n"
        "Therefore n+n is even and the claim follows.</final_solution>"
    )
    client = FakeClient([r"<final_solution>First compute \frac{</final_solution>", complete])

    result = ReasoningAgent(client).solve("Prove that an even integer plus itself is even.", {})

    assert len(client.calls) == 2
    assert "let n=2k" in result["final_response"]
    assert r"\frac{" not in result["final_response"]


def test_two_polluted_attempts_use_safe_fallback_without_leak():
    first = "Thinking Process: Analyze the Request. PRIVATE_DRAFT_ONE"
    second = "Self-Correction: Review against Constraints. PRIVATE_DRAFT_TWO"
    client = FakeClient([first, second])

    result = ReasoningAgent(client).solve("Calculate 1+1.", {"answer_type": "number"})
    serialized = json.dumps(result, ensure_ascii=False)

    assert len(client.calls) == 2
    assert result["final_response"] == FALLBACK_RESPONSE
    assert "PRIVATE_DRAFT_ONE" not in serialized
    assert "PRIVATE_DRAFT_TWO" not in serialized


def test_clean_tagged_short_answer_needs_only_one_call():
    client = FakeClient("<final_solution>x=6</final_solution>")
    result = ReasoningAgent(client).solve("Solve the equation e^x=e^6.", {"answer_type": "expression"})

    assert len(client.calls) == 1
    assert result["final_response"] == "x=6"


def test_verification_passed_requires_math_and_output_quality(monkeypatch):
    monkeypatch.setattr(
        user_agent,
        "verify_solution",
        lambda *args, **kwargs: {"status": "passed", "severity": "none", "issues": []},
    )
    polluted = ReasoningAgent(FakeClient(["Thinking Process: PRIVATE", "Thinking Process: PRIVATE"])).solve(
        "Calculate 1+1.", {}
    )
    clean = ReasoningAgent(FakeClient("<final_solution>2</final_solution>")).solve("Calculate 1+1.", {})

    polluted_summary = _summarize_trace(polluted["trace"], polluted["final_response"])
    clean_summary = _summarize_trace(clean["trace"], clean["final_response"])
    assert polluted_summary["mathematical_verification_status"] == "not_evaluated"
    assert polluted_summary["mathematical_verification_passed"] is False
    assert polluted_summary["output_quality_passed"] is False
    assert polluted_summary["verification_passed"] is False
    assert clean_summary["mathematical_verification_passed"] is False
    assert clean_summary["output_quality_passed"] is True
    assert clean_summary["verification_passed"] is True


def test_invalid_extracted_conclusion_is_never_appended():
    body = (
        "First establish the required bound from the definition.\n"
        "Then substitute it into the preceding identity to finish the argument."
    )
    response = _compose_final_response(
        problem="Prove the result.",
        response_mode=WORKED_SOLUTION,
        solution=body,
        extracted_answer="或定理",
        verification={"status": "passed", "severity": "none", "issues": []},
    )
    assert response == FALLBACK_RESPONSE
    assert "最终结论：或定理" not in response


def test_trace_records_only_quality_reason_not_polluted_text():
    secret = "Thinking Process: Analyze the Request. UNIQUE_PRIVATE_REASONING"
    result = ReasoningAgent(FakeClient([secret, secret])).solve("Calculate 1+1.", {})
    serialized = json.dumps(result["trace"], ensure_ascii=False)

    assert "UNIQUE_PRIVATE_REASONING" not in serialized
    assert "meta_reasoning_leak" in serialized


def test_extract_content_ignores_reasoning_content_and_supports_content_lists():
    response = {
        "choices": [
            {
                "message": {
                    "reasoning_content": "PRIVATE_CHAIN_OF_THOUGHT",
                    "content": [
                        {"type": "reasoning", "text": "HIDDEN_LIST_REASONING"},
                        {"type": "text", "text": "<final_solution>"},
                        {"type": "text", "text": "40</final_solution>"},
                    ],
                }
            }
        ]
    }
    extracted = extract_content(response)
    assert extracted == "<final_solution>40</final_solution>"
    assert "PRIVATE_CHAIN_OF_THOUGHT" not in extracted
    assert "HIDDEN_LIST_REASONING" not in extracted
    assert extract_content({"reasoning_content": "PRIVATE", "content": None}) == ""
    assert extract_content("plain string") == "plain string"


def test_real_client_with_fake_session_returns_only_public_content():
    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "PRIVATE_REASONING",
                            "content": "<final_solution>2</final_solution>",
                        }
                    }
                ]
            }

    class FakeSession:
        def post(self, *args, **kwargs):
            return FakeResponse()

    client = RealInternClient(api_key="fake", session=FakeSession())
    assert client.chat(messages=[]) == "<final_solution>2</final_solution>"


@pytest.mark.parametrize(
    ("problem", "mode", "answer_type"),
    [
        ("Compute phi(100).", SHORT_ANSWER, "number"),
        ("Compute P^2 and verify that every row sums to 1.", WORKED_SOLUTION, "matrix"),
        ("Derive the Bernoulli MLE.", WORKED_SOLUTION, "derivation"),
        ("Derive the normal equations.", WORKED_SOLUTION, "derivation"),
        ("Prove the dominated convergence theorem.", WORKED_SOLUTION, "proof"),
        ("Determine uniform convergence and prove both claims.", WORKED_SOLUTION, "proof"),
    ],
)
def test_problem_driven_mode_and_answer_type(problem, mode, answer_type):
    inferred = _expected_answer_type_from_problem(problem)
    assert inferred == answer_type
    assert _determine_response_mode(problem, None, inferred) == mode


def _complete_proof(tagged=True):
    body = (
        "Let x satisfy the hypotheses. Since the stated bound holds, the defining estimate applies.\n"
        "Therefore the required limit follows, and hence the claim is proved."
    )
    return f"<final_solution>{body}</final_solution>" if tagged else body


def test_worked_unknown_verifier_is_accepted_without_retry(monkeypatch):
    monkeypatch.setattr(
        user_agent,
        "verify_solution",
        lambda *args, **kwargs: {"status": "unknown", "severity": "low", "issues": []},
    )
    client = FakeClient(_complete_proof())
    result = ReasoningAgent(client).solve("Check the advanced claim carefully.", {})

    assert len(client.calls) == 1
    assert result["final_response"] != FALLBACK_RESPONSE
    verify = next(event["content"] for event in result["trace"] if event["step"] == "verify")
    assert "status=unknown" in verify


def test_worked_not_applicable_verifier_is_accepted_without_retry(monkeypatch):
    monkeypatch.setattr(
        user_agent,
        "verify_solution",
        lambda *args, **kwargs: {"status": "failed", "severity": "high", "issues": [{"code": "empty_final_answer"}]},
    )
    solution = (
        "<final_solution>Multiply the two rows by the two columns to obtain P^2.\n"
        "The result is [[3/8,5/8],[5/16,11/16]], whose row sums are both 1.</final_solution>"
    )
    client = FakeClient(solution)
    result = ReasoningAgent(client).solve("Compute P^2 and verify its row sums.", {})

    assert len(client.calls) == 1
    assert result["final_response"] != FALLBACK_RESPONSE
    verify = next(event["content"] for event in result["trace"] if event["step"] == "verify")
    assert "status=not_applicable" in verify


def test_proof_without_dedicated_verifier_remains_not_applicable():
    worked = _normalize_verification(
        {"status": "failed", "severity": "high", "issues": [{"code": "mathematical_contradiction"}]},
        response_mode=WORKED_SOLUTION,
        expected_answer_type="proof",
    )
    assert worked["status"] == VERIFICATION_NOT_APPLICABLE


def test_short_answer_without_dedicated_verifier_is_not_applicable():
    short = _normalize_verification(
        {"status": "uncertain", "severity": "medium", "issues": []},
        response_mode=SHORT_ANSWER,
        expected_answer_type="number",
    )
    assert short["status"] == VERIFICATION_NOT_APPLICABLE


def test_complete_tagged_proof_passes_quality():
    raw = _complete_proof()
    candidate, details = _extract_judgeable_solution_with_diagnostics(raw, response_mode=WORKED_SOLUTION)
    quality = _assess_output_quality(raw, candidate, None, WORKED_SOLUTION, details)
    assert quality["passed"] is True


def test_clean_untagged_proof_passes_quality():
    raw = _complete_proof(tagged=False)
    candidate, details = _extract_judgeable_solution_with_diagnostics(raw, response_mode=WORKED_SOLUTION)
    quality = _assess_output_quality(raw, candidate, None, WORKED_SOLUTION, details)
    assert quality["passed"] is True


def test_opening_only_complete_proof_is_compatibly_recovered():
    raw = "<final_solution>" + _complete_proof(tagged=False)
    candidate, details = _extract_judgeable_solution_with_diagnostics(raw, response_mode=WORKED_SOLUTION)
    quality = _assess_output_quality(raw, candidate, None, WORKED_SOLUTION, details)

    assert candidate
    assert quality["passed"] is True
    assert quality["details"]["subreason"] == "missing_closing_tag"
    assert quality["details"]["compatibility_recovered"] is True


def test_truly_truncated_formula_has_specific_quality_subreason():
    raw = r"<final_solution>First derive $x=\frac{1}{2$ and continue.</final_solution>"
    candidate, details = _extract_judgeable_solution_with_diagnostics(raw, response_mode=WORKED_SOLUTION)
    quality = _assess_output_quality(raw, candidate, "result", WORKED_SOLUTION, details)

    assert quality["passed"] is False
    assert quality["reason"] == "unbalanced_latex"
    assert quality["details"]["subreason"] in {"unbalanced_dollar", "unbalanced_braces"}


def test_569_character_regression_derivation_is_not_a_length_false_positive():
    base = (
        "First expand the least-squares objective and differentiate with respect to beta.\n"
        "Since X has full column rank, X^T X is invertible.\n"
        "Therefore the normal equations give the unique least-squares estimate.\n"
    )
    regression = (base + "a" * 569)[:569]
    assert len(regression) == 569
    assert _validate_worked_solution_quality(regression, None) == (True, "passed")


def test_complete_chinese_proof_is_not_measured_by_english_word_count():
    chinese = "任取满足条件的函数。因为控制函数可积，所以相应估计成立。\n因此极限可以交换，故结论成立。"
    assert _validate_worked_solution_quality(chinese, None) == (True, "passed")


def test_meta_retry_then_not_applicable_is_accepted_and_retry_is_minimal(monkeypatch):
    monkeypatch.setattr(
        user_agent,
        "verify_solution",
        lambda *args, **kwargs: {"status": "passed", "severity": "none", "issues": []},
    )
    polluted = "Thinking Process: Analyze the Request. UNIQUE_FIRST_DRAFT"
    client = FakeClient([polluted, _complete_proof()])
    result = ReasoningAgent(client).solve("Prove the claim.", {})

    assert len(client.calls) == 2
    assert result["final_response"] != FALLBACK_RESPONSE
    assert client.calls[0]["messages"][0] == {"role": "system", "content": PRIMARY_SYSTEM_MESSAGE}
    assert client.calls[1]["messages"][0] == {"role": "system", "content": RETRY_SYSTEM_MESSAGE}
    assert client.calls[1]["temperature"] == 0
    retry_user = client.calls[1]["messages"][1]["content"]
    assert "UNIQUE_FIRST_DRAFT" not in retry_user
    assert "Thinking Process" not in retry_user
    assert "meta_reasoning_leak" not in retry_user
    assert retry_user == "Problem:\nProve the claim.\n\nProduce the final solution now."


def test_two_meta_leaks_still_fall_back_with_two_calls():
    polluted = "Thinking Process: Analyze the Request. PRIVATE"
    client = FakeClient([polluted, polluted])
    result = ReasoningAgent(client).solve("Prove the claim.", {})

    assert len(client.calls) == 2
    assert client.calls[0]["thinking_mode"] is True
    assert client.calls[1]["thinking_mode"] is False
    assert result["final_response"] == FALLBACK_RESPONSE


def test_primary_system_message_is_first_and_exact():
    client = FakeClient("<final_solution>2</final_solution>")
    ReasoningAgent(client).solve("Compute 1+1.", {})
    assert len(client.calls) == 1
    assert client.calls[0]["thinking_mode"] is False
    assert client.calls[0]["messages"][0] == {"role": "system", "content": SHORT_ANSWER_SYSTEM_MESSAGE}


def test_retry_temperature_is_zero():
    client = FakeClient(["Thinking Process: PRIVATE", "<final_solution>2</final_solution>"])
    ReasoningAgent(client).solve("Compute 1+1.", {})
    assert client.calls[1]["temperature"] == 0


def test_retry_user_message_omits_entire_first_response():
    secret = "Thinking Process: UNIQUE_RAW_FIRST_RESPONSE"
    client = FakeClient([secret, "<final_solution>2</final_solution>"])
    ReasoningAgent(client).solve("Compute 1+1.", {})
    retry_user = client.calls[1]["messages"][1]["content"]
    assert "UNIQUE_RAW_FIRST_RESPONSE" not in retry_user
    assert "Thinking Process" not in retry_user
