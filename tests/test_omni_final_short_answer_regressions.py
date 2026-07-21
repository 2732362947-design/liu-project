from __future__ import annotations

import json
from pathlib import Path

import user_agent
from dev_tools.run_omni_real_api_eval import evaluate_item
from dev_tools.score_omni_results import build_report
from user_agent import (
    FALLBACK_RESPONSE,
    SHORT_ANSWER,
    VERIFICATION_FAILED,
    VERIFICATION_NOT_APPLICABLE,
    VERIFICATION_NOT_EVALUATED,
    VERIFICATION_PASSED,
    ReasoningAgent,
    _assess_output_quality,
    _build_correction_prompt,
    _determine_response_mode,
    _expected_answer_type_from_problem,
    _integer_polynomial_value_gcd,
    _parse_integer_polynomial_value_gcd_problem,
    _prime_factor_multiplicity_up_to_k,
    _reliability_fields,
    _short_repetition_subreason,
    _verify_factorial_floor_exact,
    _verify_integer_polynomial_value_gcd,
)


DATASET = Path(__file__).resolve().parents[1] / "evaluation/datasets/omni_math_smoke_30.jsonl"


def _item(idx: str) -> dict:
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


def _tagged(body: str) -> str:
    return f"<final_solution>{body}</final_solution>"


def _quality(text: str, final_answer: str = "5") -> dict:
    return _assess_output_quality(
        text,
        text,
        final_answer,
        SHORT_ANSWER,
        {"raw_chars": len(text), "clean_candidate_chars": len(text), "tag_status": "absent"},
    )


def test_parse_explicit_integer_polynomial_value_gcd_problem():
    assert _parse_integer_polynomial_value_gcd_problem(_item("omni_eval_001640")["problem"]) == {
        "K": 40,
        "e": 3,
        "N": 2023,
    }


def test_parse_product_notation_integer_polynomial_value_gcd_problem():
    problem = (
        r"Let $P(n)=\prod_{k=1}^{12}(n-k^2)$. Let d be the largest positive integer "
        "that divides P(n) for every integer n>100. Find the number of prime factors "
        "counted with multiplicity."
    )
    assert _parse_integer_polynomial_value_gcd_problem(problem) == {"K": 12, "e": 2, "N": 100}


def test_integer_polynomial_gcd_uses_exactly_k_plus_one_samples():
    _, sample_count = _integer_polynomial_value_gcd(7, 2, 30)
    assert sample_count == 8


def test_integer_polynomial_gcd_factorization_has_multiplicity_48():
    common_divisor, sample_count = _integer_polynomial_value_gcd(40, 3, 2023)
    multiplicity, residual, factors = _prime_factor_multiplicity_up_to_k(common_divisor, 40)
    assert sample_count == 41
    assert factors == {2: 20, 3: 13, 5: 8, 11: 3, 17: 2, 23: 1, 29: 1}
    assert multiplicity == 48
    assert residual == 1


def test_integer_polynomial_candidate_48_passes():
    result = _verify_integer_polynomial_value_gcd(_item("omni_eval_001640")["problem"], "48")
    assert result["status"] == VERIFICATION_PASSED
    assert result["reason"] == "deterministic_integer_polynomial_gcd_verification_passed"
    assert result["subreason"] == "prime_factor_multiplicity_verified"
    assert result["computed_value_summary"] == "integer:48"


def test_integer_polynomial_candidate_3_fails():
    result = _verify_integer_polynomial_value_gcd(_item("omni_eval_001640")["problem"], "3")
    assert result["status"] == VERIFICATION_FAILED
    assert result["reason"] == "short_answer_verification_failed"
    assert result["subreason"] == "prime_factor_multiplicity_mismatch"
    assert result["candidate_value_summary"] == "integer:3"


def test_integer_polynomial_verifier_does_not_read_metadata_expected_answer():
    client = FakeClient(_tagged(r"Exact computation gives \boxed{48}."))
    result = ReasoningAgent(client).solve(
        _item("omni_eval_001640")["problem"], {"expected_answer": "3", "solution": "SECRET"}
    )
    assert result["final_response"] == "48"
    assert result["mathematical_verification_status"] == VERIFICATION_PASSED


def test_ambiguous_integer_polynomial_parameters_are_unknown():
    problem = (
        "Let P(n)=(n-1^3)... be defined. Let d be the largest positive integer that divides "
        "P(n) for every integer n>20. If d is a product of m not necessarily distinct prime "
        "numbers, compute m."
    )
    result = _verify_integer_polynomial_value_gcd(problem, "2")
    assert result["status"] == "unknown"
    assert result["reason"] == "problem_parse_ambiguous"


def test_unexpected_large_residual_is_unknown(monkeypatch):
    monkeypatch.setattr(
        user_agent,
        "_prime_factor_multiplicity_up_to_k",
        lambda value, limit: (1, 101, {2: 1}),
    )
    result = _verify_integer_polynomial_value_gcd(_item("omni_eval_001640")["problem"], "48")
    assert result["status"] == "unknown"
    assert result["reason"] == "unexpected_large_prime_factor"


def test_two_wrong_integer_polynomial_answers_use_deterministic_override():
    client = FakeClient([_tagged(r"Thus \boxed{3}."), _tagged(r"Again \boxed{3}.")])
    result = ReasoningAgent(client).solve(_item("omni_eval_001640")["problem"], {})
    assert len(client.calls) == 2
    assert result["final_response"] == "48"
    assert result["deterministic_answer_override"] is True
    assert result["override_verifier_name"] == "integer_polynomial_value_gcd"


def test_no_dedicated_verifier_never_uses_override():
    client = FakeClient(_tagged(r"The listed factors total \boxed{6}."))
    result = ReasoningAgent(client).solve(_item("omni_eval_000934")["problem"], {})
    assert result["final_response"] == "6"
    assert result["mathematical_verification_status"] == VERIFICATION_NOT_APPLICABLE
    assert result["deterministic_answer_override"] is False


def test_same_substantive_sentence_three_times_is_repetitive():
    sentence = "This exact calculation establishes the requested integer value. "
    assert _short_repetition_subreason(sentence * 3) == "repeated_sentence_loop"


def test_same_boxed_answer_three_times_is_final_answer_loop():
    sentence = r"Thus the answer is \boxed{5}. "
    assert _short_repetition_subreason(sentence * 3) == "repeated_final_answer_loop"


def test_long_nonrepetitive_short_answer_is_not_rejected_for_length_alone():
    text = " ".join(
        f"Step {index} establishes distinct bound {index} from exact term {index * index}."
        for index in range(260)
    ) + r" Therefore the exact final value is \boxed{5}."
    assert len(text) > 12630
    quality = _quality(text)
    assert quality["passed"] is True


def test_long_high_ngram_repetition_is_rejected():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 1000
    quality = _quality(text)
    assert quality["passed"] is False
    assert quality["reason"] == "repetitive_output"
    assert quality["details"]["subreason"] == "high_ngram_repetition"


def test_incomplete_correct_answer_suffix_is_rejected():
    quality = _quality("A concise exact computation follows. The correct answer is", "6")
    assert quality["passed"] is False
    assert quality["reason"] == "repetitive_output"
    assert quality["details"]["subreason"] == "incomplete_repeated_suffix"


def test_single_boxed_conclusion_passes_repetition_gate():
    quality = _quality(r"The exact calculation is complete. Thus the answer is \boxed{5}.")
    assert quality["passed"] is True


def test_repetitive_first_answer_triggers_one_retry():
    repeated = "This exact calculation establishes the requested integer value. " * 3
    client = FakeClient([_tagged(repeated), _tagged(r"Recomputed exactly: \boxed{5}.")])
    result = ReasoningAgent(client).solve("Compute the value of the requested scalar.", {})
    assert len(client.calls) == 2
    assert result["final_response"] == "5"
    assert result["first_attempt_diagnostics"]["quality_reason"] == "repetitive_output"


def test_repetition_retry_prompt_does_not_contain_first_answer():
    prompt = _build_correction_prompt(
        "Compute the requested scalar.",
        {"expected_answer": "5"},
        "UNIQUE_FIRST_RESPONSE The correct answer is boxed 6. " * 4,
        "6",
        {"status": "not_evaluated"},
        response_mode=SHORT_ANSWER,
    )
    assert "UNIQUE_FIRST_RESPONSE" not in prompt
    assert "boxed 6" not in prompt
    assert "Do not repeat the conclusion." in prompt


def test_000934_style_repetition_cannot_pass_quality():
    body = ("The correct answer is boxed 6. " * 390) + "The correct answer is"
    assert len(body) > 12000
    quality = _quality(body, "6")
    assert quality["passed"] is False
    assert quality["reason"] == "repetitive_output"
    assert quality["details"]["subreason"] == "incomplete_repeated_suffix"


def test_factorial_floor_2000_and_2001_regression():
    problem = _item("omni_eval_000817")["problem"]
    assert _verify_factorial_floor_exact(problem, "2000")["status"] == VERIFICATION_PASSED
    wrong = _verify_factorial_floor_exact(problem, "2001")
    assert wrong["status"] == VERIFICATION_FAILED
    assert wrong["subreason"] == "floor_value_mismatch"


def test_two_wrong_factorial_answers_use_deterministic_override():
    client = FakeClient([_tagged(r"Thus \boxed{2001}."), _tagged(r"Again \boxed{2001}.")])
    result = ReasoningAgent(client).solve(_item("omni_eval_000817")["problem"], {})
    assert result["final_response"] == "2000"
    assert result["deterministic_answer_override"] is True
    assert result["override_verifier_name"] == "factorial_floor_exact"


def test_not_applicable_reliability_is_unverified_manual_review():
    fields = _reliability_fields({"status": VERIFICATION_NOT_APPLICABLE}, True, True)
    assert fields["answer_reliability_status"] == "unverified"
    assert fields["manual_review_required"] is True
    assert fields["mathematical_verification_passed"] is False


def test_runner_saves_unverified_numeric_reliability_fields():
    item = _item("omni_eval_000934")
    run = evaluate_item(
        item,
        FakeClient(_tagged(r"The factor count is \boxed{5}.")),
    )
    assert run["pipeline_acceptance_passed"] is True
    assert run["mathematical_verification_status"] == VERIFICATION_NOT_APPLICABLE
    assert run["mathematical_verification_passed"] is False
    assert run["answer_reliability_status"] == "unverified"
    assert run["manual_review_required"] is True


def test_passed_reliability_is_verified_without_manual_review():
    fields = _reliability_fields({"status": VERIFICATION_PASSED}, True, False)
    assert fields["answer_reliability_status"] == "verified"
    assert fields["manual_review_required"] is False
    assert fields["mathematical_verification_passed"] is True


def test_failed_reliability_is_rejected():
    fields = _reliability_fields({"status": VERIFICATION_FAILED}, False, True)
    assert fields["answer_reliability_status"] == "rejected"
    assert fields["pipeline_acceptance_passed"] is False
    assert fields["manual_review_required"] is True


def test_quality_failed_not_evaluated_is_unavailable():
    fields = _reliability_fields({"status": VERIFICATION_NOT_EVALUATED}, False, False)
    assert fields["answer_reliability_status"] == "unavailable"


def test_unavailable_requires_manual_review():
    fields = _reliability_fields({"status": VERIFICATION_NOT_EVALUATED}, False, False)
    assert fields["manual_review_required"] is True


def test_failed_pipeline_with_fallback_is_unavailable_when_math_was_not_evaluated():
    class RaisingClient:
        def chat(self, **kwargs):
            raise RuntimeError("offline test failure")

    result = ReasoningAgent(RaisingClient()).solve("Compute the requested scalar.", {})
    assert result["final_response"] == FALLBACK_RESPONSE
    assert result["pipeline_acceptance_passed"] is False
    assert result["mathematical_verification_status"] == VERIFICATION_NOT_EVALUATED
    assert result["answer_reliability_status"] == "unavailable"
    assert result["manual_review_required"] is True


def test_greatest_common_divisor_infers_numeric_short_answer():
    problem = "Find the greatest common divisor of P(1),...,P(2016)."
    answer_type = _expected_answer_type_from_problem(problem)
    assert answer_type == "number"
    assert _determine_response_mode(problem, "number_theory", answer_type) == SHORT_ANSWER


def test_find_gcd_infers_number():
    assert _expected_answer_type_from_problem("Find the gcd of 84 and 126.") == "number"


def test_find_lcm_infers_number():
    assert _expected_answer_type_from_problem("Find the lcm of 84 and 126.") == "number"


def test_metadata_answer_fields_do_not_participate_in_type_inference():
    client = FakeClient(_tagged(r"The requested expression is \boxed{x+1}."))
    result = ReasoningAgent(client).solve(
        "Find an expression for f(x).",
        {
            "answer_type": "number",
            "expected_answer": "SECRET_EXPECTED",
            "solution": "SECRET_SOLUTION",
        },
    )
    mode_trace = next(event["content"] for event in result["trace"] if event["step"] == "response_mode")
    sent_prompt = "\n".join(message["content"] for message in client.calls[0]["messages"])
    assert "expected_answer_type=expression" in mode_trace
    assert "SECRET_EXPECTED" not in sent_prompt
    assert "SECRET_SOLUTION" not in sent_prompt


def test_unavailable_is_not_counted_as_deterministic_rejection():
    dataset = [
        {"idx": "unavailable", "problem": "Compute.", "expected_answer": "1", "expected_domain": "algebra"}
    ]
    results = [
        {
            "idx": "unavailable",
            "status": "success",
            "final_response": FALLBACK_RESPONSE,
            "predicted_domain": "algebra",
            "output_quality_passed": False,
            "mathematical_verification_status": "not_evaluated",
        }
    ]
    summary = build_report(dataset, results)["summary"]
    assert summary["unavailable_count"] == 1
    assert summary["deterministically_rejected_count"] == 0


def test_report_separates_pipeline_verification_and_override_counts():
    dataset = [
        {"idx": key, "problem": "Compute.", "expected_answer": str(answer), "expected_domain": "algebra"}
        for key, answer in (
            ("verified", 1),
            ("unverified", 2),
            ("rejected", 3),
            ("unavailable", 4),
            ("override", 5),
        )
    ]
    results = [
        {
            "idx": "verified", "status": "success", "final_response": "1",
            "predicted_domain": "algebra", "pipeline_acceptance_passed": True,
            "output_quality_passed": True, "mathematical_verification_status": "passed",
        },
        {
            "idx": "unverified", "status": "success", "final_response": "2",
            "predicted_domain": "algebra", "pipeline_acceptance_passed": True,
            "output_quality_passed": True, "mathematical_verification_status": "not_applicable",
        },
        {
            "idx": "rejected", "status": "success", "final_response": FALLBACK_RESPONSE,
            "predicted_domain": "algebra", "pipeline_acceptance_passed": False,
            "output_quality_passed": True, "mathematical_verification_status": "failed",
        },
        {
            "idx": "unavailable", "status": "success", "final_response": FALLBACK_RESPONSE,
            "predicted_domain": "algebra", "pipeline_acceptance_passed": False,
            "output_quality_passed": False, "mathematical_verification_status": "not_evaluated",
        },
        {
            "idx": "override", "status": "success", "final_response": "5",
            "predicted_domain": "algebra", "pipeline_acceptance_passed": True,
            "output_quality_passed": True, "mathematical_verification_status": "passed",
            "deterministic_answer_override": True,
        },
    ]
    summary = build_report(dataset, results)["summary"]
    assert summary["pipeline_acceptance_count"] == 3
    assert summary["deterministically_verified_count"] == 2
    assert summary["unverified_accepted_count"] == 1
    assert summary["deterministically_rejected_count"] == 1
    assert summary["verified_count"] == 2
    assert summary["unverified_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["unavailable_count"] == 1
    assert summary["deterministic_override_count"] == 1
    assert summary["manual_review_required_count"] == 3
