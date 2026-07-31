from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from dev_tools.score_omni_results import (
    _candidate_from_prediction,
    score_answer,
)


def _score(
    predicted: object,
    expected: object,
    *,
    profile: str = "exact_v4",
    problem: object = None,
) -> dict:
    return score_answer(
        predicted,
        expected,
        profile=profile,
        problem_text=problem,
    )


@pytest.mark.parametrize(
    ("predicted", "expected", "source"),
    [
        (r"\(\boxed{\frac{1}{2}}\)", "0.5", "boxed"),
        ("−7", "-7", "full_response"),
        ("2^3", "8", "full_response"),
        ("1,200", "1200", "full_response"),
        ("25%", "0.25", "full_response"),
    ],
)
def test_exact_v4_common_exact_forms(
    predicted: str,
    expected: str,
    source: str,
) -> None:
    scored = _score(predicted, expected)

    assert scored["status"] == "correct"
    assert scored["equivalence_class"] == "exact"
    assert scored["candidate_source"] == source


def test_exact_v4_boxed_selection_is_conservative() -> None:
    consistent = _score(r"\boxed{1/2} and \boxed{0.5}", "0.5")
    consistent_mixed_parser = _score(r"\boxed{\sqrt{4}} and \boxed{2}", "2")
    conflicting = _score(r"\boxed{4} and \boxed{5}", "5")
    malformed = _score(r"answer: \boxed{5", "5")

    assert consistent["status"] == "correct"
    assert consistent["candidate_source"] == "consistent_multiple_boxed"
    assert consistent_mixed_parser["status"] == "correct"
    assert conflicting["status"] == "unresolved"
    assert conflicting["reason_code"] == "conflicting_boxed_candidates"
    assert malformed["status"] == "unresolved"
    assert malformed["reason_code"] == "malformed_boxed_candidate"


def test_exact_v4_multiple_explicit_finals_must_agree() -> None:
    conflicting = _score("Final answer: 4\nFinal answer: 5", "5")
    consistent = _score("Final answer: 5\nAnswer: 5.0", "5")

    assert conflicting["status"] == "unresolved"
    assert conflicting["reason_code"] == "conflicting_final_candidates"
    assert consistent["status"] == "correct"
    assert consistent["candidate_source"] == "consistent_multiple_final"


def test_exact_v4_structured_answers_are_conservative() -> None:
    assert _score(r"\{1, 2\}", r"\{2, 1\}")["status"] == "correct"
    assert _score("x=2,3", "x=3,x=2")["status"] == "correct"
    assert _score("(1, 2), (2, 1)", "(2, 1), (1, 2)")["status"] == "correct"

    for unsupported in (
        "[1, 2]",
        r"\begin{pmatrix}1&0\\0&1\end{pmatrix}",
        "4 meters",
    ):
        assert _score(unsupported, unsupported)["status"] == "unresolved"


def test_exact_v4_resource_and_undefined_inputs_are_unresolved() -> None:
    huge_integer = "9" * 300
    too_long = "derivation " * 500
    for value in (
        huge_integer,
        "2^5000",
        "1/0",
        too_long,
        "(" * 40 + "2" + ")" * 40,
    ):
        scored = _score(value, value)
        assert scored["status"] == "unresolved"


def test_candidate_selection_has_no_expected_answer_input() -> None:
    prediction = r"work omitted; therefore \boxed{\frac{3}{4}}"
    candidate = _candidate_from_prediction(prediction)
    source = inspect.getsource(_candidate_from_prediction)

    assert candidate["candidate"] == r"\frac{3}{4}"
    assert candidate["source"] == "boxed"
    assert "expected" not in source
    first = _score(prediction, "0.75")
    second = _score(prediction, "0.5")
    assert first["normalized_prediction"] == second["normalized_prediction"]
    assert first["candidate_source"] == second["candidate_source"]


class _ProblemTextMustNotBeRead:
    def __str__(self) -> str:
        raise AssertionError("exact_v4 read problem_text")


def test_exact_v4_never_reads_problem_text() -> None:
    scored = _score(
        "8",
        "8",
        problem=_ProblemTextMustNotBeRead(),
    )

    assert scored["status"] == "correct"


class _ExpectedMustNotBeRead:
    def __str__(self) -> str:
        raise AssertionError("candidate extraction read expected_answer")


def test_unresolved_candidate_does_not_access_expected_answer() -> None:
    scored = _score(
        r"unfinished \boxed{5",
        _ExpectedMustNotBeRead(),
    )

    assert scored["status"] == "unresolved"
    assert scored["normalized_expected"] == ""


def test_policy_v4_is_separate_from_exact_v4() -> None:
    problem = (
        "A sequence counts by 11s starting at 3. "
        "What is a number that will appear?"
    )
    exact = _score("47", "113", profile="exact_v4", problem=problem)
    policy = _score("47", "113", profile="policy_v4", problem=problem)
    rejected = _score("48", "113", profile="policy_v4", problem=problem)

    assert exact["status"] == "incorrect"
    assert exact["equivalence_class"] == "none"
    assert policy["status"] == "correct"
    assert policy["equivalence_class"] == "policy"
    assert policy["reason_code"] == "policy_arithmetic_progression_member"
    assert rejected["status"] == "incorrect"
    assert rejected["equivalence_class"] == "none"


def test_policy_v4_does_not_guess_without_a_generic_validator() -> None:
    scored = _score(
        "12",
        "10",
        profile="policy_v4",
        problem="Find any integer satisfying the stated conditions.",
    )

    assert scored["status"] == "unresolved"
    assert scored["reason_code"] == "policy_validator_unavailable"


def test_policy_v4_preserves_exact_classification() -> None:
    scored = _score(
        r"\boxed{0.5}",
        r"\frac{1}{2}",
        profile="policy_v4",
        problem="Find any value.",
    )

    assert scored["status"] == "correct"
    assert scored["equivalence_class"] == "exact"
    assert scored["reason_code"] == "exact_equivalent"


@pytest.mark.parametrize(
    ("predicted", "expected", "status"),
    [
        ("0.5", "1/2", "correct"),
        ("x=2,3", "x=3,x=2", "correct"),
        ("4", "5", "incorrect"),
        ("x+1", "x+1", "unresolved"),
    ],
)
def test_legacy_default_matches_frozen_historical_semantics(
    predicted: str,
    expected: str,
    status: str,
) -> None:
    implicit = score_answer(predicted, expected)
    explicit = score_answer(predicted, expected, profile="legacy")

    assert implicit == explicit
    assert explicit["status"] == status
    assert explicit["profile"] == "legacy"


def test_results_are_json_serializable_and_repeatable() -> None:
    first = _score(r"\boxed{(1+\sqrt{2})^2}", r"3+2*\sqrt{2}")
    second = _score(r"\boxed{(1+\sqrt{2})^2}", r"3+2*\sqrt{2}")

    assert first == second
    assert json.loads(json.dumps(first, ensure_ascii=False)) == first
    assert set(first) >= {
        "status",
        "profile",
        "equivalence_class",
        "normalized_prediction",
        "normalized_expected",
        "candidate_source",
        "reason_code",
    }
    assert isinstance(first["normalized_prediction"], str)
    assert isinstance(first["normalized_expected"], str)


def test_dual_counting_contract_uses_arm_rows_and_question_hashes() -> None:
    group_questions = {
        "main": ["main-a", "main-b", "main-c", "main-d"],
        "ab": ["ab-a", "ab-b", "ab-c", "ab-d", "ab-d"],
        "normalization": ["norm-a", "norm-b", "norm-c", "norm-d"],
        "closed_expression": ["closed-a", "closed-a"],
    }
    arms = {
        group: [
            {
                "question": question,
                "arm": f"arm-{position % 2}",
                "classification": (
                    "policy"
                    if group == "ab" and question == "ab-d"
                    else "exact"
                ),
            }
            for position, question in enumerate(questions)
        ]
        for group, questions in group_questions.items()
    }

    hashes = {
        group: {
            hashlib.sha256(row["question"].encode()).hexdigest()
            for row in rows
        }
        for group, rows in arms.items()
    }
    all_rows = [row for rows in arms.values() for row in rows]
    exact_questions = {
        row["question"]
        for row in all_rows
        if row["classification"] == "exact"
    }
    policy_questions = {
        row["question"]
        for row in all_rows
        if row["classification"] == "policy"
    }

    assert {group: len(rows) for group, rows in arms.items()} == {
        "main": 4,
        "ab": 5,
        "normalization": 4,
        "closed_expression": 2,
    }
    assert {group: len(values) for group, values in hashes.items()} == {
        "main": 4,
        "ab": 4,
        "normalization": 4,
        "closed_expression": 1,
    }
    assert len(all_rows) == 15
    assert len(set().union(*hashes.values())) == 13
    assert len(exact_questions) == 12
    assert len(policy_questions) == 1
    for first, first_hashes in hashes.items():
        for second, second_hashes in hashes.items():
            if first < second:
                assert first_hashes.isdisjoint(second_hashes)


def test_duplicate_questions_are_distinct_arm_rows_not_missing_records() -> None:
    ab_rows = [
        ("same-ab-question", "general"),
        ("same-ab-question", "specialized"),
    ]
    closed_rows = [
        ("same-closed-question", "general"),
        ("same-closed-question", "specialized"),
    ]

    for rows in (ab_rows, closed_rows):
        assert len(rows) == 2
        assert len(set(rows)) == 2
        assert len({question for question, _ in rows}) == 1
        assert {arm for _, arm in rows} == {"general", "specialized"}


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported scoring profile"):
        _score("1", "1", profile="future")
