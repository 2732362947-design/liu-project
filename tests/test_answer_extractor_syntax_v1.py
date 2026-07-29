from __future__ import annotations

from agents.answer_extractor_agent import (
    MAX_EXTRACTOR_INPUT_CHARS,
    MAX_LATEX_NESTING_DEPTH,
    MAX_STRUCTURED_CANDIDATES,
    _extract_balanced_boxed_candidates,
    _extract_boxed_answer,
    _extract_marked_math_block,
    extract_final_answer,
)
def test_existing_simple_boxed_fraction_is_extracted() -> None:
    assert _extract_boxed_answer(r"Thus \boxed{\frac{2}{7}}.") == r"\frac{2}{7}"


def test_nested_dfrac_is_extracted_as_a_complete_fraction() -> None:
    result = extract_final_answer(
        "Find the fictional probability.",
        r"The calculation is complete. \boxed{\dfrac{1}{501}}",
        "probability",
    )

    assert result["final_answer"] == r"\dfrac{1}{501}"
    assert "boxed" in result["reason"]


def test_nested_text_and_math_wrappers_are_preserved_completely() -> None:
    assert (
        _extract_boxed_answer(r"\boxed{\(\mathrm{\dfrac{3}{8}}\)}")
        == r"\(\mathrm{\dfrac{3}{8}}\)"
    )


def test_fraction_arguments_may_contain_signs_or_simple_expressions() -> None:
    assert (
        _extract_boxed_answer(r"\boxed{\dfrac{-a+1}{b-2}}")
        == r"\dfrac{-a+1}{b-2}"
    )


def test_final_answer_marker_accepts_one_complete_adjacent_math_block() -> None:
    solution = (
        "The explanation is complete.\n"
        "**Final Answer:**\n"
        r"\[\dfrac{-2}{7}\]"
        "\nThis denominator is nonzero."
    )

    assert _extract_marked_math_block(solution) == r"\[\dfrac{-2}{7}\]"
    result = extract_final_answer("Find a fictional ratio.", solution, "algebra")
    assert result["final_answer"] == r"\[\dfrac{-2}{7}\]"
    assert "keyword" in result["reason"]


def test_existing_keyword_priority_precedes_boxed_priority() -> None:
    result = extract_final_answer(
        "Find the fictional value.",
        "Final answer is 9.\n" + r"\boxed{8}",
        "algebra",
    )

    assert result["final_answer"] == "9"
    assert "keyword" in result["reason"]


def test_incomplete_box_is_not_repaired() -> None:
    assert _extract_boxed_answer(r"\boxed{\dfrac{2}{7}") is None


def test_denominator_only_does_not_create_a_fraction() -> None:
    result = extract_final_answer(
        "Find a fictional value.",
        "The only explicit value supplied is 7.",
        "algebra",
    )

    assert result["final_answer"] == "7"
    assert result["final_answer"] != "1/7"


def test_conflicting_multiline_final_markers_are_not_resolved_by_new_rule() -> None:
    solution = (
        "Final Answer:\n"
        r"\[\dfrac{1}{2}\]"
        "\nFinal Answer:\n"
        r"\[\dfrac{2}{3}\]"
    )

    assert _extract_marked_math_block(solution) is None


def test_problem_quote_without_final_marker_is_not_a_new_math_candidate() -> None:
    solution = (
        r"The problem mentions \dfrac{2}{7}, but the derivation is unfinished."
    )

    assert _extract_marked_math_block(solution) is None
    assert _extract_boxed_answer(solution) is None


def test_prose_and_cross_language_behavior_is_unchanged() -> None:
    result = extract_final_answer(
        "Determine whether a fictional object exists.",
        "不存在这样的对象。",
        "algebra",
    )

    assert result["final_answer"] == "不存在"
    assert "no_solution" in result["reason"]


def test_long_unmarked_text_is_not_searched_for_a_target_substring() -> None:
    solution = ("Unfinished explanation without a final marker. " * 200) + "2/7"

    assert _extract_marked_math_block(solution) is None
    assert _extract_boxed_answer(solution) is None


def test_balanced_parser_limits_depth_length_and_candidate_count() -> None:
    too_deep = (
        r"\boxed{"
        + "{" * MAX_LATEX_NESTING_DEPTH
        + "2"
        + "}" * MAX_LATEX_NESTING_DEPTH
        + "}"
    )
    too_long = "x" * MAX_EXTRACTOR_INPUT_CHARS + r"\boxed{2}"
    many = " ".join(
        rf"\boxed{{{number}}}"
        for number in range(MAX_STRUCTURED_CANDIDATES + 5)
    )

    assert _extract_boxed_answer(too_deep) is None
    assert _extract_boxed_answer(too_long) is None
    assert len(_extract_balanced_boxed_candidates(many)) == MAX_STRUCTURED_CANDIDATES
