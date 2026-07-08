from agents.tools.combinatorics_counting_tool import solve_combinatorics_counting_problem
from agents.tools.elementary_algebra_tool import solve_elementary_algebra_problem
from agents.tools.finite_field_tool import solve_finite_field_problem
from agents.tools.number_theory_tool import solve_number_theory_problem
from agents.tools.recurrence_sequence_tool import solve_recurrence_sequence_problem


def test_function_definition_does_not_trigger_elementary_algebra_tool():
    assert solve_elementary_algebra_problem("Given f(x)=x^2, find f(3).") is None


def test_geometry_text_with_x_assignment_does_not_trigger_elementary_algebra_tool():
    problem = "In a geometry diagram, the side label is x=6 and the triangle area is requested."

    assert solve_elementary_algebra_problem(problem) is None


def test_probability_text_with_x_assignment_does_not_trigger_elementary_algebra_tool():
    problem = "In a probability problem, suppose x=6 outcomes are favorable out of 10."

    assert solve_elementary_algebra_problem(problem) is None


def test_bare_f_notation_does_not_trigger_recurrence_sequence_tool():
    assert solve_recurrence_sequence_problem("Compute F_10 for this unrelated function notation.") is None


def test_bare_l_notation_does_not_trigger_recurrence_sequence_tool():
    assert solve_recurrence_sequence_problem("Compute L_5 for this unrelated line label.") is None


def test_group_order_does_not_trigger_number_theory_tool():
    assert solve_number_theory_problem("Find the order of finite group G.") is None


def test_single_congruence_does_not_trigger_crt_tool():
    assert solve_number_theory_problem("Solve x ≡ 2 mod 3.") is None


def test_quadratic_function_does_not_trigger_elementary_algebra_tool():
    assert solve_elementary_algebra_problem("Consider the quadratic function y=x^2-5x+6.") is None


def test_plain_set_problem_does_not_trigger_combinatorics_counting_tool():
    problem = "How many subsets of a 5-element set have exactly 2 elements?"

    assert solve_combinatorics_counting_problem(problem) is None


def test_plain_finite_set_problem_does_not_trigger_finite_field_tool():
    problem = "Let S be a finite set with 81 elements. How many subsets of S have one element?"

    assert solve_finite_field_problem(problem) is None
