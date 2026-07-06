from agents.tools.recurrence_sequence_tool import (
    arithmetic_nth,
    arithmetic_sum,
    fibonacci,
    geometric_nth,
    geometric_sum,
    linear_recurrence_first_order,
    linear_recurrence_second_order,
    lucas,
    solve_recurrence_sequence_problem,
)


def test_recurrence_sequence_math_functions():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(10) == 55
    assert lucas(0) == 2
    assert lucas(1) == 1
    assert lucas(5) == 11
    assert arithmetic_nth(3, 2, 10) == 21
    assert arithmetic_sum(3, 2, 10) == 120
    assert geometric_nth(3, 2, 5) == 48
    assert geometric_sum(3, 2, 5) == 93
    assert linear_recurrence_first_order(1, 2, 3, 5) == 125
    assert linear_recurrence_second_order(0, 1, 1, 1, 10) == 55


def test_english_fibonacci_problem():
    result = solve_recurrence_sequence_problem("Find Fibonacci number F_10.")
    assert result is not None
    assert result["final_answer"] == "55"
    assert result["tool_name"] == "recurrence_sequence_tool"


def test_chinese_fibonacci_problem():
    result = solve_recurrence_sequence_problem("求斐波那契数列 F_10")
    assert result is not None
    assert result["final_answer"] == "55"


def test_english_lucas_problem():
    result = solve_recurrence_sequence_problem("Find Lucas number L_5.")
    assert result is not None
    assert result["final_answer"] == "11"


def test_chinese_arithmetic_nth_problem():
    result = solve_recurrence_sequence_problem("等差数列首项为3，公差为2，求第10项")
    assert result is not None
    assert result["final_answer"] == "21"


def test_chinese_arithmetic_sum_problem():
    result = solve_recurrence_sequence_problem("等差数列首项为3，公差为2，求前10项和")
    assert result is not None
    assert result["final_answer"] == "120"


def test_english_geometric_nth_problem():
    result = solve_recurrence_sequence_problem(
        "In a geometric sequence, first term is 3 and common ratio is 2. Find the 5th term."
    )
    assert result is not None
    assert result["final_answer"] == "48"


def test_chinese_geometric_sum_problem():
    result = solve_recurrence_sequence_problem("等比数列首项为3，公比为2，求前5项和")
    assert result is not None
    assert result["final_answer"] == "93"


def test_first_order_recurrence_problem():
    result = solve_recurrence_sequence_problem("a_0=1, a_n=2a_{n-1}+3, 求 a_5")
    assert result is not None
    assert result["final_answer"] == "125"


def test_second_order_recurrence_problem():
    result = solve_recurrence_sequence_problem("a_0=0, a_1=1, a_n=a_{n-1}+a_{n-2}, 求 a_10")
    assert result is not None
    assert result["final_answer"] == "55"


def test_nonmatching_problem_returns_none():
    assert solve_recurrence_sequence_problem("Find the area of a triangle with sides 3, 4, 5.") is None
