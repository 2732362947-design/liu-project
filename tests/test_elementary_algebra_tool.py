from agents.tools.elementary_algebra_tool import (
    solve_2x2_linear_system,
    solve_elementary_algebra_problem,
    solve_linear_equation,
    solve_quadratic_integer_roots,
)


def test_elementary_algebra_math_functions():
    assert solve_linear_equation(2, 5, 17) == 6
    assert solve_linear_equation(3, -4, 11) == 5
    assert solve_quadratic_integer_roots(1, -5, 6) == (2, 3)
    assert solve_quadratic_integer_roots(1, 0, -9) == (-3, 3)
    assert solve_2x2_linear_system(2, 1, 5, 1, -1, 1) == (2, 1)


def test_linear_equation_problem():
    result = solve_elementary_algebra_problem("Solve the equation 2x + 5 = 17.")
    assert result is not None
    assert result["final_answer"] == "x=6"
    assert result["tool_name"] == "elementary_algebra_tool"


def test_chinese_linear_equation_problem():
    result = solve_elementary_algebra_problem("解方程 3x - 4 = 11")
    assert result is not None
    assert result["final_answer"] == "x=5"


def test_quadratic_integer_roots_problem():
    result = solve_elementary_algebra_problem("Solve the equation x^2 - 5x + 6 = 0.")
    assert result is not None
    assert "2" in result["final_answer"]
    assert "3" in result["final_answer"]


def test_quadratic_square_problem():
    result = solve_elementary_algebra_problem("Solve the equation x^2 = 9.")
    assert result is not None
    assert "-3" in result["final_answer"]
    assert "3" in result["final_answer"]


def test_2x2_linear_system_problem():
    result = solve_elementary_algebra_problem("Solve the system 2x + y = 5, x - y = 1.")
    assert result is not None
    assert result["final_answer"] == "x=2, y=1"


def test_no_solve_signal_returns_none():
    assert solve_elementary_algebra_problem("The expression 2x + 5 = 17 appears in a sentence.") is None


def test_unsupported_complex_or_decimal_problem_returns_none():
    assert solve_elementary_algebra_problem("Solve the equation x^2 + 1 = 0.") is None
    assert solve_elementary_algebra_problem("Solve the equation 0.5x + 1 = 2.") is None


def test_elementary_algebra_tool_zero_api_calls():
    from user_agent import ReasoningAgent

    class CountingClient:
        def __init__(self):
            self.call_count = 0

        def chat(self, **kwargs):
            self.call_count += 1
            raise AssertionError("client.chat should not be called")

    client = CountingClient()
    result = ReasoningAgent(client).solve("Solve the equation 2x + 5 = 17.", {"answer_type": "number"})

    assert result["final_response"] == "x=6"
    assert client.call_count == 0
    assert "elementary_algebra_tool" in str(result["trace"])


def test_elementary_algebra_quadratic_zero_api_calls():
    from user_agent import ReasoningAgent

    class CountingClient:
        def __init__(self):
            self.call_count = 0

        def chat(self, **kwargs):
            self.call_count += 1
            raise AssertionError("client.chat should not be called")

    client = CountingClient()
    result = ReasoningAgent(client).solve("Solve the equation x^2 - 5x + 6 = 0.", {"answer_type": "number"})

    assert result["final_response"] == "x=2,3"
    assert client.call_count == 0
    assert "elementary_algebra_tool" in str(result["trace"])


def test_elementary_algebra_system_zero_api_calls():
    from user_agent import ReasoningAgent

    class CountingClient:
        def __init__(self):
            self.call_count = 0

        def chat(self, **kwargs):
            self.call_count += 1
            raise AssertionError("client.chat should not be called")

    client = CountingClient()
    result = ReasoningAgent(client).solve("Solve the system 2x + y = 5, x - y = 1.", {"answer_type": "number"})

    assert result["final_response"] == "x=2, y=1"
    assert client.call_count == 0
    assert "elementary_algebra_tool" in str(result["trace"])
