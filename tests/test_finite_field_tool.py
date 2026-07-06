from __future__ import annotations

import pytest

from agents.tools.finite_field_tool import (
    divisors,
    factor_int,
    is_prime,
    mobius,
    prime_power_decomposition,
    solve_finite_field_problem,
)


class TestIsPrime:
    def test_small_primes(self):
        assert is_prime(2) is True
        assert is_prime(3) is True
        assert is_prime(5) is True
        assert is_prime(7) is True

    def test_non_primes(self):
        assert is_prime(0) is False
        assert is_prime(1) is False
        assert is_prime(4) is False
        assert is_prime(6) is False
        assert is_prime(8) is False
        assert is_prime(9) is False

    def test_larger_prime(self):
        assert is_prime(97) is True
        assert is_prime(100) is False


class TestFactorInt:
    def test_prime(self):
        assert factor_int(7) == {7: 1}

    def test_composite(self):
        assert factor_int(12) == {2: 2, 3: 1}

    def test_prime_power(self):
        assert factor_int(81) == {3: 4}
        assert factor_int(16) == {2: 4}
        assert factor_int(8) == {2: 3}

    def test_one(self):
        assert factor_int(1) == {}


class TestPrimePowerDecomposition:
    def test_81(self):
        assert prime_power_decomposition(81) == (3, 4)

    def test_16(self):
        assert prime_power_decomposition(16) == (2, 4)

    def test_8(self):
        assert prime_power_decomposition(8) == (2, 3)

    def test_4(self):
        assert prime_power_decomposition(4) == (2, 2)

    def test_prime(self):
        assert prime_power_decomposition(5) == (5, 1)

    def test_not_prime_power(self):
        assert prime_power_decomposition(6) is None
        assert prime_power_decomposition(12) is None

    def test_invalid(self):
        assert prime_power_decomposition(1) is None
        assert prime_power_decomposition(0) is None


class TestDivisors:
    def test_divisors_of_4(self):
        assert divisors(4) == [1, 2, 4]

    def test_divisors_of_6(self):
        assert divisors(6) == [1, 2, 3, 6]

    def test_divisors_of_1(self):
        assert divisors(1) == [1]

    def test_divisors_of_12(self):
        assert divisors(12) == [1, 2, 3, 4, 6, 12]

    def test_invalid(self):
        assert divisors(0) == []
        assert divisors(-1) == []


class TestMobius:
    def test_mobius_1(self):
        assert mobius(1) == 1

    def test_mobius_2(self):
        assert mobius(2) == -1

    def test_mobius_4(self):
        assert mobius(4) == 0

    def test_mobius_6(self):
        assert mobius(6) == 1

    def test_mobius_3(self):
        assert mobius(3) == -1

    def test_mobius_12(self):
        assert mobius(12) == 0

    def test_mobius_30(self):
        assert mobius(30) == -1

    def test_mobius_invalid(self):
        assert mobius(0) == 0
        assert mobius(-1) == 0


class TestSolveFiniteFieldProblem:
    def test_f81_over_f3_generator_count(self):
        problem = (
            "Let F_{81} be the finite field with 81 elements. "
            "Let T be the set of elements alpha in F_{81} such that "
            "F_{81} = F_3(alpha). Find the number of elements in T."
        )
        result = solve_finite_field_problem(problem)
        assert result is not None
        assert result["tool_name"] == "finite_field_tool"
        assert result["details"]["problem_type"] == "finite_field_generator_count"
        assert result["details"]["p"] == 3
        assert result["details"]["n"] == 4
        assert result["details"]["q"] == 81
        assert result["details"]["answer"] == 72
        assert result["final_answer"] == "72"

    def test_f16_over_f2_generator_count(self):
        problem = (
            "Consider the finite field F_{16} with 16 elements. "
            "How many elements alpha are there such that F_{16} = F_2(alpha)? "
            "Find the number of such elements."
        )
        result = solve_finite_field_problem(problem)
        assert result is not None
        assert result["details"]["p"] == 2
        assert result["details"]["n"] == 4
        assert result["details"]["q"] == 16
        assert result["details"]["answer"] == 12
        assert result["final_answer"] == "12"

    def test_f8_over_f2_generator_count(self):
        problem = (
            "Let GF(8) be the finite field with 8 elements. "
            "Find the number of elements alpha such that GF(8) = GF(2)(alpha)."
        )
        result = solve_finite_field_problem(problem)
        assert result is not None
        assert result["details"]["p"] == 2
        assert result["details"]["n"] == 3
        assert result["details"]["q"] == 8
        assert result["details"]["answer"] == 6
        assert result["final_answer"] == "6"

    def test_no_match_returns_none(self):
        problem = "What is 2 + 3?"
        result = solve_finite_field_problem(problem)
        assert result is None

    def test_no_match_geometry(self):
        problem = "Find the area of a circle with radius 5."
        result = solve_finite_field_problem(problem)
        assert result is None

    def test_chinese_generator_problem(self):
        problem = (
            "设 F_{27} 为 27 元有限域。求 T 中元素个数，其中 T 是满足 "
            "F_{27} = F_3(alpha) 的所有 alpha 的集合。"
        )
        result = solve_finite_field_problem(problem)
        assert result is not None
        assert result["details"]["p"] == 3
        assert result["details"]["n"] == 3
        assert result["details"]["q"] == 27
        assert result["details"]["answer"] == 24
        assert result["final_answer"] == "24"

    def test_f4_over_f2(self):
        problem = (
            "Let F_4 be the finite field with 4 elements. "
            "How many elements generate the extension F_4 over F_2? "
            "Find the number of such primitive elements."
        )
        result = solve_finite_field_problem(problem)
        assert result is not None
        assert result["details"]["p"] == 2
        assert result["details"]["n"] == 2
        assert result["details"]["q"] == 4
        assert result["details"]["answer"] == 2
        assert result["final_answer"] == "2"


class TestMonicIrreduciblePolynomial:
    def test_f2_degree_3(self):
        problem = (
            "How many monic irreducible polynomials of degree 3 "
            "are there over F_2?"
        )
        result = solve_finite_field_problem(problem)
        assert result is not None
        assert result["details"]["problem_type"] == "monic_irreducible_polynomial_count"
        assert result["details"]["p"] == 2
        assert result["details"]["n"] == 3
        assert result["details"]["answer"] == 2
        assert result["final_answer"] == "2"

    def test_f3_degree_2(self):
        problem = (
            "Find the number of monic irreducible polynomials of degree 2 "
            "over GF(3)."
        )
        result = solve_finite_field_problem(problem)
        assert result is not None
        assert result["details"]["problem_type"] == "monic_irreducible_polynomial_count"
        assert result["details"]["p"] == 3
        assert result["details"]["n"] == 2
        assert result["details"]["answer"] == 3
        assert result["final_answer"] == "3"

    def test_chinese_irreducible(self):
        problem = (
            "求 F_5 上 2 次首一不可约多项式的个数。"
        )
        result = solve_finite_field_problem(problem)
        assert result is not None
        assert result["details"]["problem_type"] == "monic_irreducible_polynomial_count"
        assert result["details"]["p"] == 5
        assert result["details"]["n"] == 2
        assert result["details"]["answer"] == 10
        assert result["final_answer"] == "10"


class TestUserAgentFiniteFieldIntegration:
    def test_finite_field_tool_zero_api_calls(self):
        from user_agent import ReasoningAgent

        class CountingClient:
            def __init__(self):
                self.call_count = 0

            def chat(self, **kwargs):
                self.call_count += 1
                return "This should not be called"

        client = CountingClient()
        agent = ReasoningAgent(client)

        problem = (
            "Let F_{81} be the finite field with 81 elements. "
            "Let T be the set of elements alpha in F_{81} such that "
            "F_{81} = F_3(alpha). Find the number of elements in T."
        )
        result = agent.solve(problem, {"answer_type": "number"})

        assert result["final_response"] == "72"
        assert client.call_count == 0
        trace_steps = [entry["step"] for entry in result["trace"]]
        assert "local_tool_solve" in trace_steps
        assert "finite_field_tool" in str(result["trace"])

    def test_finite_field_tool_with_exception_client(self):
        from user_agent import ReasoningAgent

        class ExplodingClient:
            def chat(self, **kwargs):
                raise RuntimeError("API unavailable")

        client = ExplodingClient()
        agent = ReasoningAgent(client)

        problem = (
            "Let F_{81} be the finite field with 81 elements. "
            "Let T be the set of elements alpha in F_{81} such that "
            "F_{81} = F_3(alpha). Find the number of elements in T."
        )
        result = agent.solve(problem, {"answer_type": "number"})

        assert result["final_response"] == "72"
        trace_steps = [entry["step"] for entry in result["trace"]]
        assert "local_tool_solve" in trace_steps

    def test_finite_field_tool_f16(self):
        from user_agent import ReasoningAgent

        class CountingClient:
            def __init__(self):
                self.call_count = 0

            def chat(self, **kwargs):
                self.call_count += 1
                return "Should not be called"

        client = CountingClient()
        agent = ReasoningAgent(client)

        problem = (
            "Consider the finite field F_{16} with 16 elements. "
            "How many elements alpha are there such that F_{16} = F_2(alpha)?"
        )
        result = agent.solve(problem, {"answer_type": "number"})

        assert result["final_response"] == "12"
        assert client.call_count == 0