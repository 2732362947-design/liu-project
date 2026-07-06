from agents.tools.combinatorics_counting_tool import (
    comb,
    derangement,
    solve_combinatorics_counting_problem,
    stirling_second_kind,
    surjection_count,
)


def test_combinatorics_counting_math_functions():
    assert derangement(0) == 1
    assert derangement(1) == 0
    assert derangement(5) == 44
    assert derangement(6) == 265
    assert stirling_second_kind(5, 2) == 15
    assert stirling_second_kind(6, 3) == 90
    assert stirling_second_kind(3, 5) == 0
    assert surjection_count(4, 2) == 14
    assert surjection_count(5, 3) == 150
    assert comb(5, 2) == 10
    assert comb(5, 7) == 0


def test_combinatorics_counting_rejects_invalid_comb_k():
    assert comb(5, -1) == 0


def test_english_derangement_problem():
    result = solve_combinatorics_counting_problem("How many derangements are there of 5 elements?")
    assert result is not None
    assert result["final_answer"] == "44"
    assert result["tool_name"] == "combinatorics_counting_tool"
    assert result["details"]["problem_type"] == "derangement_count"


def test_chinese_derangement_problem():
    result = solve_combinatorics_counting_problem("求 6 个元素的错排数")
    assert result is not None
    assert result["final_answer"] == "265"


def test_english_onto_problem():
    result = solve_combinatorics_counting_problem(
        "How many onto functions are there from a 4-element set to a 2-element set?"
    )
    assert result is not None
    assert result["final_answer"] == "14"
    assert result["details"]["problem_type"] == "surjection_count"


def test_chinese_surjection_problem():
    result = solve_combinatorics_counting_problem("从 5 元集合到 3 元集合的满射有多少个？")
    assert result is not None
    assert result["final_answer"] == "150"


def test_english_stirling_problem():
    result = solve_combinatorics_counting_problem("Compute the Stirling number of the second kind S(5,2).")
    assert result is not None
    assert result["final_answer"] == "15"
    assert result["details"]["problem_type"] == "stirling_second_kind"


def test_chinese_stirling_problem():
    result = solve_combinatorics_counting_problem("求第二类 Stirling 数 S(6,3)")
    assert result is not None
    assert result["final_answer"] == "90"


def test_nonmatching_problem_returns_none():
    assert solve_combinatorics_counting_problem("How many subsets of a 5-element set have size 2?") is None


def test_s_notation_without_stirling_signal_returns_none():
    assert solve_combinatorics_counting_problem("Compute S(5,2).") is None


def test_combinatorics_counting_tool_zero_api_calls():
    from user_agent import ReasoningAgent

    class CountingClient:
        def __init__(self):
            self.call_count = 0

        def chat(self, **kwargs):
            self.call_count += 1
            return "This should not be called"

    client = CountingClient()
    agent = ReasoningAgent(client)

    result = agent.solve("How many derangements are there of 5 elements?", {"answer_type": "number"})

    assert result["final_response"] == "44"
    assert client.call_count == 0
    assert "combinatorics_counting_tool" in str(result["trace"])
