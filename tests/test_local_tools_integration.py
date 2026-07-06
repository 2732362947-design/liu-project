from user_agent import ReasoningAgent


class CountingClient:
    def __init__(self):
        self.call_count = 0

    def chat(self, *args, **kwargs):
        self.call_count += 1
        raise AssertionError("client.chat should not be called for local tool hits")


def test_all_local_exact_tools_hit_without_calling_client():
    cases = [
        {
            "problem": (
                "Find the smallest positive integer K such that every K-element subset of "
                "{1,2,...,50} contains two distinct elements a,b such that a+b divides ab."
            ),
            "expected": "39",
            "tool_marker": "divisibility_subset",
        },
        {
            "problem": "设 F_81 为 81 元有限域。T={α∈F_81 | F_81=F_3(α)}。求 T 中元素的个数。",
            "expected": "72",
            "tool_marker": "finite_field_tool",
        },
        {
            "problem": "How many derangements are there of 5 elements?",
            "expected": "44",
            "tool_marker": "combinatorics_counting_tool",
        },
        {
            "problem": "Compute Euler's totient function phi(100).",
            "expected": "40",
            "tool_marker": "number_theory_tool",
        },
        {
            "problem": "Find Fibonacci number F_10.",
            "expected": "55",
            "tool_marker": "recurrence_sequence_tool",
        },
        {
            "problem": "Solve the equation 2x + 5 = 17.",
            "expected": "x=6",
            "tool_marker": "elementary_algebra_tool",
        },
    ]

    for case in cases:
        client = CountingClient()
        agent = ReasoningAgent(client)

        result = agent.solve(case["problem"], {"answer_type": "number"})

        assert result["final_response"] == case["expected"]
        assert client.call_count == 0
        assert isinstance(result.get("trace"), list)

        trace_text = str(result["trace"])
        trace_steps = [entry.get("step") for entry in result["trace"]]
        assert "local_tool_solve" in trace_steps
        assert case["tool_marker"] in trace_text
