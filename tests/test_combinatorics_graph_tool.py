from agents.tools.combinatorics_graph_tool import (
    build_divisibility_graph,
    detect_divisibility_subset_problem,
    is_independent_set,
    normalize_problem_text,
    solve_divisibility_subset_problem,
)
from user_agent import ReasoningAgent


PROBLEM = (
    "Find the smallest positive integer K such that every K-element subset of "
    "{1,2,...,50} contains two distinct elements a,b such that a+b divides ab."
)
OMNI_LATEX_PROBLEM = (
    r"Find the smallest positive integer $ K$ such that every $ K$-element subset of "
    r"$ \{1,2,...,50 \}$ contains two distinct elements $a,b$ such that $ a\plus{}b$ divides $ ab$."
)


class ExplodingClient:
    calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("FakeClient should not be called for local exact tool problems")


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_detect_divisibility_subset_problem_plain_text():
    result = detect_divisibility_subset_problem(PROBLEM)

    assert result == {"type": "divisibility_subset_graph", "n": 50}


def test_detect_divisibility_subset_problem_latex_text():
    problem = (
        r"Find the smallest positive integer K such that every K-element subset of "
        r"$ \{1,2,\dots,50\}$ contains two distinct elements a,b such that $a+b \mid ab$."
    )

    result = detect_divisibility_subset_problem(problem)

    assert result == {"type": "divisibility_subset_graph", "n": 50}


def test_normalize_problem_text_handles_latex_plus_variants():
    variants = [
        r"a+b divides ab",
        r"a + b divides ab",
        r"a\plus{}b divides ab",
        r"a\plus b divides ab",
        r"a+b \mid ab",
        r"$ a\plus{}b$ divides $ ab$",
    ]

    for variant in variants:
        normalized = normalize_problem_text(variant)
        assert "a+b divides ab" in normalized or "a+b|ab" in normalized


def test_detect_divisibility_subset_problem_real_omni_latex_text():
    result = detect_divisibility_subset_problem(OMNI_LATEX_PROBLEM)

    assert result == {"type": "divisibility_subset_graph", "n": 50}


def test_solve_real_omni_latex_text_is_not_none():
    result = solve_divisibility_subset_problem(OMNI_LATEX_PROBLEM)

    assert result is not None
    assert result["final_answer"] == "39"


def test_build_divisibility_graph_edges_satisfy_formula():
    n = 12
    graph = build_divisibility_graph(n)

    for a, neighbors in graph.items():
        for b in neighbors:
            assert 1 <= a <= n
            assert 1 <= b <= n
            assert a != b
            assert (a * b) % (a + b) == 0
            assert a in graph[b]


def test_solve_divisibility_subset_problem_n_50_returns_exact_algorithm_result():
    result = solve_divisibility_subset_problem(PROBLEM)

    assert result is not None
    assert result["final_answer"] == "39"
    details = result["details"]
    assert details["n"] == 50
    assert details["max_independent_set_size"] == 38
    assert details["K"] == 39
    assert details["isolated_count"] == len(details["isolated_vertices"])
    assert details["isolated_count"] >= 26
    assert len(details["one_max_independent_set"]) == 38
    assert "精确最大独立集搜索" in details["verification_note"]


def test_witness_is_independent_set_for_n_50():
    graph = build_divisibility_graph(50)
    result = solve_divisibility_subset_problem(PROBLEM)
    witness = result["details"]["one_max_independent_set"]

    assert len(witness) == result["details"]["max_independent_set_size"]
    assert is_independent_set(witness, graph) is True


def test_n_50_has_26_isolated_vertices_so_k_26_cannot_force_an_edge():
    graph = build_divisibility_graph(50)
    isolated = [vertex for vertex, neighbors in graph.items() if not neighbors]

    assert len(isolated) == 26


def test_user_agent_uses_local_tool_without_calling_client():
    client = ExplodingClient()

    result = ReasoningAgent(client).solve(PROBLEM, {"answer_type": "number"})

    assert result["final_response"] == "39"
    assert client.calls == []
    steps = [item["step"] for item in result["trace"]]
    assert "local_tool_detect" in steps
    assert "local_tool_solve" in steps
    solve_trace = next(item["content"] for item in result["trace"] if item["step"] == "local_tool_solve")
    assert "max_independent_set_size=38" in solve_trace
    assert "K=39" in solve_trace
    assert "isolated_count=26" in solve_trace


def test_nonmatching_problem_uses_client():
    client = FakeClient("最终答案：2")

    result = ReasoningAgent(client).solve("1+1=?", {})

    assert result["final_response"] == "2"
    assert len(client.calls) == 1


def test_metadata_answers_do_not_affect_local_tool_result():
    result = ReasoningAgent(ExplodingClient()).solve(
        PROBLEM,
        {"answer": "999", "expected_answer": "26", "answer_type": "number"},
    )

    assert result["final_response"] == "39"


def test_user_agent_uses_local_tool_for_real_omni_latex_text_without_calling_client():
    client = ExplodingClient()

    result = ReasoningAgent(client).solve(
        OMNI_LATEX_PROBLEM,
        {"answer": "999", "expected_answer": "26", "answer_type": "number"},
    )

    assert result["final_response"] == "39"
    assert client.calls == []
    steps = [item["step"] for item in result["trace"]]
    assert "local_tool_detect" in steps
    assert "local_tool_solve" in steps
