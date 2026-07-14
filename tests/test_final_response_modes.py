import json

import pytest

import user_agent
from user_agent import (
    FALLBACK_RESPONSE,
    SHORT_ANSWER,
    WORKED_SOLUTION,
    ReasoningAgent,
    _compose_final_response,
    _determine_response_mode,
    _safe_metadata,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


def _assert_response_schema(result: dict) -> None:
    assert isinstance(result, dict)
    assert isinstance(result["final_response"], str)
    assert result["final_response"].strip()
    json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize(
    "problem",
    [
        "证明这个命题。",
        "试证该结论。",
        "由此证得结论。",
        "推导所给公式。",
        "从定义导出公式。",
        "论证该命题。",
        "说明为什么该函数连续。",
        "解释为什么结论成立。",
        "请说明该步骤。",
        "给出证明。",
        "证明下列结论。",
        "Prove the claim.",
        "Show that the sequence converges.",
        "Derive the identity.",
        "Justify the equality.",
        "Explain why the map is continuous.",
        "Demonstrate the result.",
        "Give a proof of the claim.",
    ],
)
def test_worked_solution_strong_signals(problem):
    assert _determine_response_mode(problem, None, None) == WORKED_SOLUTION


@pytest.mark.parametrize(
    "problem",
    ["选择正确选项。", "填空：1+1=__。", "求值 24+16。", "计算 1/4+1/4。", "解方程 2x=12。"],
)
def test_short_answer_defaults(problem):
    assert _determine_response_mode(problem, None, None) == SHORT_ANSWER


def test_proof_domain_defaults_to_worked_but_analysis_calculation_stays_short():
    assert _determine_response_mode("Establish the claim.", "proof", None) == WORKED_SOLUTION
    assert _determine_response_mode("Compute the limit.", "real_analysis", "number") == SHORT_ANSWER
    assert _determine_response_mode("Calculate the invariant.", "topology", "number") == SHORT_ANSWER


def test_numeric_calculation_keeps_only_verified_answer():
    solution = "先合并十位与个位，24+16=40。\n最终答案：40"
    result = ReasoningAgent(FakeClient(solution)).solve("计算 24+16。", {"answer_type": "number"})

    assert result["final_response"] == "40"
    _assert_response_schema(result)


def test_linear_equation_keeps_short_answer():
    result = ReasoningAgent(FakeClient("移项得到 2x=12。\n最终答案：x=6")).solve(
        "解方程 2x+3=15。",
        {"answer_type": "expression"},
    )

    assert result["final_response"] == "x=6"
    _assert_response_schema(result)


def test_chinese_proof_preserves_full_reasoning():
    solution = (
        "第一步，任取偶数 n，由定义可写成 n=2k。\n\n"
        "第二步，因为 n+n=4k=2(2k)，所以 n+n 仍能被 2 整除。\n\n"
        "因此，两个相同偶数之和仍为偶数，命题成立。"
    )
    result = ReasoningAgent(FakeClient(solution)).solve("证明：任意偶数与自身之和仍是偶数。", {})

    assert "第一步" in result["final_response"]
    assert "第二步" in result["final_response"]
    assert "命题成立" in result["final_response"]
    assert result["final_response"] != "命题成立"
    _assert_response_schema(result)


def test_english_proof_preserves_full_reasoning():
    solution = (
        "Let n be even, so n=2k for some integer k.\n\n"
        "Then n+n=4k=2(2k), which is divisible by 2.\n\n"
        "Therefore n+n is even, and the claim follows."
    )
    result = ReasoningAgent(FakeClient(solution)).solve("Prove that the sum of an even integer with itself is even.", {})

    assert "Let n be even" in result["final_response"]
    assert "Then n+n=4k" in result["final_response"]
    assert "claim follows" in result["final_response"]
    _assert_response_schema(result)


def test_derivation_preserves_key_steps():
    solution = (
        "从等差数列定义 a_k=a_1+(k-1)d 出发。\n"
        "将首尾两种次序相加，得到 2S_n=n(a_1+a_n)。\n"
        "因此 S_n=n(a_1+a_n)/2。\n最终答案：S_n=n(a_1+a_n)/2"
    )
    result = ReasoningAgent(FakeClient(solution)).solve("推导等差数列前 n 项和公式。", {"answer_type": "expression"})

    assert "2S_n" in result["final_response"]
    assert "从等差数列定义" in result["final_response"]
    _assert_response_schema(result)


def test_explanation_preserves_body():
    solution = (
        "A polynomial is a finite sum of continuous monomials.\n"
        "Finite sums of continuous functions are continuous.\n"
        "Therefore every polynomial is continuous.\n"
        "The final answer is that every polynomial is continuous."
    )
    result = ReasoningAgent(FakeClient(solution)).solve("Explain why every polynomial is continuous.", {})

    assert "finite sum" in result["final_response"]
    assert "Finite sums" in result["final_response"]
    _assert_response_schema(result)


def test_retry_proof_uses_second_full_solution(monkeypatch):
    calls = {"count": 0}

    def fake_verify(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "failed", "severity": "high", "issues": [{"code": "bad_proof"}]}
        return {"status": "passed", "severity": "none", "issues": []}

    monkeypatch.setattr(user_agent, "verify_solution", fake_verify)
    client = FakeClient(
        [
            "第一次错误步骤。\n因此得到错误结论。",
            "第二次先任取 n=2k。\n于是 n+n=4k=2(2k)。\n因此第二次证明成立。",
        ]
    )
    result = ReasoningAgent(client).solve("证明偶数与自身之和仍是偶数。", {})

    assert "第二次先任取" in result["final_response"]
    assert "第一次错误步骤" not in result["final_response"]
    _assert_response_schema(result)


def test_retry_calculation_uses_second_short_answer(monkeypatch):
    calls = {"count": 0}

    def fake_verify(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "failed", "severity": "high", "issues": [{"code": "bad"}]}
        return {"status": "passed", "severity": "none", "issues": []}

    monkeypatch.setattr(user_agent, "verify_solution", fake_verify)
    result = ReasoningAgent(FakeClient(["错误计算。\n最终答案：39", "重新计算 24+16=40。\n最终答案：40"])).solve(
        "计算 24+16。",
        {"answer_type": "number"},
    )

    assert result["final_response"] == "40"
    _assert_response_schema(result)


def test_invalid_worked_solution_degrades_to_verified_short_answer(monkeypatch):
    monkeypatch.setattr(
        user_agent,
        "verify_solution",
        lambda *args, **kwargs: {"status": "passed", "severity": "none", "issues": []},
    )
    result = ReasoningAgent(FakeClient("最终答案：命题成立")).solve("证明该命题。", {})

    assert result["final_response"] == "命题成立"
    _assert_response_schema(result)


def test_worked_solution_appends_missing_final_conclusion():
    response = _compose_final_response(
        problem="推导所需关系。",
        response_mode=WORKED_SOLUTION,
        solution="从定义出发逐项展开，再合并同类项即可得到所需关系。",
        extracted_answer="x=y+1",
        verification={"status": "passed", "severity": "none", "issues": []},
    )

    assert response.endswith("最终结论：x=y+1")


def test_invalid_solution_and_answer_use_fallback():
    result = ReasoningAgent(FakeClient(["没有明确答案。", "仍然没有明确答案。"])).solve("证明该命题。", {})

    assert result["final_response"] == FALLBACK_RESPONSE
    _assert_response_schema(result)


def test_worked_local_tool_uses_its_existing_solution(monkeypatch):
    monkeypatch.setattr(
        user_agent,
        "solve_divisibility_subset_problem",
        lambda problem: {
            "tool_name": "fake_exact_tool",
            "final_answer": "2",
            "details": {"method": "existing exact derivation"},
            "solution": "因为相邻两项配对后总和相同，所以总和为 2。因此结论成立。",
        },
    )
    result = ReasoningAgent(FakeClient("不应调用模型")).solve("证明该配对结论并给出结果。", {})

    assert "因为相邻两项配对" in result["final_response"]
    assert "因此结论成立" in result["final_response"]
    _assert_response_schema(result)


def test_metadata_denylist_normalizes_case_spaces_and_hyphens():
    blocked = {
        "answer": "blocked-1",
        "Answer": "blocked-2",
        "expected-answer": "blocked-3",
        "Ground Truth": "blocked-4",
        "ground_truth": "blocked-5",
        "gold": "blocked-6",
        "reference": "blocked-7",
        "official_answer": "blocked-8",
        "label": "blocked-9",
        "target": "blocked-10",
    }
    safe = _safe_metadata(
        {**blocked, "idx": 7, "domain": "algebra", "subject": "math", "source": "local", "target_score": 0.5}
    )

    assert safe == {"idx": 7, "domain": "algebra", "subject": "math", "source": "local", "target_score": 0.5}


def test_blocked_metadata_does_not_affect_prompt_trace_or_response():
    blocked = {
        "answer": "secret_answer_value",
        "Answer": "secret_case_value",
        "expected-answer": "secret_expected_value",
        "Ground Truth": "secret_ground_value",
        "ground_truth": "secret_ground_snake_value",
        "gold": "secret_gold_value",
        "reference": "secret_reference_value",
        "official_answer": "secret_official_value",
        "label": "secret_label_value",
        "target": "secret_target_value",
    }
    client = FakeClient("计算得到 24+16=40。\n最终答案：40")
    result = ReasoningAgent(client).solve("计算 24+16。", {**blocked, "idx": 1, "source": "safe_source"})

    assert result["final_response"] == "40"
    serialized = json.dumps({"calls": client.calls, "result": result}, ensure_ascii=False)
    for value in blocked.values():
        assert value not in serialized
    _assert_response_schema(result)
