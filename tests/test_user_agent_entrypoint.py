import json
from types import SimpleNamespace

import user_agent
from user_agent import (
    ReasoningAgent,
    _build_correction_prompt,
    _is_meaningful_final_answer,
    _safe_metadata,
)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, list):
            index = min(len(self.calls) - 1, len(self.response) - 1)
            return self.response[index]
        return self.response


class ErrorClient:
    def chat(self, **kwargs):
        raise RuntimeError("network token should not leak")


EXTREMAL_SUBSET_PROBLEM = (
    "Find the smallest positive integer K such that every K-element subset of "
    "{1,2,...,50} contains two distinct elements a,b such that a+b divides ab."
)


def _steps(result):
    return [item["step"] for item in result["trace"]]


def test_can_import_and_initialize_reasoning_agent():
    agent = ReasoningAgent(client=FakeClient("最终答案：2"))

    assert agent.client is not None


def test_solve_calls_chat_and_returns_dict_with_trace():
    client = FakeClient("解：1+1=2\n最终答案：2")
    agent = ReasoningAgent(client=client)

    result = agent.solve("1+1=?", {})

    assert client.calls
    assert isinstance(result, dict)
    assert isinstance(result["final_response"], str)
    assert result["final_response"]
    assert isinstance(result["trace"], list)


def test_trace_contains_required_steps():
    result = ReasoningAgent(FakeClient("f'(x)=2x，所以 f'(3)=6。\n最终答案：6")).solve(
        "求 f(x)=x^2 在 x=3 处的导数值。",
        {},
    )

    steps = _steps(result)

    for step in ("classify", "plan", "model_call", "extract", "verify", "finalize"):
        assert step in steps


def test_string_response_is_supported():
    result = ReasoningAgent(FakeClient("最终答案：2")).solve("1+1=?", {})

    assert "2" in result["final_response"]


def test_dict_response_is_supported():
    result = ReasoningAgent(FakeClient({"content": "最终答案：3"})).solve("1+2=?", {})

    assert "3" in result["final_response"]


def test_object_content_response_is_supported():
    result = ReasoningAgent(FakeClient(SimpleNamespace(content="最终答案：4"))).solve("2+2=?", {})

    assert "4" in result["final_response"]


def test_openai_like_response_is_supported():
    response = {"choices": [{"message": {"content": "最终答案：5"}}]}
    result = ReasoningAgent(FakeClient(response)).solve("2+3=?", {})

    assert "5" in result["final_response"]


def test_metadata_answer_is_not_used_as_final_response():
    client = FakeClient("解：1+1=2\n最终答案：2")
    result = ReasoningAgent(client).solve("1+1=?", {"answer": "999"})

    assert "2" in result["final_response"]
    assert result["final_response"] != "999"


def test_model_call_exception_returns_nonempty_response_and_error_trace():
    result = ReasoningAgent(ErrorClient()).solve("1+1=?", {})

    assert isinstance(result, dict)
    assert result["final_response"]
    assert any(item["step"] == "model_call" and "error" in item["content"] for item in result["trace"])


def test_no_retry_when_first_verification_passes():
    client = FakeClient("解：1+1=2\n最终答案：2")
    result = ReasoningAgent(client).solve("1+1=?", {})

    assert len(client.calls) == 1
    assert result["final_response"]
    assert any(item["step"] == "retry_decision" and "retry_used=False" in item["content"] for item in result["trace"])


def test_empty_final_answer_triggers_retry():
    client = FakeClient(["这是一段没有明确答案的说明。", "修正后：1+1=2\n最终答案：2"])
    result = ReasoningAgent(client).solve("1+1=?", {})

    assert len(client.calls) == 2
    assert "2" in result["final_response"]
    assert "retry_model_call" in _steps(result)


def test_verifier_failed_triggers_retry(monkeypatch):
    calls = {"count": 0}

    def fake_verify(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "failed", "severity": "high", "issues": [{"code": "bad"}]}
        return {"status": "passed", "severity": "none", "issues": []}

    monkeypatch.setattr(user_agent, "verify_solution", fake_verify)
    client = FakeClient(["最终答案：1", "最终答案：2"])

    result = ReasoningAgent(client).solve("1+1=?", {})

    assert len(client.calls) == 2
    assert "2" in result["final_response"]


def test_second_attempt_without_answer_falls_back_to_nonempty_response():
    client = FakeClient(["没有明确答案。", "仍然没有明确答案。"])

    result = ReasoningAgent(client).solve("1+1=?", {})

    assert len(client.calls) == 2
    assert result["final_response"]


def test_correction_prompt_filters_metadata_answer():
    prompt = _build_correction_prompt(
        "1+1=?",
        {
            "idx": 0,
            "answer": "999",
            "expected_answer": "888",
            "solution": "secret solution",
            "subject": "math",
        },
        "第一次解答",
        "",
        {"status": "failed", "issues": [], "suggestion": "retry"},
        solver_key="algebra",
    )

    assert "999" not in prompt
    assert "888" not in prompt
    assert "secret solution" not in prompt


def test_metadata_answer_not_leaked_to_retry_prompt():
    client = FakeClient(["没有明确答案。", "最终答案：2"])
    result = ReasoningAgent(client).solve("1+1=?", {"idx": 0, "answer": "999"})

    assert len(client.calls) == 2
    retry_prompt = client.calls[1]["messages"][0]["content"]
    assert "999" not in retry_prompt
    assert result["final_response"] != "999"


def test_trace_is_json_serializable_with_retry():
    result = ReasoningAgent(FakeClient(["没有明确答案。", "最终答案：2"])).solve("1+1=?", {})

    json.dumps(result, ensure_ascii=False)


def test_retry_model_call_exception_is_captured():
    class RetryErrorClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return "没有明确答案。"
            raise RuntimeError("Authorization token should not leak")

    client = RetryErrorClient()
    result = ReasoningAgent(client).solve("1+1=?", {})

    assert len(client.calls) == 2
    assert result["final_response"]
    assert any(item["step"] == "retry_model_call" and "error" in item["content"] for item in result["trace"])


def test_safe_metadata_filters_answer_fields():
    safe = _safe_metadata({"answer": "999", "gold_answer": "888", "solution": "secret solution", "idx": 1})

    assert safe == {"idx": 1}


def test_metadata_domain_overrides_problem_classification_and_filters_answer():
    client = FakeClient("推理过程\n最终答案：26")
    result = ReasoningAgent(client).solve("Find the optimum.", {"domain": "optimization", "answer": "999"})

    assert result["final_response"] != "999"
    assert "26" in result["final_response"]
    classify_trace = next(item["content"] for item in result["trace"] if item["step"] == "classify")
    assert "domain=optimization" in classify_trace
    assert "solver_key=optimization" in classify_trace


def test_combinatorics_metadata_routes_to_discrete():
    result = ReasoningAgent(FakeClient("最终答案：10")).solve("How many choices?", {"domain": "combinatorics"})

    classify_trace = next(item["content"] for item in result["trace"] if item["step"] == "classify")
    assert "domain=combinatorics" in classify_trace
    assert "solver_key=discrete" in classify_trace


def test_punctuation_fragment_final_answer_triggers_retry():
    client = FakeClient(['最终答案：".', "修正后\n最终答案：26"])

    result = ReasoningAgent(client).solve("求答案", {})

    assert len(client.calls) == 2
    assert "26" in result["final_response"]
    assert "retry_model_call" in _steps(result)
    retry_trace = next(item["content"] for item in result["trace"] if item["step"] == "retry_decision")
    assert "not_meaningful_final_answer" in retry_trace


def test_two_punctuation_fragment_answers_fall_back():
    result = ReasoningAgent(FakeClient(['最终答案：".', "最终答案：'."])).solve("求答案", {})

    assert result["final_response"] == user_agent.FALLBACK_RESPONSE


def test_number_two_is_meaningful_and_does_not_retry():
    client = FakeClient("最终答案：2")

    result = ReasoningAgent(client).solve("1+1=?", {})

    assert _is_meaningful_final_answer("2") is True
    assert len(client.calls) == 1
    assert "2" in result["final_response"]


def test_placeholder_final_answer_is_not_meaningful():
    assert _is_meaningful_final_answer('<答案>", then concise reasoning.') is False
    assert _is_meaningful_final_answer("<answer>") is False
    assert _is_meaningful_final_answer("then concise reasoning") is False


def test_metadata_answer_type_number_overrides_extracted_set():
    client = FakeClient("最终答案：x = 2, x = 3")

    result = ReasoningAgent(client).solve("Find the number.", {"answer_type": "number"})

    assert len(client.calls) == 2
    assert result["final_response"] == user_agent.FALLBACK_RESPONSE
    extract_trace = next(item["content"] for item in result["trace"] if item["step"] == "extract")
    verify_trace = next(item["content"] for item in result["trace"] if item["step"] == "verify")
    assert "extracted_answer_type=set" in extract_trace
    assert "expected_answer_type=number" in extract_trace
    assert "expected_answer_type=number" in verify_trace


def test_retry_still_enforces_metadata_answer_type_number():
    client = FakeClient(['最终答案：".', "最终答案：x = 2, x = 3"])

    result = ReasoningAgent(client).solve("Find the number.", {"answer_type": "number"})

    assert len(client.calls) == 2
    assert result["final_response"] == user_agent.FALLBACK_RESPONSE
    retry_extract_trace = next(item["content"] for item in result["trace"] if item["step"] == "retry_extract")
    retry_verify_trace = next(item["content"] for item in result["trace"] if item["step"] == "retry_verify")
    assert "extracted_answer_type=set" in retry_extract_trace
    assert "expected_answer_type=number" in retry_extract_trace
    assert "status=failed" in retry_verify_trace
    assert "expected_answer_type=number" in retry_verify_trace


def test_metadata_answer_type_number_positive_answer_passes_without_retry():
    client = FakeClient("最终答案：26")

    result = ReasoningAgent(client).solve("Find the number.", {"answer_type": "number"})

    assert len(client.calls) == 1
    assert result["final_response"] == "26"
    verify_trace = next(item["content"] for item in result["trace"] if item["step"] == "verify")
    assert "status=passed" in verify_trace
    assert "expected_answer_type=number" in verify_trace


def test_two_placeholder_number_answers_fall_back():
    client = FakeClient(
        [
            '最终答案：<答案>", then concise reasoning.',
            '最终答案：<答案>", then concise reasoning.',
        ]
    )

    result = ReasoningAgent(client).solve("Find the number.", {"answer_type": "number"})

    assert len(client.calls) == 2
    assert result["final_response"] == user_agent.FALLBACK_RESPONSE
    assert any("meaningful_final=False" in item["content"] for item in result["trace"] if item["step"] in {"extract", "retry_extract"})


def test_retry_uncertain_verification_is_not_finalized(monkeypatch):
    calls = {"count": 0}

    def fake_verify(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "failed", "severity": "high", "issues": [{"code": "bad"}]}
        return {"status": "uncertain", "severity": "medium", "issues": [{"code": "still_uncertain"}]}

    monkeypatch.setattr(user_agent, "verify_solution", fake_verify)
    client = FakeClient(["最终答案：1", "最终答案：26"])

    result = ReasoningAgent(client).solve("Find the number.", {"answer_type": "number"})

    assert len(client.calls) == 2
    assert result["final_response"] == user_agent.FALLBACK_RESPONSE


def test_metadata_answer_type_expression_allows_equation_list():
    client = FakeClient("最终答案：x = 2, x = 3")

    result = ReasoningAgent(client).solve("Solve the equation.", {"answer_type": "expression"})

    assert len(client.calls) == 1
    assert result["final_response"] == "x = 2, x = 3"
    verify_trace = next(item["content"] for item in result["trace"] if item["step"] == "verify")
    assert "status=passed" in verify_trace
    assert "expected_answer_type=expression" in verify_trace


def test_optimization_metadata_extremal_subset_routes_to_discrete():
    result = ReasoningAgent(FakeClient("最终答案：26")).solve(
        EXTREMAL_SUBSET_PROBLEM,
        {"domain": "optimization", "answer_type": "number"},
    )

    classify_trace = next(item["content"] for item in result["trace"] if item["step"] == "classify")
    assert "solver_key=optimization" not in classify_trace
    assert "solver_key=discrete" in classify_trace or "solver_key=number_theory" in classify_trace


def test_plain_optimization_metadata_still_routes_to_optimization():
    result = ReasoningAgent(FakeClient("最终答案：12")).solve(
        "Minimize f(x)=x^2 subject to x >= 3 and linear programming constraints.",
        {"domain": "optimization", "answer_type": "number"},
    )

    classify_trace = next(item["content"] for item in result["trace"] if item["step"] == "classify")
    assert "domain=optimization" in classify_trace
    assert "solver_key=optimization" in classify_trace


def test_number_prompt_requires_clear_final_answer_at_end():
    client = FakeClient("最终答案：26")

    ReasoningAgent(client).solve(
        "Find the smallest positive integer K.",
        {"answer_type": "number"},
    )

    prompt = client.calls[0]["messages"][0]["content"]
    assert "请先正确求解" in prompt
    assert "解答末尾" in prompt
    assert "最终答案：26" not in prompt
    assert "<答案>" not in prompt
    assert "<单个整数" not in prompt
    assert "<单个数值" not in prompt


def test_expression_prompt_requires_expression_not_bare_number():
    client = FakeClient("最终答案：x+1")

    ReasoningAgent(client).solve(
        "Find an expression for the length.",
        {"answer_type": "expression", "domain": "geometry", "solver_key": "geometry"},
    )

    prompt = client.calls[0]["messages"][0]["content"]
    assert "最终答案必须是表达式" in prompt
    assert "不要用单个数字作为占位答案" in prompt
    assert "适用时请使用题目中的变量" in prompt


def test_discrete_extremal_prompt_forbids_full_edge_enumeration():
    client = FakeClient("最终答案：26")
    prompt_only_problem = (
        "Find the smallest positive integer K such that every K-element subset of "
        "{1,2,...,50} contains two distinct elements with a specified graph property."
    )

    ReasoningAgent(client).solve(
        prompt_only_problem,
        {"domain": "combinatorics", "answer_type": "number"},
    )

    prompt = client.calls[0]["messages"][0]["content"]
    assert "不要完整枚举所有边" in prompt
    assert "邻接表" in prompt
    assert "结构分组" in prompt


def test_model_call_trace_contains_solution_summary():
    solution = "前置推理。" + ("中间步骤" * 80) + "\n最终答案：26"

    result = ReasoningAgent(FakeClient(solution)).solve("Find the number.", {"answer_type": "number"})

    model_trace = next(item["content"] for item in result["trace"] if item["step"] == "model_call")
    assert "status=success" in model_trace
    assert "solution_chars=" in model_trace
    assert "solution_head=" in model_trace
    assert "solution_tail=" in model_trace
    assert len(model_trace) < 1400


def test_retry_model_call_trace_contains_retry_solution_summary():
    client = FakeClient(["没有明确答案。", "修正推理。\n最终答案：26"])

    result = ReasoningAgent(client).solve("Find the number.", {"answer_type": "number"})

    retry_trace = next(item["content"] for item in result["trace"] if item["step"] == "retry_model_call")
    assert "status=success" in retry_trace
    assert "retry_solution_chars=" in retry_trace
    assert "retry_solution_head=" in retry_trace
    assert "retry_solution_tail=" in retry_trace


def test_extract_trace_contains_final_answer_summary():
    result = ReasoningAgent(FakeClient("最终答案：26")).solve("Find the number.", {"answer_type": "number"})

    extract_trace = next(item["content"] for item in result["trace"] if item["step"] == "extract")
    assert "extracted_final_answer='26'" in extract_trace
    assert "extracted_answer_type=number" in extract_trace
    assert "expected_answer_type=number" in extract_trace
    assert "meaningful_final=True" in extract_trace
    assert "final_answer_chars=2" in extract_trace


def test_correction_prompt_compresses_long_first_solution():
    long_solution = "HEAD" + (" 很长的推理" * 700) + "TAIL"
    prompt = _build_correction_prompt(
        "Find the number.",
        {"answer": "999", "expected_answer": "888", "solution": "hidden", "answer_type": "number"},
        long_solution,
        "x = 2, x = 3",
        {
            "status": "failed",
            "severity": "high",
            "issues": [{"code": "answer_type_mismatch", "message": "bad"}],
            "suggestion": "retry",
        },
        solver_key="optimization",
        domain="optimization",
        expected_answer_type="number",
    )

    assert len(prompt) < 3500
    assert len(prompt) < len(long_solution)
    assert "first_solution_head" in prompt
    assert "first_solution_tail" in prompt
    assert "999" not in prompt
    assert "888" not in prompt
    assert "hidden" not in prompt
    assert "单独的数值" in prompt
    assert "解答末尾" in prompt
    assert "最终答案：26" not in prompt
    assert "<答案>" not in prompt
    assert "<单个整数" not in prompt
    assert "<单个数值" not in prompt


def test_expression_correction_prompt_mentions_type_mismatch():
    prompt = _build_correction_prompt(
        "Find an expression for the length.",
        {"answer_type": "expression"},
        "推理过程。\n最终答案：2",
        "2",
        {
            "status": "uncertain",
            "severity": "medium",
            "issues": [{"code": "expression_without_math_markers"}],
            "suggestion": "retry",
        },
        solver_key="geometry",
        domain="geometry",
        expected_answer_type="expression",
    )

    assert "需要 expression 类型" in prompt
    assert "不要再次返回纯数字" in prompt
    assert "使用题目中的变量" in prompt


def test_correction_prompt_for_extremal_subset_says_do_not_continue_listing_edges():
    prompt = _build_correction_prompt(
        EXTREMAL_SUBSET_PROBLEM,
        {"answer_type": "number"},
        "已列出很多边，但没有最终答案。",
        "",
        {"status": "failed", "severity": "high", "issues": [], "suggestion": "retry"},
        solver_key="discrete",
        domain="combinatorics",
        expected_answer_type="number",
    )

    assert "不要继续列边" in prompt
    assert "末尾给出最终答案" in prompt
    assert "单个整数" in prompt


def test_trace_redacts_sensitive_model_output_and_is_json_serializable():
    result = ReasoningAgent(FakeClient("Authorization Bearer api_key token\n最终答案：2")).solve("1+1=?", {})

    serialized = json.dumps(result, ensure_ascii=False)
    assert "Authorization" not in serialized
    assert "Bearer" not in serialized
    assert "api_key" not in serialized
    assert "token" not in serialized


def test_solve_with_metadata_none_returns_nonempty_response():
    result = ReasoningAgent(FakeClient("最终答案：2")).solve("1+1=?", None)

    assert isinstance(result, dict)
    assert result["final_response"]
    assert isinstance(result["trace"], list)
    json.dumps(result, ensure_ascii=False)


def test_solve_with_empty_string_chat_response_returns_nonempty_response():
    result = ReasoningAgent(FakeClient("")).solve("1+1=?", {})

    assert isinstance(result, dict)
    assert result["final_response"]
    assert isinstance(result["trace"], list)
    json.dumps(result, ensure_ascii=False)
