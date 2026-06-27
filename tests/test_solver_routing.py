from agents.classifier_agent import classify_problem
from agents import solver_agent


def test_classifier_outputs_solver_key_for_probability():
    result = classify_problem("一个袋子里有红球和蓝球，随机取一个球的概率是多少？")

    assert result["domain"] == "probability"
    assert result["solver_key"] == "probability"


def test_classifier_routes_pde_to_ode_pde_solver():
    result = classify_problem("pde: 写出一维热方程的标准形式。")

    assert result["domain"] == "ode_pde"
    assert result["solver_key"] == "ode_pde"


def test_classifier_routes_discrete_logic_to_discrete_solver():
    result = classify_problem("discrete_math: 命题 P->Q 的逆否命题是什么？")

    assert result["domain"] == "discrete_math"
    assert result["solver_key"] == "discrete"


def test_build_solver_prompt_uses_selected_template():
    prompt = solver_agent.build_solver_prompt(
        "求 x^2 - 5x + 6 = 0 的根。",
        "algebra",
        ["整理方程。"],
        solver_key="algebra",
    )

    assert "algebra solver" in prompt
    assert "最终答案" in prompt


def test_unknown_solver_key_falls_back_to_general_template():
    prompt = solver_agent.build_solver_prompt(
        "求解。",
        "unknown",
        ["提取条件。"],
        solver_key="missing",
    )

    assert "general solver" in prompt


def test_solve_problem_calls_intern_s1_with_routed_prompt(monkeypatch):
    captured = {}

    def fake_call(prompt):
        captured["prompt"] = prompt
        return "最终答案：10"

    monkeypatch.setattr(solver_agent, "call_intern_s1", fake_call)

    result = solver_agent.solve_problem(
        "从5个不同元素中任选2个，有多少种选法？",
        "discrete_math",
        ["使用组合数。"],
        solver_key="discrete",
    )

    assert result == "最终答案：10"
    assert "discrete solver" in captured["prompt"]
