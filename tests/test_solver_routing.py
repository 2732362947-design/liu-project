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


def test_classifier_routes_complex_analysis():
    result = classify_problem("complex_analysis: 用 Cauchy 积分公式计算解析函数积分。")

    assert result["domain"] == "complex_analysis"
    assert result["solver_key"] == "complex_analysis"


def test_classifier_routes_geometry():
    result = classify_problem("几何：已知三角形和圆，求面积与角。")

    assert result["domain"] == "geometry"
    assert result["solver_key"] == "geometry"


def test_classifier_routes_linear_algebra():
    result = classify_problem("线性代数：求矩阵的特征值和秩。")

    assert result["domain"] == "linear_algebra"
    assert result["solver_key"] == "linear_algebra"


def test_classifier_routes_number_theory():
    result = classify_problem("数论：判断素数并计算 gcd 最大公约数。")

    assert result["domain"] == "number_theory"
    assert result["solver_key"] == "number_theory"


def test_classifier_routes_optimization():
    result = classify_problem("operations_research: 求线性规划的最优化解。")

    assert result["domain"] == "optimization"
    assert result["solver_key"] == "optimization"


def test_classifier_routes_combinatorics_to_discrete():
    result = classify_problem("combinatorics: 从 5 个元素中选法有多少种组合数？")

    assert result["domain"] == "discrete_math"
    assert result["solver_key"] == "discrete"


def test_classifier_routes_graph_theory_to_discrete():
    result = classify_problem("graph_theory: 一个图有多少顶点和边？")

    assert result["domain"] == "discrete_math"
    assert result["solver_key"] == "discrete"


def test_classifier_unknown_falls_back_to_general():
    result = classify_problem("这是一道没有明显领域关键词的题。")

    assert result["domain"] == "unknown"
    assert result["solver_key"] == "general"


def test_build_solver_prompt_uses_selected_template():
    prompt = solver_agent.build_solver_prompt(
        "求 x^2 - 5x + 6 = 0 的根。",
        "algebra",
        ["整理方程。"],
        solver_key="algebra",
    )

    assert "algebra solver" in prompt
    assert "最终答案" in prompt


def test_build_solver_prompt_loads_new_templates():
    for solver_key in (
        "complex_analysis",
        "geometry",
        "linear_algebra",
        "number_theory",
        "optimization",
    ):
        prompt = solver_agent.build_solver_prompt(
            "测试题。",
            solver_key,
            ["分析条件。"],
            solver_key=solver_key,
        )

        assert f"{solver_key} solver" in prompt
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
