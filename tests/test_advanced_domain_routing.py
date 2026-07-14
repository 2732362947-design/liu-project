from pathlib import Path

import pytest

from agents import solver_agent
from agents.classifier_agent import ADVANCED_DOMAINS, classify_problem, domain_from_hint, solver_key_for_domain
from user_agent import ReasoningAgent, _apply_metadata_domain


POSITIVE_CASES = {
    "numerical_analysis": (
        "数值分析中，用牛顿迭代求根并说明停止准则。",
        "分析该数值迭代的截断误差和误差界。",
        "求这个离散算法的收敛阶，并验证初值条件。",
        "Analyze the convergence order of Newton iteration for this root-finding problem.",
        "Estimate the truncation error of the proposed numerical method.",
    ),
    "measure_theory": (
        "在给定测度空间上证明该集合可测。",
        "判断这个可测函数是否几乎处处收敛。",
        "使用勒贝格积分计算并检查可积性。",
        "Show that this function is measurable with respect to the sigma algebra.",
        "Apply the Lebesgue dominated convergence theorem almost everywhere.",
    ),
    "differential_geometry": (
        "在微分几何中求该流形一点处的切空间。",
        "给定黎曼度量，写出对应测地线方程。",
        "计算该曲率张量并说明坐标变换性质。",
        "Compute the tangent space of this smooth manifold.",
        "Find the geodesic induced by the Riemannian metric and curvature tensor.",
    ),
    "abstract_algebra": (
        "抽象代数中证明这个群同态的核是正规子群。",
        "求多项式环的极大理想及对应商环。",
        "讨论该域扩张的 Galois 群。",
        "Determine the kernel of this group homomorphism and the quotient group.",
        "Find the degree of the field extension and describe its Galois group.",
    ),
    "stochastic_processes": (
        "求该随机过程对应马尔可夫链的转移概率。",
        "证明布朗运动首次到达时间是停时。",
        "判断这个泊松过程是否具有平稳增量。",
        "Compute the transition probabilities of the Markov chain.",
        "Show that this Brownian motion stopping time defines a stationary process.",
    ),
    "statistics": (
        "用极大似然进行统计推断并求参数估计。",
        "构造总体均值的置信区间并进行假设检验。",
        "证明该充分统计量给出的估计是无偏估计。",
        "Find the maximum likelihood estimator and a confidence interval.",
        "Carry out a hypothesis test using the given sufficient statistic.",
    ),
    "functional_analysis": (
        "在泛函分析中证明这个 Banach 空间是完备的。",
        "研究 Hilbert 空间上的有界线性算子。",
        "判断该紧算子序列是否弱收敛，并说明谱定理条件。",
        "Prove that the bounded operator on this Hilbert space is compact.",
        "Analyze weak convergence in a Banach space using a compact operator.",
    ),
    "linear_regression": (
        "在线性回归模型中求最小二乘估计。",
        "计算回归系数和残差平方和。",
        "使用 OLS 推导设计矩阵对应的正规方程。",
        "Derive the least squares estimator for this linear regression model.",
        "Compute the regression coefficient and residual sum of squares using OLS.",
    ),
    "mathematical_analysis": (
        "数学分析中判断该函数列是否一致收敛。",
        "比较这个函数项级数的逐点收敛与一致收敛。",
        "说明在什么条件下可以交换极限与积分。",
        "Determine whether the series of functions converges uniformly or pointwise.",
        "Justify the interchange of limits under uniform convergence.",
    ),
}


NEGATIVE_CASES = {
    "numerical_analysis": (
        "这个函数的实验测量值存在误差，请解释误差来源。",
        "求解方程 x^2-5x+6=0。",
    ),
    "measure_theory": (
        "计算函数在区间上的 Riemann 积分。",
        "测量三角形的边长并计算面积。",
    ),
    "differential_geometry": (
        "在平面解析几何中求圆与直线的交点。",
        "已知三角形两边和夹角，求面积。",
    ),
    "abstract_algebra": (
        "Find the multiplicative order of 2 modulo 7.",
        "求这个二阶常微分方程的通解。",
    ),
    "stochastic_processes": (
        "掷一枚公平骰子一次，点数为偶数的概率是多少？",
        "Two fair coins are tossed; compute the probability of two heads.",
    ),
    "statistics": (
        "计算 2、4、6、8 的普通算术平均数。",
        "从袋中随机取一个球，求取到红球的概率。",
    ),
    "functional_analysis": (
        "求一个三阶矩阵的特征值和行列式。",
        "Compute the rank of the matrix and solve the linear system.",
    ),
    "linear_regression": (
        "解一元一次方程 3x+2=11。",
        "求经过两个给定点的直线方程。",
    ),
    "mathematical_analysis": (
        "计算 f(x)=x^3 在 x=2 处的导数。",
        "Evaluate the Riemann integral of x squared on the unit interval.",
    ),
}


POSITIVE_PARAMS = [
    pytest.param(problem, domain, id=f"{domain}-{index}")
    for domain, problems in POSITIVE_CASES.items()
    for index, problem in enumerate(problems, start=1)
]
NEGATIVE_PARAMS = [
    pytest.param(problem, domain, id=f"not-{domain}-{index}")
    for domain, problems in NEGATIVE_CASES.items()
    for index, problem in enumerate(problems, start=1)
]


@pytest.mark.parametrize(("problem", "expected_domain"), POSITIVE_PARAMS)
def test_advanced_domain_positive_examples(problem, expected_domain):
    result = classify_problem(problem)

    assert result["domain"] == expected_domain
    assert result["solver_key"] == expected_domain


@pytest.mark.parametrize(("problem", "excluded_domain"), NEGATIVE_PARAMS)
def test_advanced_domain_confusing_negatives(problem, excluded_domain):
    assert classify_problem(problem)["domain"] != excluded_domain


def test_each_advanced_domain_has_required_language_coverage():
    assert set(POSITIVE_CASES) == set(ADVANCED_DOMAINS)
    assert set(NEGATIVE_CASES) == set(ADVANCED_DOMAINS)
    assert all(len(cases) == 5 for cases in POSITIVE_CASES.values())
    assert all(len(cases) >= 2 for cases in NEGATIVE_CASES.values())


def test_every_advanced_domain_has_valid_solver_key_and_template():
    for domain in ADVANCED_DOMAINS:
        solver_key = solver_key_for_domain(domain)
        template_path = solver_agent.SOLVER_TEMPLATE_DIR / f"{solver_key}.txt"

        assert solver_key in solver_agent.VALID_SOLVER_KEYS
        assert solver_agent.normalize_solver_key(None, domain) == solver_key
        assert template_path.is_file()
        assert f"{solver_key} solver" in solver_agent.load_solver_template(solver_key)


def test_solver_templates_are_loaded_relative_to_solver_module():
    expected_directory = Path(solver_agent.__file__).parent / "solvers"
    source = Path(solver_agent.__file__).read_text(encoding="utf-8")

    assert solver_agent.SOLVER_TEMPLATE_DIR == expected_directory
    assert 'Path(__file__).parent / "solvers"' in source
    assert "/home/" not in source


def test_advanced_templates_contain_shared_scoring_requirements():
    for solver_key in ADVANCED_DOMAINS:
        template = solver_agent.load_solver_template(solver_key)

        assert "先识别" in template
        assert "标准" in template and "记号" in template
        assert "必要推导" in template
        assert "条件" in template
        assert "清晰最终答案" in template
        assert "可独立判分" in template
        assert "不得伪造" in template
        assert "metadata.subject" in template
        assert "无关" in template
        assert not any(f"最终答案：{digit}" in template for digit in "0123456789")


def test_metadata_missing_keeps_problem_only_classification():
    classification = classify_problem("Use Newton iteration and determine its convergence order.")

    result = _apply_metadata_domain(classification, None)

    assert result["domain"] == "numerical_analysis"


@pytest.mark.parametrize("field", ("domain", "subject", "category"))
def test_metadata_fields_are_optional_weak_hints(field):
    classification = classify_problem("研究下列数学结构并给出结论。")

    result = _apply_metadata_domain(classification, {field: "Measure Theory"})

    assert result["domain"] == "measure_theory"
    assert result["solver_key"] == "measure_theory"


def test_metadata_hierarchical_subject_hint_is_supported():
    assert domain_from_hint("Mathematics -> Functional Analysis") == "functional_analysis"


def test_metadata_unknown_hint_is_ignored():
    result = _apply_metadata_domain(classify_problem("研究下列结构。"), {"subject": "unknown"})

    assert result["domain"] == "unknown"
    assert result["solver_key"] == "general"


def test_problem_strong_signal_wins_over_conflicting_metadata_subject():
    classification = classify_problem("Use Newton iteration and analyze its convergence order.")

    result = _apply_metadata_domain(
        classification,
        {"domain": "abstract_algebra", "subject": "statistics", "category": "measure_theory"},
    )

    assert result["domain"] == "numerical_analysis"
    assert result["solver_key"] == "numerical_analysis"


def test_reasoning_agent_uses_subject_only_for_ambiguous_problem():
    class RecordingClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return "由定义检查条件。\n最终答案：成立"

    client = RecordingClient()
    result = ReasoningAgent(client).solve("研究下列数学结构并判断结论。", {"subject": "Functional Analysis"})
    classify_trace = next(item["content"] for item in result["trace"] if item["step"] == "classify")
    prompt = client.calls[0]["messages"][0]["content"]

    assert "domain=functional_analysis" in classify_trace
    assert "solver_key=functional_analysis" in classify_trace
    assert "functional_analysis solver" in prompt


def test_idx_and_source_do_not_change_routing_or_final_response():
    class FakeClient:
        def chat(self, **kwargs):
            return "迭代误差满足二次估计。\n最终答案：二阶收敛"

    problem = "Analyze the convergence order of Newton iteration."
    first = ReasoningAgent(FakeClient()).solve(problem, {"idx": "one", "source": "source_a"})
    second = ReasoningAgent(FakeClient()).solve(problem, {"idx": "two", "source": "source_b"})

    assert first["final_response"] == second["final_response"]
    assert next(item["content"] for item in first["trace"] if item["step"] == "classify") == next(
        item["content"] for item in second["trace"] if item["step"] == "classify"
    )


def test_reasoning_agent_routes_without_metadata_and_preserves_interface():
    class FakeClient:
        def chat(self, **kwargs):
            return "迭代误差逐步减小。\n最终答案：二阶收敛"

    result = ReasoningAgent(FakeClient()).solve("分析 Newton iteration 的 convergence order。", None)
    classify_trace = next(item["content"] for item in result["trace"] if item["step"] == "classify")

    assert "domain=numerical_analysis" in classify_trace
    assert "solver_key=numerical_analysis" in classify_trace
    assert isinstance(result["final_response"], str) and result["final_response"].strip()
