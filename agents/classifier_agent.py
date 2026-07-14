import re


ADVANCED_DOMAINS = (
    "numerical_analysis",
    "measure_theory",
    "differential_geometry",
    "abstract_algebra",
    "stochastic_processes",
    "statistics",
    "functional_analysis",
    "linear_regression",
    "mathematical_analysis",
)

DOMAINS = (
    "calculus",
    "algebra",
    "geometry",
    "probability",
    "ode_pde",
    "discrete_math",
    "combinatorics",
    "graph_theory",
    "proof",
    "real_analysis",
    "topology",
    "complex_analysis",
    "optimization",
    "linear_algebra",
    "number_theory",
    *ADVANCED_DOMAINS,
    "unknown",
)

SOLVER_KEYS = (
    "algebra",
    "calculus",
    "probability",
    "ode_pde",
    "proof",
    "discrete",
    "complex_analysis",
    "geometry",
    "linear_algebra",
    "number_theory",
    "optimization",
    *ADVANCED_DOMAINS,
    "general",
)

DOMAIN_TO_SOLVER_KEY = {
    "algebra": "algebra",
    "calculus": "calculus",
    "probability": "probability",
    "ode_pde": "ode_pde",
    "proof": "proof",
    "geometry": "geometry",
    "complex_analysis": "complex_analysis",
    "optimization": "optimization",
    "linear_algebra": "linear_algebra",
    "number_theory": "number_theory",
    "real_analysis": "proof",
    "topology": "proof",
    "discrete_math": "discrete",
    "combinatorics": "discrete",
    "graph_theory": "discrete",
    **{domain: domain for domain in ADVANCED_DOMAINS},
}

DOMAIN_HINT_ALIASES = {
    domain: domain
    for domain in DOMAINS
    if domain != "unknown"
}
DOMAIN_HINT_ALIASES.update(
    {
        "numerical analysis": "numerical_analysis",
        "数值分析": "numerical_analysis",
        "measure theory": "measure_theory",
        "测度论": "measure_theory",
        "differential geometry": "differential_geometry",
        "微分几何": "differential_geometry",
        "abstract algebra": "abstract_algebra",
        "抽象代数": "abstract_algebra",
        "stochastic processes": "stochastic_processes",
        "stochastic process": "stochastic_processes",
        "随机过程": "stochastic_processes",
        "统计学": "statistics",
        "functional analysis": "functional_analysis",
        "泛函分析": "functional_analysis",
        "linear regression": "linear_regression",
        "线性回归": "linear_regression",
        "mathematical analysis": "mathematical_analysis",
        "数学分析": "mathematical_analysis",
        "linear algebra": "linear_algebra",
        "线性代数": "linear_algebra",
        "number theory": "number_theory",
        "数论": "number_theory",
        "real analysis": "real_analysis",
        "complex analysis": "complex_analysis",
        "graph theory": "graph_theory",
        "discrete mathematics": "discrete_math",
        "operations research": "optimization",
    }
)


def solver_key_for_domain(domain: str) -> str:
    return DOMAIN_TO_SOLVER_KEY.get(str(domain or "").strip().lower(), "general")


def domain_from_hint(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if not text or text == "unknown":
        return None
    normalized = re.sub(r"[\s-]+", "_", text)
    if normalized in DOMAIN_HINT_ALIASES:
        return DOMAIN_HINT_ALIASES[normalized]
    if text in DOMAIN_HINT_ALIASES:
        return DOMAIN_HINT_ALIASES[text]
    parts = [part.strip() for part in re.split(r"->|/|:|>", text) if part.strip()]
    for part in reversed(parts):
        normalized_part = re.sub(r"[\s-]+", "_", part)
        if normalized_part in DOMAIN_HINT_ALIASES:
            return DOMAIN_HINT_ALIASES[normalized_part]
        if part in DOMAIN_HINT_ALIASES:
            return DOMAIN_HINT_ALIASES[part]
    return None


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _contains_word(text: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def _advanced_domain_from_text(text: str) -> tuple[str, str] | None:
    if _contains_any(
        text,
        (
            "数值分析",
            "数值迭代",
            "误差界",
            "截断误差",
            "收敛阶",
            "牛顿迭代",
            "newton iteration",
            "newton's iteration",
            "numerical method",
            "numerical analysis",
            "truncation error",
            "convergence order",
            "error bound of the algorithm",
        ),
    ):
        return "numerical_analysis", "题面包含数值方法、误差控制或收敛阶的强信号。"

    if _contains_any(
        text,
        (
            "测度论",
            "测度",
            "可测函数",
            "勒贝格",
            "几乎处处",
            "σ-代数",
            "σ代数",
            "measure theory",
            "lebesgue",
            "almost everywhere",
            "sigma algebra",
            "sigma-algebra",
            "measurable function",
            "a.e.",
        ),
    ) or _contains_word(text, ("measurable",)):
        return "measure_theory", "题面包含测度、可测性或 Lebesgue 理论的强信号。"

    if _contains_any(
        text,
        (
            "微分几何",
            "流形",
            "切空间",
            "测地线",
            "曲率张量",
            "黎曼度量",
            "differential geometry",
            "tangent space",
            "curvature tensor",
        ),
    ) or _contains_word(text, ("manifold", "geodesic", "riemannian")):
        return "differential_geometry", "题面包含流形、切空间、测地线或曲率张量的强信号。"

    if _contains_any(
        text,
        (
            "抽象代数",
            "群同态",
            "群作用",
            "正规子群",
            "阿贝尔群",
            "有限群",
            "环同态",
            "环的理想",
            "素理想",
            "极大理想",
            "商环",
            "域扩张",
            "group homomorphism",
            "group action",
            "normal subgroup",
            "quotient group",
            "quotient ring",
            "ring homomorphism",
            "polynomial ring",
            "field extension",
            "abstract algebra",
            "galois",
        ),
    ):
        return "abstract_algebra", "题面包含群、环、理想、域扩张或 Galois 理论的强信号。"

    if _contains_any(
        text,
        (
            "随机过程",
            "马尔可夫链",
            "布朗运动",
            "停时",
            "平稳过程",
            "泊松过程",
            "stochastic process",
            "markov chain",
            "brownian motion",
            "stopping time",
            "stationary process",
            "poisson process",
        ),
    ):
        return "stochastic_processes", "题面包含随机过程、马尔可夫链、布朗运动或停时的强信号。"

    if _contains_any(
        text,
        (
            "线性回归",
            "最小二乘估计",
            "回归系数",
            "残差平方和",
            "linear regression",
            "least squares estimator",
            "regression coefficient",
            "residual sum of squares",
            "ordinary least squares",
        ),
    ) or _contains_word(text, ("ols",)):
        return "linear_regression", "题面包含线性回归、最小二乘估计或回归诊断的强信号。"

    if _contains_any(
        text,
        (
            "统计推断",
            "极大似然",
            "最大似然",
            "置信区间",
            "假设检验",
            "充分统计量",
            "无偏估计",
            "statistical inference",
            "maximum likelihood",
            "confidence interval",
            "hypothesis test",
            "hypothesis testing",
            "sufficient statistic",
            "unbiased estimator",
        ),
    ) or _contains_word(text, ("statistics",)):
        return "statistics", "题面包含统计推断、估计、区间或假设检验的强信号。"

    functional_markers = (
        "泛函分析",
        "有界线性算子",
        "弱收敛",
        "紧算子",
        "bounded linear operator",
        "bounded operator",
        "weak convergence",
        "compact operator",
        "functional analysis",
    )
    spectral_context = _contains_any(text, ("谱定理", "spectral theorem")) and _contains_any(
        text, ("算子", "operator", "hilbert", "banach")
    )
    if _contains_any(text, functional_markers) or _contains_word(text, ("banach", "hilbert")) or spectral_context:
        return "functional_analysis", "题面包含 Banach/Hilbert 空间或算子理论的强信号。"

    if _contains_any(
        text,
        (
            "数学分析",
            "一致收敛",
            "逐点收敛",
            "函数项级数",
            "交换极限",
            "mathematical analysis",
            "uniform convergence",
            "pointwise convergence",
            "series of functions",
            "interchange of limits",
            "exchange the limit",
        ),
    ):
        return "mathematical_analysis", "题面包含函数列、函数项级数或收敛方式的强信号。"
    return None


def _looks_like_extremal_discrete_problem(text: str) -> bool:
    subset_markers = ("k-element subset", "every k-element subset", "subset")
    structure_markers = (
        "contains two distinct elements",
        "two distinct elements",
        "pair of elements",
        "divides",
        "positive integer",
        "integer",
        "{1,2,...",
        "{1, 2, ...",
        "independent set",
        "coloring",
        "tournament",
        "choose",
    )
    return any(marker in text for marker in subset_markers) and any(marker in text for marker in structure_markers)


def classify_problem(problem: str) -> dict:
    text = str(problem or "").lower()
    advanced = _advanced_domain_from_text(text)
    if advanced is not None:
        domain, reason = advanced
    elif any(token in text for token in ("probability", "概率", "随机", "骰子", "硬币", "红球", "蓝球")):
        domain = "probability"
        reason = "题目涉及随机试验或概率计算。"
    elif _looks_like_extremal_discrete_problem(text):
        domain = "combinatorics"
        reason = "题目是极值集合、组合图论或离散结构问题。"
    elif any(token in text for token in ("pde", "ode", "ordinary differential equation", "热方程", "偏微分", "微分方程", "u_t", "u_xx")):
        domain = "ode_pde"
        reason = "题目涉及常微分方程或偏微分方程。"
    elif any(token in text for token in ("combinatorics", "组合", "选法", "排列", "组合数")):
        domain = "discrete_math"
        reason = "题目涉及组合计数或排列组合。"
    elif any(token in text for token in ("graph_theory", "图论", "顶点", "图的边", "vertices and edges")):
        domain = "discrete_math"
        reason = "题目涉及图论或离散结构。"
    elif any(token in text for token in ("discrete", "离散", "命题", "逆否")):
        domain = "discrete_math"
        reason = "题目涉及离散数学、逻辑或组合计数。"
    elif any(token in text for token in ("triangle", "circle", "geometry", "三角形", "圆", "几何", "面积", "角")):
        domain = "geometry"
        reason = "题目包含几何对象或空间关系。"
    elif any(
        token in text
        for token in ("complex_analysis", "complex", "复分析", "复数", "留数", "解析函数", "cauchy", "laurent", "解析")
    ):
        domain = "complex_analysis"
        reason = "题目涉及复数或复分析术语。"
    elif any(token in text for token in ("prove", "proof", "证明", "得证")):
        domain = "proof"
        reason = "题目要求证明命题。"
    elif any(token in text for token in ("limit", "derivative", "integral", "极限", "导数", "积分", "微分")):
        domain = "calculus"
        reason = "题目涉及极限、导数、积分或微积分概念。"
    elif any(
        token in text
        for token in (
            "linear_algebra",
            "linear algebra",
            "线性代数",
            "矩阵",
            "matrix",
            "行列式",
            "determinant",
            "特征值",
            "eigenvalue",
            "向量空间",
            "vector space",
            "秩",
        )
    ):
        domain = "linear_algebra"
        reason = "题目涉及矩阵、线性空间或线性代数计算。"
    elif any(
        token in text
        for token in (
            "number_theory",
            "数论",
            "素数",
            "同余",
            "gcd",
            "最大公约数",
            "整除",
            "multiplicative order",
            "modulo",
        )
    ):
        domain = "number_theory"
        reason = "题目涉及数论、整除、同余或素数。"
    elif any(
        token in text
        for token in (
            "linear programming",
            "objective function",
            "subject to",
            "optimization",
            "operations_research",
            "optimize",
            "minimize",
            "maximize",
            "运筹",
            "线性规划",
            "最优化",
            "约束优化",
            "优化目标",
        )
    ):
        domain = "optimization"
        reason = "题目包含目标函数、约束或明确的最优化强信号。"
    elif any(token in text for token in ("equation", "solve", "方程", "二次", "代数", "x^2")):
        domain = "algebra"
        reason = "题目主要是方程求解或代数运算。"
    elif any(
        token in text
        for token in (
            "optimization",
            "operations_research",
            "optimize",
            "maximum",
            "minimum",
            "运筹",
            "线性规划",
            "最优化",
            "约束优化",
            "最值",
            "优化",
        )
    ):
        domain = "optimization"
        reason = "题目关注最优化或极值。"
    elif any(token in text for token in ("topology", "拓扑", "连续映射")):
        domain = "topology"
        reason = "题目涉及拓扑相关概念。"
    else:
        domain = "unknown"
        reason = "题目缺少足够明显的领域关键词。"

    return {"domain": domain, "solver_key": solver_key_for_domain(domain), "reason": reason}
