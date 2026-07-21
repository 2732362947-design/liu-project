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


ROUTING_PRIORITY = (
    *ADVANCED_DOMAINS,
    "number_theory",
    "combinatorics",
    "probability",
    "graph_theory",
    "algebra",
    "geometry",
    "calculus",
    "linear_algebra",
    "complex_analysis",
    "ode_pde",
    "optimization",
    "real_analysis",
    "topology",
    "proof",
    "discrete_math",
)


def _score_signal(
    scores: dict[str, int],
    categories: dict[str, set[str]],
    domain: str,
    category: str,
    weight: int,
    condition: bool,
) -> None:
    if not condition:
        return
    scores[domain] += weight
    categories[domain].add(category)


def _phrase(text: str, *markers: str) -> bool:
    return any(marker in text for marker in markers)


def _word(text: str, *markers: str) -> bool:
    return any(re.search(rf"\b{re.escape(marker)}\b", text) for marker in markers)


def classify_problem(problem: str) -> dict:
    text = str(problem or "").lower()
    scores = {domain: 0 for domain in ROUTING_PRIORITY}
    categories = {domain: set() for domain in ROUTING_PRIORITY}

    advanced = _advanced_domain_from_text(text)
    if advanced is not None:
        _score_signal(scores, categories, advanced[0], "advanced_structure", 20, True)

    # Explicit labels remain strong, but are combined with all other evidence.
    explicit_markers = {
        "calculus": ("calculus", "微积分"),
        "algebra": ("algebra", "代数"),
        "geometry": ("geometry", "几何"),
        "probability": ("probability", "概率"),
        "number_theory": ("number_theory", "number theory", "数论"),
        "combinatorics": ("combinatorics", "组合数学"),
        "graph_theory": ("graph_theory", "graph theory", "图论"),
        "linear_algebra": ("linear_algebra", "linear algebra", "线性代数"),
        "complex_analysis": ("complex_analysis", "complex analysis", "复分析"),
        "optimization": ("optimization", "operations_research", "最优化", "运筹"),
        "topology": ("topology", "拓扑"),
        "discrete_math": ("discrete_math", "discrete mathematics", "离散数学"),
    }
    for signal_domain, markers in explicit_markers.items():
        _score_signal(scores, categories, signal_domain, "explicit_domain", 8, _phrase(text, *markers))

    _score_signal(
        scores,
        categories,
        "ode_pde",
        "differential_equation_structure",
        7,
        _phrase(
            text,
            "ordinary differential equation",
            "partial differential equation",
            "微分方程",
            "偏微分",
            "热方程",
            "u_t",
            "u_xx",
        )
        or _word(text, "ode", "pde"),
    )
    _score_signal(
        scores,
        categories,
        "complex_analysis",
        "complex_structure",
        5,
        _phrase(text, "复数", "留数", "解析函数", "cauchy", "laurent") or _word(text, "complex"),
    )
    _score_signal(
        scores,
        categories,
        "linear_algebra",
        "linear_algebra_structure",
        5,
        _phrase(
            text,
            "matrix",
            "矩阵",
            "determinant",
            "行列式",
            "eigenvalue",
            "特征值",
            "vector space",
            "向量空间",
        )
        or _word(text, "rank"),
    )

    integral_coefficient_context = _phrase(
        text,
        "integral coefficients",
        "integer coefficients",
        "integral polynomial",
        "integer polynomial",
    )
    polynomial_context = _phrase(text, "polynomial", "多项式")
    _score_signal(
        scores,
        categories,
        "algebra",
        "polynomial_structure",
        6,
        _phrase(
            text,
            "monic irreducible polynomial",
            "polynomial factorization",
            "integral coefficients",
            "integer coefficients",
            "functional equation",
            "system of equations",
        ),
    )
    _score_signal(scores, categories, "algebra", "polynomial", 4, polynomial_context)
    _score_signal(
        scores,
        categories,
        "algebra",
        "algebraic_form",
        3,
        _phrase(
            text,
            "rational expression",
            "coefficient",
            "factorization",
            "factorisation",
            "floor function",
            "algebraic expression",
            "integer-coordinate",
            "coordinates that are both integers",
        )
        or _phrase(text, r"\lfloor"),
    )
    _score_signal(
        scores,
        categories,
        "algebra",
        "equation_target",
        3,
        _phrase(text, "solve for", "system of equations", "what is the value of")
        or _word(text, "equation", "solve"),
    )
    _score_signal(
        scores,
        categories,
        "algebra",
        "sequence_structure",
        4,
        _phrase(text, "sequence with an algebraic pattern", "following sequence", "let $a_n", "\\left(a_{n}\\right)"),
    )
    _score_signal(
        scores,
        categories,
        "algebra",
        "algebraic_extremum",
        4,
        _phrase(text, "minimum of an algebraic expression", "maximum of an algebraic expression", "maximum possible value", "smallest possible value"),
    )

    _score_signal(
        scores,
        categories,
        "number_theory",
        "divisibility_structure",
        6,
        _phrase(
            text,
            "largest integer that divides",
            "largest positive integer that divides",
            "greatest common divisor",
            "relatively prime",
            "integer-valued polynomial",
            "p-adic valuation",
        )
        or _phrase(text, r"\gcd", r"\pmod", r"\nu_p"),
    )
    _score_signal(
        scores,
        categories,
        "number_theory",
        "arithmetic_structure",
        4,
        _phrase(
            text,
            "divisible",
            "prime factor",
            "not necessarily distinct prime numbers",
            "coprime",
            "lowest terms",
            "repeating decimal",
            "repeats consecutively",
            "squarefree",
            "digit sum",
            "sum of the digits",
            "base representation",
            "multiplicative order",
        )
        or _word(text, "gcd", "remainder", "congruence", "modulo"),
    )
    _score_signal(
        scores,
        categories,
        "number_theory",
        "divides_relation",
        3,
        _word(text, "divides", "divisible") or _phrase(text, r"\mid", r"\nmid"),
    )

    arrangement_context = (
        _word(text, "permutation", "permutations", "arrangement", "arrangements", "subset", "select", "choose")
        or _phrase(text, "combination", "组合", "选法", "排列", "number of arrangements")
    )
    factorial_floor_value_context = (
        ("!" in text or _phrase(text, "factorial"))
        and (_phrase(text, "floor", "greatest integer", r"\lfloor", "ceiling") or r"\lceil" in text)
        and _phrase(text, "find the value", "compute the value", "evaluate", "determine the value")
        and not arrangement_context
    )
    _score_signal(
        scores,
        categories,
        "number_theory",
        "factorial_floor_value",
        8,
        factorial_floor_value_context,
    )

    _score_signal(
        scores,
        categories,
        "combinatorics",
        "counting_target",
        5,
        _phrase(
            text,
            "how many ways",
            "number of ways",
            "count arrangements",
            "count positive integers satisfying",
            "standing around a circle",
            "standing in a circle",
            "arranged in a circle",
            "consecutive integer representations",
        ),
    )
    _score_signal(
        scores,
        categories,
        "combinatorics",
        "discrete_structure",
        4,
        _looks_like_extremal_discrete_problem(text)
        or _phrase(
            text,
            "color a grid",
            "color a window",
            "color a lattice",
            "neighbor restrictions",
            "inclusion-exclusion",
            "pigeonhole",
        ),
    )
    _score_signal(
        scores,
        categories,
        "combinatorics",
        "arrangement_structure",
        3,
        arrangement_context,
    )
    _score_signal(scores, categories, "combinatorics", "generic_count", 2, _phrase(text, "how many", "find the number of ways"))

    random_context = _phrase(
        text,
        "uniformly at random",
        "at random",
        "independently",
        "random process",
        "randomly",
        "随机",
    )
    _score_signal(scores, categories, "probability", "random_structure", 6, random_context)
    _score_signal(
        scores,
        categories,
        "probability",
        "expectation_target",
        5,
        _phrase(
            text,
            "expected length",
            "expected value",
            "average value",
            "average over all equally likely",
            "probability of winning",
        ),
    )
    _score_signal(
        scores,
        categories,
        "probability",
        "probability_term",
        4,
        _word(text, "probability", "distribution", "variance") or _phrase(text, "概率", "骰子", "硬币", "红球", "蓝球"),
    )

    circle_arrangement = _phrase(text, "standing in a circle", "standing around a circle", "arranged in a circle")
    geometry_context = _phrase(
        text,
        "parallelogram",
        "angle bisector",
        "triangle",
        "octahedron",
        "icosahedron",
        "inscribed",
        "tangent",
        "radius",
        "diameter",
        "chord",
        "side length",
        "central angle",
        "ratio of geometric segments",
        "三角形",
        "平行四边形",
        "内切",
        "外接",
        "半径",
        "切线",
        "弦",
        "面积",
    )
    _score_signal(scores, categories, "geometry", "geometric_structure", 5, geometry_context)
    _score_signal(
        scores,
        categories,
        "geometry",
        "circle_with_geometry_context",
        3,
        not circle_arrangement and _word(text, "circle") and geometry_context,
    )

    _score_signal(
        scores,
        categories,
        "graph_theory",
        "graph_structure",
        5,
        _word(text, "vertices", "graph", "tree", "bipartite")
        or (_word(text, "edges", "degree", "cycle", "path") and _word(text, "graph", "vertices"))
        or _phrase(text, "independent set", "pairwise incompatible", "adjacency restrictions"),
    )
    _score_signal(
        scores,
        categories,
        "graph_theory",
        "implicit_incompatibility_graph",
        8,
        _phrase(text, "lattice points") and _phrase(text, "no two", "any two") and _phrase(text, "integer distance", "not an integer"),
    )

    _score_signal(
        scores,
        categories,
        "calculus",
        "calculus_structure",
        6,
        _phrase(
            text,
            "definite integral",
            "indefinite integral",
            "antiderivative",
            "infinite series",
            "infinite sum",
        )
        or _phrase(text, r"\int"),
    )
    _score_signal(
        scores,
        categories,
        "calculus",
        "calculus_operation",
        4,
        _word(text, "integrate", "derivative", "differentiate", "convergence")
        or _phrase(text, "导数", "积分", "极限"),
    )
    _score_signal(
        scores,
        categories,
        "calculus",
        "limit_operation",
        3,
        _word(text, "limit") or _phrase(text, r"\lim", r"^{\infty}", r"_\infty"),
    )

    _score_signal(
        scores,
        categories,
        "optimization",
        "optimization_structure",
        6,
        _phrase(text, "linear programming", "objective function", "subject to", "约束优化", "优化目标", "线性规划"),
    )
    _score_signal(
        scores,
        categories,
        "optimization",
        "optimization_target",
        3,
        _word(text, "optimize", "minimize", "maximize") or _phrase(text, "最值", "优化"),
    )
    _score_signal(scores, categories, "proof", "proof_target", 3, _word(text, "prove", "proof") or _phrase(text, "证明", "得证"))
    _score_signal(scores, categories, "topology", "topology_structure", 5, _phrase(text, "continuous map", "连续映射"))
    _score_signal(scores, categories, "discrete_math", "logic_structure", 4, _phrase(text, "命题", "逆否", "discrete logic"))

    # Exclusion and conflict contexts prevent known lexical false positives.
    _score_signal(scores, categories, "calculus", "excluded_integral_coefficient_context", -5, integral_coefficient_context)
    _score_signal(scores, categories, "geometry", "excluded_circle_arrangement_context", -4, circle_arrangement)
    _score_signal(
        scores,
        categories,
        "number_theory",
        "polynomial_prime_factor_conflict",
        -2,
        polynomial_context and _phrase(text, "prime factorization", "prime factors"),
    )
    pure_counting_target = _phrase(text, "number of ways", "how many ways", "number of permutations", "count arrangements")
    _score_signal(scores, categories, "probability", "pure_counting_conflict", -2, pure_counting_target and not random_context)

    ordered = sorted(scores, key=lambda candidate: (-scores[candidate], ROUTING_PRIORITY.index(candidate)))
    top_domain = ordered[0]
    runner_up_domain = ordered[1]
    top_score = scores[top_domain]
    runner_up_score = scores[runner_up_domain]
    score_margin = top_score - runner_up_score
    if top_score <= 0:
        domain = "unknown"
        confidence = "none"
        matched_categories: list[str] = []
        runner_up = None
        score_margin = 0
        reason = "题面没有有效的领域结构信号。"
    else:
        domain = top_domain
        confidence = "high" if top_score >= 6 and score_margin >= 3 else "medium" if top_score >= 4 and score_margin >= 2 else "low"
        matched_categories = sorted(categories[top_domain])
        runner_up = runner_up_domain if runner_up_score > 0 else None
        reason = "基于题面结构、目标词与冲突上下文的加权领域路由。"

    return {
        "domain": domain,
        "solver_key": solver_key_for_domain(domain),
        "reason": reason,
        "routing_confidence": confidence,
        "matched_signal_categories": matched_categories,
        "runner_up_domain": runner_up,
        "score_margin": score_margin,
    }
