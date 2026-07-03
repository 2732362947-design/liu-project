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
    "general",
)


def solver_key_for_domain(domain: str) -> str:
    mapping = {
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
    }
    return mapping.get(domain, "general")


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
    text = problem.lower()
    if any(token in text for token in ("probability", "概率", "随机", "骰子", "硬币", "红球", "蓝球")):
        domain = "probability"
        reason = "题目涉及随机试验或概率计算。"
    elif _looks_like_extremal_discrete_problem(text):
        domain = "combinatorics"
        reason = "题目是极值集合、组合图论或离散结构问题。"
    elif any(token in text for token in ("pde", "ode", "热方程", "偏微分", "微分方程", "u_t", "u_xx")):
        domain = "ode_pde"
        reason = "题目涉及常微分方程或偏微分方程。"
    elif any(token in text for token in ("combinatorics", "组合", "选法", "排列", "组合数")):
        domain = "discrete_math"
        reason = "题目涉及组合计数或排列组合。"
    elif any(token in text for token in ("graph_theory", "图论", "顶点", "边")):
        domain = "discrete_math"
        reason = "题目涉及图论或离散结构。"
    elif any(token in text for token in ("discrete", "离散", "命题", "逆否")):
        domain = "discrete_math"
        reason = "题目涉及离散数学、逻辑或组合计数。"
    elif any(token in text for token in ("prove", "proof", "证明", "得证")):
        domain = "proof"
        reason = "题目要求证明命题。"
    elif any(token in text for token in ("limit", "derivative", "极限", "导数", "微分")):
        domain = "calculus"
        reason = "题目涉及极限、导数或微积分概念。"
    elif any(
        token in text
        for token in ("linear_algebra", "线性代数", "矩阵", "行列式", "特征值", "向量空间", "秩")
    ):
        domain = "linear_algebra"
        reason = "题目涉及矩阵、线性空间或线性代数计算。"
    elif any(token in text for token in ("number_theory", "数论", "素数", "同余", "gcd", "最大公约数", "整除")):
        domain = "number_theory"
        reason = "题目涉及数论、整除、同余或素数。"
    elif any(token in text for token in ("triangle", "circle", "geometry", "三角形", "圆", "几何", "面积", "角")):
        domain = "geometry"
        reason = "题目包含几何对象或空间关系。"
    elif any(token in text for token in ("equation", "solve", "方程", "二次", "代数", "x^2")):
        domain = "algebra"
        reason = "题目主要是方程求解或代数运算。"
    elif any(
        token in text
        for token in ("optimization", "operations_research", "optimize", "maximum", "minimum", "运筹", "线性规划", "最优化", "约束优化", "最值", "优化")
    ):
        domain = "optimization"
        reason = "题目关注最优化或极值。"
    elif any(token in text for token in ("complex_analysis", "complex", "复分析", "复数", "留数", "解析函数", "cauchy", "laurent", "解析")):
        domain = "complex_analysis"
        reason = "题目涉及复数或复分析术语。"
    elif any(token in text for token in ("topology", "拓扑", "连续映射")):
        domain = "topology"
        reason = "题目涉及拓扑相关概念。"
    else:
        domain = "unknown"
        reason = "题目缺少足够明显的领域关键词。"

    return {"domain": domain, "solver_key": solver_key_for_domain(domain), "reason": reason}
