DOMAINS = (
    "calculus",
    "algebra",
    "geometry",
    "probability",
    "ode_pde",
    "discrete_math",
    "proof",
    "real_analysis",
    "topology",
    "complex_analysis",
    "optimization",
    "unknown",
)

SOLVER_KEYS = (
    "algebra",
    "calculus",
    "probability",
    "ode_pde",
    "proof",
    "discrete",
    "general",
)


def solver_key_for_domain(domain: str) -> str:
    mapping = {
        "algebra": "algebra",
        "calculus": "calculus",
        "probability": "probability",
        "ode_pde": "ode_pde",
        "proof": "proof",
        "real_analysis": "proof",
        "topology": "proof",
        "discrete_math": "discrete",
    }
    return mapping.get(domain, "general")


def classify_problem(problem: str) -> dict:
    text = problem.lower()
    if any(token in text for token in ("probability", "概率", "随机", "骰子", "硬币", "红球", "蓝球")):
        domain = "probability"
        reason = "题目涉及随机试验或概率计算。"
    elif any(token in text for token in ("pde", "ode", "热方程", "偏微分", "微分方程", "u_t", "u_xx")):
        domain = "ode_pde"
        reason = "题目涉及常微分方程或偏微分方程。"
    elif any(token in text for token in ("discrete", "离散", "命题", "逆否", "图论", "组合", "排列", "选法")):
        domain = "discrete_math"
        reason = "题目涉及离散数学、逻辑或组合计数。"
    elif any(token in text for token in ("prove", "proof", "证明", "得证")):
        domain = "proof"
        reason = "题目要求证明命题。"
    elif any(token in text for token in ("limit", "derivative", "极限", "导数", "微分")):
        domain = "calculus"
        reason = "题目涉及极限、导数或微积分概念。"
    elif any(token in text for token in ("triangle", "circle", "geometry", "三角形", "圆", "几何")):
        domain = "geometry"
        reason = "题目包含几何对象或空间关系。"
    elif any(token in text for token in ("equation", "solve", "方程", "二次", "代数", "x^2")):
        domain = "algebra"
        reason = "题目主要是方程求解或代数运算。"
    elif any(token in text for token in ("optimize", "maximum", "minimum", "最值", "优化")):
        domain = "optimization"
        reason = "题目关注最优化或极值。"
    elif any(token in text for token in ("complex", "复数", "解析")):
        domain = "complex_analysis"
        reason = "题目涉及复数或复分析术语。"
    elif any(token in text for token in ("topology", "拓扑", "连续映射")):
        domain = "topology"
        reason = "题目涉及拓扑相关概念。"
    else:
        domain = "unknown"
        reason = "题目缺少足够明显的领域关键词。"

    return {"domain": domain, "solver_key": solver_key_for_domain(domain), "reason": reason}
