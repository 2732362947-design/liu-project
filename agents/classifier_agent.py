DOMAINS = (
    "calculus",
    "algebra",
    "geometry",
    "probability",
    "topology",
    "complex_analysis",
    "optimization",
    "unknown",
)


def classify_problem(problem: str) -> dict:
    text = problem.lower()
    if any(token in text for token in ("probability", "概率", "随机", "骰子", "硬币", "红球", "蓝球")):
        domain = "probability"
        reason = "题目涉及随机试验或概率计算。"
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

    return {"domain": domain, "reason": reason}
