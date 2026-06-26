def _core_hint(problem: str) -> str:
    if "x^2" in problem and "方程" in problem:
        return "把二次方程化成两个一次因式，根来自每个因式为 0 的情况。"
    if "红球" in problem and "蓝球" in problem:
        return "这是等可能抽取问题，概率等于有利情况数除以总情况数。"
    if "导数" in problem and "x^2" in problem:
        return "先用幂函数求导法则得到导函数，再代入指定点。"
    if "三角形" in problem or "圆" in problem:
        return "先找几何对象之间的边、角、面积关系，再用定理列式。"
    return "先把题目目标和已知条件分开，再按计划把条件转成可计算或可证明的步骤。"


def explain_solution(
    problem: str,
    solution: str,
    plan: list[str],
    final_answer: str | None = None,
) -> str:
    lower_solution = solution.strip().lower()
    if lower_solution.startswith("[intern-s1 error]") or lower_solution.startswith("[mock intern-s1]"):
        return (
            "正式模型调用失败，当前仅保留诊断信息。"
            "因此这里不把本地规则或猜测内容解释成正式解法；请先检查网络、API 地址或模型服务状态。"
        )

    plan_text = "；".join(plan)
    answer_text = final_answer or "尚未抽取到明确答案"
    return (
        f"本题核心思想：{_core_hint(problem)}"
        f"关键步骤：{plan_text}。"
        f"final_answer 的含义：{answer_text} 是从正式解答中抽取出的最终结论。"
        "易错点：不要只看最终数字，还要确认它确实回答了题目所问；"
        "涉及方程要代回检验，涉及概率要核对总样本数，涉及导数要区分函数值和导数值。"
    )
