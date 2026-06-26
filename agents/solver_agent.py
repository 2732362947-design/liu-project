from intern_s1_client import call_intern_s1


def build_solver_prompt(
    problem: str,
    domain: str,
    plan: list[str],
    retry_context: str | None = None,
) -> str:
    plan_text = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(plan))
    retry_text = ""
    if retry_context:
        retry_text = (
            "\n上一轮问题：\n"
            f"{retry_context}\n"
        )
    return (
        "你是 Intern-S1 数学解题模型。\n"
        f"domain: {domain}\n"
        f"题目: {problem}\n"
        f"plan:\n{plan_text}\n"
        f"{retry_text}"
        "请给出必要推理，不要写冗长教学解释。\n"
        "最后必须单独写一行：最终答案：..."
    )


def solve_problem(
    problem: str,
    domain: str,
    plan: list[str],
    retry_context: str | None = None,
) -> str:
    prompt = build_solver_prompt(problem, domain, plan, retry_context)
    return call_intern_s1(prompt)
