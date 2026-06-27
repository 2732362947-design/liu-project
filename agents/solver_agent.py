from functools import lru_cache
from pathlib import Path

from intern_s1_client import call_intern_s1


SOLVER_TEMPLATE_DIR = Path(__file__).resolve().parent / "solvers"
VALID_SOLVER_KEYS = {
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
}
DOMAIN_TO_SOLVER_KEY = {
    "algebra": "algebra",
    "calculus": "calculus",
    "probability": "probability",
    "ode_pde": "ode_pde",
    "proof": "proof",
    "complex_analysis": "complex_analysis",
    "geometry": "geometry",
    "linear_algebra": "linear_algebra",
    "number_theory": "number_theory",
    "optimization": "optimization",
    "real_analysis": "proof",
    "topology": "proof",
    "discrete_math": "discrete",
    "combinatorics": "discrete",
    "graph_theory": "discrete",
}


def normalize_solver_key(solver_key: str | None, domain: str | None = None) -> str:
    if solver_key in VALID_SOLVER_KEYS:
        return solver_key
    if domain in DOMAIN_TO_SOLVER_KEY:
        return DOMAIN_TO_SOLVER_KEY[domain]
    return "general"


@lru_cache(maxsize=None)
def load_solver_template(solver_key: str) -> str:
    key = normalize_solver_key(solver_key)
    return (SOLVER_TEMPLATE_DIR / f"{key}.txt").read_text(encoding="utf-8")


def build_solver_prompt(
    problem: str,
    domain: str,
    plan: list[str],
    retry_context: str | None = None,
    solver_key: str | None = None,
) -> str:
    plan_text = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(plan))
    retry_block = ""
    if retry_context:
        retry_block = (
            "上一轮问题：\n"
            f"{retry_context}\n"
        )
    key = normalize_solver_key(solver_key, domain)
    template = load_solver_template(key)
    return template.format(
        domain=domain,
        problem=problem,
        plan_text=plan_text,
        retry_block=retry_block,
    )


def solve_problem(
    problem: str,
    domain: str,
    plan: list[str],
    retry_context: str | None = None,
    solver_key: str | None = None,
) -> str:
    prompt = build_solver_prompt(problem, domain, plan, retry_context, solver_key)
    return call_intern_s1(prompt)
