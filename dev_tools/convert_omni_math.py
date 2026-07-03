import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = Path(
    "/home/ubuntu/.cache/modelscope/hub/datasets/AI-ModelScope/Omni-MATH/test.jsonl"
)
DEFAULT_OUTPUT_FILE = ROOT / "data" / "omni_math_sample.json"


DOMAIN_RULES = (
    ("complex_analysis", ("Complex Analysis",)),
    ("linear_algebra", ("Linear Algebra",)),
    ("number_theory", ("Number Theory",)),
    ("operations_research", ("Operations Research",)),
    ("optimization", ("Optimization",)),
    ("ode_pde", ("Differential Equation", "Differential Equations", "ODE", "PDE")),
    ("real_analysis", ("Real Analysis", "Analysis")),
    ("graph_theory", ("Graph Theory",)),
    ("combinatorics", ("Combinatorics", "Discrete Mathematics")),
    ("probability", ("Probability", "Statistics", "Mathematical Statistics")),
    ("calculus", ("Calculus",)),
    ("geometry", ("Geometry",)),
    ("algebra", ("Algebra",)),
)
OTHER_DOMAIN_MARKERS = ("other", "misc", "unknown", "uncategorized")
EXTREMAL_SET_MARKERS = (
    "k-element subset",
    "every k-element subset",
    "subset",
)
EXTREMAL_DISCRETE_MARKERS = (
    "contains two distinct elements",
    "two distinct elements",
    "pair of elements",
    "divides",
    "integer",
    "positive integer",
    "{1,2,...",
    "{1, 2, ...",
    "extremal",
    "independent set",
    "choose",
)
OPTIMIZATION_STRONG_MARKERS = (
    "linear programming",
    "objective function",
    "minimize cost",
    "maximize profit",
    "maximize function",
    "minimize function",
    "operations research",
    "feasible region",
    "constraints",
    "subject to",
)
PROBLEM_DOMAIN_RULES = (
    (
        "graph_theory",
        (
            "independent set",
            "graph model",
            "vertices",
            "vertex",
            "edges",
        ),
    ),
    (
        "graph_theory",
        (
            "graph theory",
            "tournament",
            "coloring",
            "vertices",
            "vertex",
            "edges",
        ),
    ),
    (
        "combinatorics",
        (
            "combinatorics",
            "subset",
            "k-element subset",
            "every k-element subset",
            "contains two distinct elements",
            "pair of elements",
            "round table",
            "transfer",
            "adjacent",
            "cards",
            "invariant",
            "strategy",
            "game",
            "diode",
            "electron",
            "same state",
            "guarantee",
            "construction",
            "choose",
            "selection",
            "arrangement",
            "permutation",
            "usamon",
            "any order",
            "less than or equal",
            "satisfy any order",
            "cups",
            "positive $x$",
            "negative $x$",
            "caught",
            "phone number",
            "digits",
            "letter",
            "occur",
            "how many times",
            "standing in a circle",
            "next to",
            "between two",
            "ducks",
            "cows",
            "rabbits",
            "in how many ways",
            "dollar notes",
            "unlimited supply",
            "committee",
            "committees",
            "senator",
            "senators",
            "aides",
        ),
    ),
    (
        "algebra",
        (
            "constant ground speed",
            "speed",
            "km",
            "meters",
            "distance",
            "flies at",
            "working together",
            "demolish",
            "fraction in lowest terms",
            "days",
        ),
    ),
    (
        "real_analysis",
        (
            "function f",
            "functions f",
            "f:",
            r"\mathbb{z}^2",
            "recurrence",
            "average",
            "bounded",
            "constant function",
            "[0,1]",
            "[0, 1]",
        ),
    ),
    (
        "optimization",
        (
            "optimize",
            "cost",
            "linear programming",
            "constraints",
            "objective function",
            "feasible region",
            "subject to",
            "minimize cost",
            "maximize function",
            "how many seconds",
            "takes",
            "cannot begin",
            "every 8 minutes",
        ),
    ),
    (
        "probability",
        (
            "independent and identically distributed",
            "density function",
            "exponential distribution",
            "unbiased estimator",
            "maximum likelihood",
            "google",
            "hits",
            "search",
        ),
    ),
)
NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+)$")


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def raw_domain_text(domain_value: Any) -> str:
    if isinstance(domain_value, list) and domain_value:
        return str(domain_value[0])
    if isinstance(domain_value, str):
        return domain_value
    return ""


def _domain_from_text(text: str) -> str:
    normalized = text.lower()
    for simplified, keywords in DOMAIN_RULES:
        if any(keyword.lower() in normalized for keyword in keywords):
            return "optimization" if simplified == "operations_research" else simplified
    return "unknown"


def _domain_from_extremal_set_context(problem: Any, raw_domain: Any = "") -> str:
    combined = f"{raw_domain or ''} {problem or ''}".lower()
    has_subset = any(marker in combined for marker in EXTREMAL_SET_MARKERS)
    has_discrete_structure = any(marker in combined for marker in EXTREMAL_DISCRETE_MARKERS)
    if has_subset and has_discrete_structure:
        return "combinatorics"
    if "graph" in combined and any(marker in combined for marker in ("independent set", "coloring", "tournament")):
        return "graph_theory"
    if "divides" in combined and any(marker in combined for marker in ("integer", "positive integer", "pair")):
        return "number_theory"
    return "unknown"


def _domain_from_problem(problem: Any) -> str:
    normalized = str(problem or "").lower()
    extremal_domain = _domain_from_extremal_set_context(normalized)
    if extremal_domain != "unknown":
        return extremal_domain
    for domain, keywords in PROBLEM_DOMAIN_RULES:
        if any(keyword in normalized for keyword in keywords):
            return domain
    return "unknown"


def simplify_domain(domain_value: Any, problem: Any = "") -> str:
    raw_domain = raw_domain_text(domain_value)
    raw_normalized = raw_domain.lower()
    extremal_domain = _domain_from_extremal_set_context(problem, raw_domain)
    if extremal_domain != "unknown":
        return extremal_domain
    domain = _domain_from_text(raw_domain)
    if domain != "unknown" and not any(marker in raw_normalized for marker in OTHER_DOMAIN_MARKERS):
        return domain

    inferred_from_problem = _domain_from_problem(problem)
    if inferred_from_problem != "unknown":
        return inferred_from_problem
    return domain


def infer_answer_type(answer: str | None, problem: str | None = None) -> str:
    answer_text = str(answer or "").strip()
    problem_text = str(problem or "").lower()
    lower_answer = answer_text.lower()
    combined = f"{lower_answer} {problem_text}"

    if not answer_text:
        return "text"
    if any(token in combined for token in ("prove", "proof", "show that")):
        return "proof"
    if NUMBER_PATTERN.fullmatch(answer_text.replace(" ", "")):
        return "number"
    if any(
        token in answer_text
        for token in ("\\", "=", "^", "frac", "sqrt", "sin", "cos", "ceil", "floor")
    ):
        return "expression"
    if "{" in answer_text or "}" in answer_text or re.search(r"\b\d+\s*,\s*\d+\b", answer_text):
        return "set"
    if re.search(r"[a-zA-Z]\s*(?:\^|_|\()", answer_text):
        return "expression"
    return "text"


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with _resolve_path(path).open(encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _difficulty_allowed(
    difficulty: Any,
    min_difficulty: float | None,
    max_difficulty: float | None,
) -> bool:
    if min_difficulty is None and max_difficulty is None:
        return True
    try:
        value = float(difficulty)
    except (TypeError, ValueError):
        return False
    if min_difficulty is not None and value < min_difficulty:
        return False
    if max_difficulty is not None and value > max_difficulty:
        return False
    return True


def convert_records(
    records: list[dict],
    max_per_domain: int = 3,
    max_total: int = 90,
    min_difficulty: float | None = None,
    max_difficulty: float | None = None,
) -> list[dict]:
    converted = []
    per_domain_counts: Counter = Counter()

    for item in records:
        if len(converted) >= max_total:
            break
        if not _difficulty_allowed(item.get("difficulty"), min_difficulty, max_difficulty):
            continue

        raw_domain = raw_domain_text(item.get("domain"))
        domain = simplify_domain(item.get("domain"), item.get("problem", ""))
        if domain == "unknown":
            continue
        if per_domain_counts[domain] >= max_per_domain:
            continue

        answer = item.get("answer", "")
        converted_item = {
            "problem_id": f"omni_{len(converted) + 1:06d}",
            "domain": domain,
            "raw_domain": raw_domain,
            "problem": item.get("problem", ""),
            "answer_type": infer_answer_type(answer, item.get("problem", "")),
            "expected_answer": answer,
            "source": item.get("source", ""),
            "difficulty": item.get("difficulty"),
        }
        if isinstance(item.get("domain"), list):
            converted_item["raw_domain_list"] = item.get("domain")
        converted.append(converted_item)
        per_domain_counts[domain] += 1

    return converted


def convert_omni_math_file(
    input_path: str | Path = DEFAULT_INPUT_FILE,
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
    max_per_domain: int = 3,
    max_total: int = 90,
    min_difficulty: float | None = None,
    max_difficulty: float | None = None,
) -> list[dict]:
    converted = convert_records(
        load_jsonl(input_path),
        max_per_domain=max_per_domain,
        max_total=max_total,
        min_difficulty=min_difficulty,
        max_difficulty=max_difficulty,
    )
    output_file = _resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    return converted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an Omni-MATH jsonl sample into project question format.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE.relative_to(ROOT)))
    parser.add_argument("--max-per-domain", type=int, default=3)
    parser.add_argument("--max-total", type=int, default=90)
    parser.add_argument("--min-difficulty", type=float, default=None)
    parser.add_argument("--max-difficulty", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    converted = convert_omni_math_file(
        args.input,
        args.output,
        max_per_domain=args.max_per_domain,
        max_total=args.max_total,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
    )
    output_file = _resolve_path(args.output)
    domain_counts = Counter(item["domain"] for item in converted)
    domain_counts_text = ", ".join(
        f"{domain}:{domain_counts[domain]}" for domain in sorted(domain_counts)
    )
    print(f"Converted {len(converted)} Omni-MATH items to: {output_file}")
    print(f"domain_counts={domain_counts_text}")


if __name__ == "__main__":
    main()
