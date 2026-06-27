import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.classifier_agent import classify_problem, solver_key_for_domain


DEFAULT_QUESTIONS_FILE = ROOT / "data" / "dev_questions.json"
DEFAULT_OUTPUT_FILE = ROOT / "outputs" / "domain_coverage.md"
RECOMMENDATION_RULES = (
    (
        ("complex_analysis", "复分析", "留数", "解析函数", "cauchy"),
        "agents/solvers/complex_analysis.txt",
        "route complex analysis items to a dedicated template",
    ),
    (
        ("topology", "拓扑", "开集", "闭集", "连续映射"),
        "agents/solvers/topology.txt",
        "route topology items to a dedicated template",
    ),
    (
        ("optimization", "operations_research", "运筹", "线性规划", "最优化"),
        "agents/solvers/optimization.txt",
        "route optimization or operations research items to a dedicated template",
    ),
    (
        ("linear_algebra", "线性代数", "矩阵", "特征值", "向量空间"),
        "agents/solvers/linear_algebra.txt",
        "route linear algebra items to a dedicated template",
    ),
    (
        ("number_theory", "数论", "素数", "同余", "gcd", "最大公约数"),
        "agents/solvers/number_theory.txt",
        "route number theory items to a dedicated template",
    ),
    (
        ("geometry", "几何", "三角形", "圆", "面积", "角"),
        "agents/solvers/geometry.txt",
        "route geometry items to a dedicated template",
    ),
    (
        ("real_analysis", "实分析", "收敛", "极限", "连续", "一致收敛"),
        "agents/solvers/real_analysis.txt",
        "route real analysis items to a dedicated template",
    ),
    (
        ("combinatorics", "组合", "选法", "排列", "组合数"),
        "agents/solvers/discrete.txt",
        "route combinatorics items to existing discrete template",
    ),
    (
        ("graph_theory", "图论", "顶点", "边"),
        "agents/solvers/discrete.txt",
        "route graph theory items to existing discrete template",
    ),
)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_questions(path: str | Path) -> list[dict]:
    data = json.loads(_resolve_path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _domain_for_question(question: dict) -> str:
    domain = question.get("domain")
    if domain:
        return str(domain)
    return "unknown"


def infer_solver_key(question: dict) -> str:
    if question.get("solver_key"):
        return str(question["solver_key"])
    domain = _domain_for_question(question)
    if domain != "unknown":
        return solver_key_for_domain(domain)
    problem = str(question.get("problem") or "")
    if problem:
        return classify_problem(problem).get("solver_key", "general")
    return "general"


def _problem_id(question: dict, index: int) -> str:
    return str(question.get("problem_id") or f"question_{index + 1}")


def _problem_head(problem: Any, limit: int = 80) -> str:
    text = " ".join(str(problem or "").split())
    return text if len(text) <= limit else text[:limit] + "..."


def _recommendation_key(domain: str, problem: Any) -> tuple[str, str] | None:
    haystack = f"{domain} {problem}".lower()
    for keywords, template, reason in RECOMMENDATION_RULES:
        for keyword in keywords:
            if keyword.lower() in haystack:
                return template, reason
    return None


def build_domain_coverage(questions: list[dict]) -> dict:
    enriched = []
    domain_counts: Counter = Counter()
    solver_counts: Counter = Counter()
    domain_solver_counts: Counter = Counter()
    recommendations: Counter = Counter()

    for index, question in enumerate(questions):
        domain = _domain_for_question(question)
        solver_key = infer_solver_key(question)
        problem_id = _problem_id(question, index)
        problem = question.get("problem", "")
        row = {
            "problem_id": problem_id,
            "domain": domain,
            "solver_key": solver_key,
            "problem_head": _problem_head(problem),
        }
        enriched.append(row)
        domain_counts[domain] += 1
        solver_counts[solver_key] += 1
        domain_solver_counts[(domain, solver_key)] += 1
        if solver_key == "general":
            recommendation = _recommendation_key(domain, problem)
            if recommendation:
                recommendations[recommendation] += 1

    total = len(questions)
    general_count = solver_counts["general"]
    return {
        "total_questions": total,
        "unique_domains": len(domain_counts),
        "unique_solver_keys": len(solver_counts),
        "general_count": general_count,
        "general_ratio": general_count / total if total else 0.0,
        "unknown_domain_count": domain_counts["unknown"],
        "domain_counts": dict(domain_counts),
        "solver_key_counts": dict(solver_counts),
        "domain_solver_counts": {
            f"{domain}||{solver_key}": count
            for (domain, solver_key), count in domain_solver_counts.items()
        },
        "items": enriched,
        "recommendations": [
            {
                "recommended_template": template,
                "reason": reason,
                "affected_count": count,
            }
            for (template, reason), count in sorted(recommendations.items())
        ],
    }


def _escape(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_escape(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _count_rows(counts: dict[str, int]) -> list[list[Any]]:
    return [[key, counts[key]] for key in sorted(counts)]


def build_markdown_report(coverage: dict) -> str:
    overview_rows = [
        ["total_questions", coverage.get("total_questions", 0)],
        ["unique_domains", coverage.get("unique_domains", 0)],
        ["unique_solver_keys", coverage.get("unique_solver_keys", 0)],
        ["general_count", coverage.get("general_count", 0)],
        ["general_ratio", f"{coverage.get('general_ratio', 0.0):.4f}"],
        ["unknown_domain_count", coverage.get("unknown_domain_count", 0)],
    ]
    domain_solver_rows = []
    for key, count in sorted(coverage.get("domain_solver_counts", {}).items()):
        domain, solver_key = key.split("||", 1)
        domain_solver_rows.append([domain, solver_key, count])

    general_rows = [
        [item["problem_id"], item["domain"], item["problem_head"]]
        for item in coverage.get("items", [])
        if item.get("solver_key") == "general"
    ]
    recommendation_rows = [
        [item["recommended_template"], item["reason"], item["affected_count"]]
        for item in coverage.get("recommendations", [])
    ]

    sections = [
        "# Domain Coverage Report",
        "## 1. Overview",
        _table(["metric", "value"], overview_rows),
        "## 2. Domain Distribution",
        _table(["domain", "count"], _count_rows(coverage.get("domain_counts", {}))),
        "## 3. Solver Key Distribution",
        _table(["solver_key", "count"], _count_rows(coverage.get("solver_key_counts", {}))),
        "## 4. Domain to Solver Mapping",
        _table(["domain", "solver_key", "count"], domain_solver_rows),
        "## 5. General Fallback Items",
    ]
    if general_rows:
        sections.append(_table(["problem_id", "domain", "problem_head"], general_rows))
    else:
        sections.append("No general fallback items found.")
    sections.append("## 6. Recommended Template Additions")
    if recommendation_rows:
        sections.append(_table(["recommended_template", "reason", "affected_count"], recommendation_rows))
    elif coverage.get("general_count", 0) > 0:
        sections.append(
            "General fallback items exist, but no specific template rule matched. "
            "Consider improving classifier keywords."
        )
    else:
        sections.append("No additional templates recommended for current dataset.")
    return "\n\n".join(sections) + "\n"


def write_domain_coverage_report(questions_path: str | Path, output_path: str | Path) -> dict:
    coverage = build_domain_coverage(load_questions(questions_path))
    output_file = _resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(build_markdown_report(coverage), encoding="utf-8")
    return coverage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check domain and solver_key coverage for a question set.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_FILE.relative_to(ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE.relative_to(ROOT)))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_file = _resolve_path(args.output)
    coverage = write_domain_coverage_report(args.questions, output_file)
    recommendations = coverage.get("recommendations", [])
    recommended_templates = ", ".join(item["recommended_template"] for item in recommendations) or "none"
    print(f"Domain coverage report written to: {output_file}")
    print(f"general_count={coverage['general_count']}")
    print(f"recommended_templates={recommended_templates}")


if __name__ == "__main__":
    main()
