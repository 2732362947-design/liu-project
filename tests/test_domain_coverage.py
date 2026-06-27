from dev_tools.check_domain_coverage import (
    build_domain_coverage,
    build_markdown_report,
    infer_solver_key,
)


def test_basic_coverage_counts_domains():
    questions = [
        {"problem_id": "a1", "domain": "algebra", "problem": "解方程。"},
        {"problem_id": "p1", "domain": "probability", "problem": "求概率。"},
        {"problem_id": "t1", "domain": "topology", "problem": "连续映射。"},
    ]

    coverage = build_domain_coverage(questions)

    assert coverage["total_questions"] == 3
    assert coverage["domain_counts"]["algebra"] == 1
    assert coverage["domain_counts"]["probability"] == 1
    assert coverage["domain_counts"]["topology"] == 1


def test_solver_key_distribution_uses_classifier_routes():
    questions = [
        {"problem_id": "p1", "problem": "probability: 求事件概率。"},
        {"problem_id": "o1", "problem": "pde: 写出一维热方程。"},
        {"problem_id": "d1", "problem": "离散数学：命题 P->Q 的逆否命题是什么？"},
    ]

    coverage = build_domain_coverage(questions)

    assert infer_solver_key(questions[0]) == "probability"
    assert infer_solver_key(questions[1]) == "ode_pde"
    assert infer_solver_key(questions[2]) == "discrete"
    assert coverage["solver_key_counts"]["probability"] == 1
    assert coverage["solver_key_counts"]["ode_pde"] == 1
    assert coverage["solver_key_counts"]["discrete"] == 1


def test_general_fallback_items_are_listed():
    questions = [{"problem_id": "u1", "domain": "unknown", "problem": "暂未分类的问题。"}]

    coverage = build_domain_coverage(questions)
    report = build_markdown_report(coverage)

    assert coverage["general_count"] == 1
    assert "General Fallback Items" in report
    assert "u1" in report


def test_recommended_templates_for_general_domains():
    questions = [
        {"problem_id": "c1", "domain": "complex_analysis", "solver_key": "general", "problem": "复分析。"},
        {"problem_id": "t1", "domain": "topology", "solver_key": "general", "problem": "拓扑。"},
        {"problem_id": "o1", "domain": "optimization", "solver_key": "general", "problem": "优化。"},
    ]

    coverage = build_domain_coverage(questions)
    templates = {item["recommended_template"] for item in coverage["recommendations"]}

    assert "agents/solvers/complex_analysis.txt" in templates
    assert "agents/solvers/topology.txt" in templates
    assert "agents/solvers/optimization.txt" in templates


def test_general_complex_analysis_recommends_template():
    questions = [
        {
            "problem_id": "c1",
            "domain": "complex_analysis",
            "solver_key": "general",
            "problem": "计算留数。",
        }
    ]

    coverage = build_domain_coverage(questions)

    assert coverage["recommendations"][0]["recommended_template"] == "agents/solvers/complex_analysis.txt"


def test_unknown_topology_problem_recommends_topology_template():
    questions = [
        {
            "problem_id": "t1",
            "domain": "unknown",
            "solver_key": "general",
            "problem": "拓扑题：证明开集在连续映射下的原像性质。",
        }
    ]

    coverage = build_domain_coverage(questions)

    assert coverage["recommendations"][0]["recommended_template"] == "agents/solvers/topology.txt"


def test_unknown_matrix_problem_recommends_linear_algebra_template():
    questions = [
        {
            "problem_id": "l1",
            "domain": "unknown",
            "solver_key": "general",
            "problem": "给定矩阵，求它的特征值。",
        }
    ]

    coverage = build_domain_coverage(questions)

    assert coverage["recommendations"][0]["recommended_template"] == "agents/solvers/linear_algebra.txt"


def test_general_without_matching_rule_reports_classifier_hint():
    coverage = build_domain_coverage(
        [{"problem_id": "u1", "domain": "unknown", "solver_key": "general", "problem": "暂未归类。"}]
    )

    report = build_markdown_report(coverage)

    assert "General fallback items exist" in report


def test_missing_fields_get_fallback_id_and_unknown_domain():
    questions = [{"problem": "没有显式 id 和 domain。"}]

    coverage = build_domain_coverage(questions)
    report = build_markdown_report(coverage)

    assert coverage["unknown_domain_count"] == 1
    assert coverage["items"][0]["problem_id"] == "question_1"
    assert "unknown" in report
    assert "question_1" in report


def test_markdown_report_contains_required_sections():
    coverage = build_domain_coverage(
        [{"problem_id": "a1", "domain": "algebra", "problem": "解方程。"}]
    )

    report = build_markdown_report(coverage)

    assert "Domain Coverage Report" in report
    assert "Domain Distribution" in report
    assert "Solver Key Distribution" in report
    assert "Domain to Solver Mapping" in report
    assert "Recommended Template Additions" in report
