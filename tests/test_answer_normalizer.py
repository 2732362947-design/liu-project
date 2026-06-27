from agents.answer_normalizer import answers_match, normalize_answer


def test_fraction_normalization():
    assert normalize_answer(r"\frac{3}{5}") == normalize_answer("3/5")


def test_root_set_normalization():
    assert normalize_answer("x_1=2, x_2=3") == normalize_answer("{2,3}")


def test_none_normalization():
    assert normalize_answer(None) == ""


def test_number_assignment_matches_plain_number():
    matched, _ = answers_match("x = 6", "6", answer_type="number")

    assert matched


def test_number_count_unit_matches_plain_number():
    matched, _ = answers_match("10 种", "10", answer_type="number")

    assert matched


def test_number_selection_unit_matches_plain_number():
    matched, _ = answers_match("10种选法", "10", answer_type="number")

    assert matched


def test_number_solution_count_unit_matches_plain_number():
    matched, _ = answers_match("2 个解", "2", answer_type="number")

    assert matched


def test_number_count_unit_mismatch_fails():
    matched, _ = answers_match("11 种", "10", answer_type="number")

    assert not matched


def test_expression_drops_simple_left_side():
    matched, _ = answers_match("y = e^x", "e^x", answer_type="expression")

    assert matched


def test_expression_constant_spacing_matches():
    matched, _ = answers_match("y = x^2 + C", "x^2+C", answer_type="expression")

    assert matched


def test_pde_expression_matches_partial_derivative_forms():
    matched, _ = answers_match(
        r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}",
        "u_t=αu_xx",
        answer_type="expression",
    )

    assert matched


def test_logic_expression_matches_chinese_and_latex_forms():
    matched, _ = answers_match(r"$\neg Q \to \neg P$", "非Q->非P", answer_type="expression")

    assert matched


def test_logic_expression_matches_latex_rightarrow():
    matched, _ = answers_match(
        r"$\neg Q \rightarrow \neg P$",
        "非Q->非P",
        answer_type="expression",
    )

    assert matched


def test_logic_expression_matches_latex_to():
    matched, _ = answers_match(r"\neg Q \to \neg P", "非Q->非P", answer_type="expression")

    assert matched


def test_heat_equation_matches_k_coefficient():
    matched, _ = answers_match(
        r"\frac{\partial u}{\partial t} = k \frac{\partial^2 u}{\partial x^2}",
        "u_t=αu_xx",
        answer_type="expression",
        problem="pde: 写出一维热方程的标准形式。",
    )

    assert matched


def test_heat_equation_matches_alpha_coefficient():
    matched, _ = answers_match(
        r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}",
        "u_t=αu_xx",
        answer_type="expression",
        problem="pde: 写出一维热方程的标准形式。",
    )

    assert matched


def test_non_heat_expression_does_not_unify_k_and_alpha():
    matched, _ = answers_match(
        "kx",
        "αx",
        answer_type="expression",
        problem="比较表达式",
    )

    assert not matched


def test_text_match_uses_problem_context():
    matched, _ = answers_match(
        "是素数",
        "29是素数",
        answer_type="text",
        problem="number_theory: 判断 29 是否为素数，并说明理由。",
    )

    assert matched


def test_topology_text_matches_keywords():
    matched, _ = answers_match(
        "映射 f: X -> Y 连续当且仅当对于 Y 中任意开集 V，原像 f^{-1}(V) 是 X 中的开集",
        "开集的原像是开集",
        answer_type="text",
    )

    assert matched


def test_proof_uses_solution_when_final_answer_is_generic():
    matched, _ = answers_match(
        "命题得证",
        "子列也收敛于a",
        answer_type="proof",
        solution="由定义可得，故子列 {a_{n_k}} 收敛于 a。",
    )

    assert matched


def test_number_mismatch_fails():
    matched, _ = answers_match("5", "6", answer_type="number")

    assert not matched


def test_expression_mismatch_fails():
    matched, _ = answers_match("x^2", "x^2+C", answer_type="expression")

    assert not matched
