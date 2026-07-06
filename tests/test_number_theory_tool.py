from agents.tools.number_theory_tool import (
    crt,
    crt_pair,
    euler_phi,
    extended_gcd,
    factor_int,
    gcd,
    mod_inverse,
    multiplicative_order,
    solve_number_theory_problem,
)


def test_number_theory_math_functions():
    assert gcd(54, 24) == 6
    g, x, y = extended_gcd(30, 12)
    assert g == 6
    assert 30 * x + 12 * y == 6
    assert mod_inverse(3, 11) == 4
    assert mod_inverse(2, 4) is None
    assert factor_int(1) == {}
    assert factor_int(60) == {2: 2, 3: 1, 5: 1}
    assert euler_phi(1) == 1
    assert euler_phi(36) == 12
    assert euler_phi(100) == 40
    assert crt_pair(2, 3, 3, 5) == (8, 15)
    assert crt_pair(1, 2, 0, 4) is None
    assert crt([(2, 3), (3, 5)]) == (8, 15)
    assert multiplicative_order(2, 7) == 3
    assert multiplicative_order(2, 9) == 6
    assert multiplicative_order(2, 4) is None


def test_english_euler_phi_problem():
    result = solve_number_theory_problem("Compute Euler's totient function phi(100).")
    assert result is not None
    assert result["final_answer"] == "40"
    assert result["tool_name"] == "number_theory_tool"
    assert result["details"]["problem_type"] == "euler_phi"


def test_chinese_euler_phi_problem():
    result = solve_number_theory_problem("求欧拉函数 φ(36)")
    assert result is not None
    assert result["final_answer"] == "12"


def test_english_modular_inverse_problem():
    result = solve_number_theory_problem("Find the modular inverse of 3 modulo 11.")
    assert result is not None
    assert result["final_answer"] == "4"
    assert result["details"]["problem_type"] == "modular_inverse"


def test_chinese_modular_inverse_problem():
    result = solve_number_theory_problem("求 7 在模 26 下的乘法逆元")
    assert result is not None
    assert result["final_answer"] == "15"


def test_modular_inverse_nonexistent_problem():
    result = solve_number_theory_problem("Find the modular inverse of 2 modulo 4.")
    assert result is not None
    assert result["final_answer"] == "不存在"


def test_english_crt_problem():
    result = solve_number_theory_problem("Solve x ≡ 2 (mod 3), x ≡ 3 (mod 5).")
    assert result is not None
    assert "8" in result["final_answer"]
    assert "15" in result["final_answer"]
    assert result["details"]["problem_type"] == "chinese_remainder_theorem"


def test_chinese_crt_problem():
    result = solve_number_theory_problem("解同余方程组 x≡2 (mod 3), x≡3 (mod 5)")
    assert result is not None
    assert "8" in result["final_answer"]
    assert "15" in result["final_answer"]


def test_crt_no_solution_problem():
    result = solve_number_theory_problem("Solve the congruences x ≡ 1 (mod 2), x ≡ 0 (mod 4).")
    assert result is not None
    assert result["final_answer"] == "无解"


def test_english_multiplicative_order_problem():
    result = solve_number_theory_problem("Find the multiplicative order of 2 modulo 7.")
    assert result is not None
    assert result["final_answer"] == "3"
    assert result["details"]["problem_type"] == "multiplicative_order"


def test_chinese_multiplicative_order_problem():
    result = solve_number_theory_problem("求 2 模 9 的乘法阶")
    assert result is not None
    assert result["final_answer"] == "6"


def test_multiplicative_order_nonexistent_problem():
    result = solve_number_theory_problem("Find the multiplicative order of 2 modulo 4.")
    assert result is not None
    assert result["final_answer"] == "不存在"


def test_nonmatching_problem_returns_none():
    assert solve_number_theory_problem("How many subsets of a 5-element set have size 2?") is None


def test_single_congruence_does_not_trigger_crt():
    assert solve_number_theory_problem("Solve x ≡ 2 (mod 3).") is None


def test_group_order_does_not_trigger_multiplicative_order():
    assert solve_number_theory_problem("Find the order of finite group G.") is None
    assert solve_number_theory_problem("What is the order of group S_3?") is None


def test_number_theory_tool_zero_api_calls():
    from user_agent import ReasoningAgent

    class CountingClient:
        def __init__(self):
            self.call_count = 0

        def chat(self, **kwargs):
            self.call_count += 1
            return "This should not be called"

    client = CountingClient()
    agent = ReasoningAgent(client)

    result = agent.solve("Compute Euler's totient function phi(100).", {"answer_type": "number"})

    assert result["final_response"] == "40"
    assert client.call_count == 0
    assert "number_theory_tool" in str(result["trace"])
