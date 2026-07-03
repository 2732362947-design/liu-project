import json

from dev_tools.convert_omni_math import (
    convert_omni_math_file,
    convert_records,
    infer_answer_type,
    simplify_domain,
)


def test_simplify_domain_rules():
    assert simplify_domain(["Mathematics -> Geometry -> Plane Geometry"]) == "geometry"
    assert simplify_domain(["Mathematics -> Number Theory -> Primes"]) == "number_theory"
    assert simplify_domain(["Mathematics -> Complex Analysis -> Residues"]) == "complex_analysis"
    assert simplify_domain(["Mathematics -> Linear Algebra -> Eigenvalues"]) == "linear_algebra"
    assert simplify_domain(["Mathematics -> Something Else"]) == "unknown"


def test_simplify_unknown_usamon_diode_problem_as_combinatorics():
    problem = "Physicist USAMON studies a series of diode usage. Can electrons be in the same state?"

    assert simplify_domain(["Mathematics -> Other"], problem) in {"combinatorics", "discrete"}


def test_simplify_function_average_problem_as_real_analysis():
    problem = "Find all functions f: Z^2 -> [0,1] such that f(x,y) is the average of neighbors."

    assert simplify_domain(["Mathematics -> Other"], problem) == "real_analysis"


def test_simplify_round_table_transfer_cards_as_combinatorics_or_optimization():
    problem = "101 people sit at a round table and transfer cards to adjacent people. Find the smallest k."

    assert simplify_domain(["Mathematics -> Other"], problem) in {"combinatorics", "optimization"}


def test_simplify_extremal_subset_divisibility_not_optimization():
    problem = (
        "Find the smallest positive integer K such that every K-element subset of "
        "{1,2,...,50} contains two distinct elements a,b such that a+b divides ab."
    )

    assert simplify_domain(["Mathematics -> Optimization"], problem) in {
        "combinatorics",
        "number_theory",
        "graph_theory",
    }


def test_simplify_phone_digit_counting_as_combinatorics():
    problem = "How many phone numbers can be formed from these digits?"

    assert simplify_domain(["Mathematics -> Applied Mathematics -> Math Word Problems"], problem) == "combinatorics"


def test_simplify_speed_distance_word_problem_as_algebra():
    problem = "Two cars move at constant ground speed measured in km/hr. How many meters does the fly travel?"

    assert simplify_domain(["Mathematics -> Applied Mathematics -> Math Word Problems"], problem) == "algebra"


def test_simplify_payment_ways_as_combinatorics():
    problem = "He has an unlimited supply of 2, 5, and 10 dollar notes. In how many ways can he pay?"

    assert simplify_domain(["Mathematics -> Applied Mathematics -> Math Word Problems"], problem) == "combinatorics"


def test_simplify_statistics_as_probability():
    problem = "Find an unbiased estimator from independent and identically distributed observations."

    assert simplify_domain(["Mathematics -> Applied Mathematics -> Statistics"], problem) == "probability"


def test_infer_answer_type_rules_are_stable():
    assert infer_answer_type("1") == "number"
    assert infer_answer_type("1/2") == "number"
    assert infer_answer_type(r"\frac{1}{2}") == "expression"
    assert infer_answer_type("{1,2}") == "set"
    assert infer_answer_type("x^2+1") == "expression"
    assert infer_answer_type("The answer is...") == "text"


def test_convert_records_respects_max_per_domain():
    records = [
        {
            "domain": ["Mathematics -> Geometry"],
            "difficulty": 1,
            "problem": f"geometry problem {index}",
            "answer": str(index),
            "source": "fake",
        }
        for index in range(3)
    ]
    records.extend(
        [
            {
                "domain": ["Mathematics -> Number Theory"],
                "difficulty": 2,
                "problem": f"number theory problem {index}",
                "answer": "1/2",
                "source": "fake",
            }
            for index in range(3)
        ]
    )

    converted = convert_records(records, max_per_domain=2, max_total=90)

    assert len(converted) == 4
    assert sum(1 for item in converted if item["domain"] == "geometry") == 2
    assert sum(1 for item in converted if item["domain"] == "number_theory") == 2
    for item in converted:
        assert item["problem_id"].startswith("omni_")
        assert {"problem_id", "domain", "problem", "expected_answer", "answer_type"}.issubset(item)


def test_convert_records_preserves_raw_domain_difficulty_and_source():
    converted = convert_records(
        [
            {
                "domain": ["Mathematics -> Geometry -> Plane Geometry"],
                "difficulty": 4,
                "problem": "Find an angle.",
                "answer": "30",
                "source": "omni-test",
            }
        ]
    )

    item = converted[0]
    assert item["raw_domain"] == "Mathematics -> Geometry -> Plane Geometry"
    assert item["raw_domain_list"] == ["Mathematics -> Geometry -> Plane Geometry"]
    assert item["difficulty"] == 4
    assert item["source"] == "omni-test"


def test_convert_omni_math_file_writes_json(tmp_path):
    input_path = tmp_path / "omni.jsonl"
    output_path = tmp_path / "out" / "sample.json"
    rows = [
        {
            "domain": ["Mathematics -> Geometry"],
            "difficulty": 1,
            "problem": "Find an area.",
            "solution": "solution",
            "answer": "12",
            "source": "unit",
        },
        {
            "domain": ["Mathematics -> Geometry"],
            "difficulty": 2,
            "problem": "Find an angle.",
            "solution": "solution",
            "answer": "30",
            "source": "unit",
        },
        {
            "domain": ["Mathematics -> Geometry"],
            "difficulty": 3,
            "problem": "Find another angle.",
            "solution": "solution",
            "answer": "60",
            "source": "unit",
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    converted = convert_omni_math_file(input_path, output_path, max_per_domain=2, max_total=90)

    assert len(converted) == 2
    assert json.loads(output_path.read_text(encoding="utf-8")) == converted
