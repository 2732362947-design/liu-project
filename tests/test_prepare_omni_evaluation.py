import json

from dev_tools.prepare_omni_evaluation import normalize_records, prepare_evaluation, split_records


def _records(count=60):
    domains = ["Algebra", "Number Theory", "Geometry", "Combinatorics"]
    return [
        {
            "idx": f"raw-{index}",
            "problem": f"Problem {index}",
            "answer": str(index),
            "domain": [f"Mathematics -> {domains[index % len(domains)]}"],
            "solution": f"Solution {index}",
            "difficulty": index % 10,
        }
        for index in range(count)
    ]


def test_split_is_disjoint_and_reproducible():
    rows = normalize_records(_records())
    first = split_records(rows, sample_size=30, smoke_size=10, holdout_size=5, seed=42)
    second = split_records(rows, sample_size=30, smoke_size=10, holdout_size=5, seed=42)

    assert first == second
    sets = [{row["source_idx"] for row in first[name]} for name in ("smoke", "main", "holdout")]
    assert len(sets[0]) == 10
    assert len(sets[1]) == 25
    assert len(sets[2]) == 5
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])


def test_normalize_deduplicates_source_idx_and_preserves_offline_solution():
    records = _records(3) + [{**_records(1)[0], "problem": "duplicate"}]
    rows = normalize_records(records)

    assert len(rows) == 3
    assert len({row["source_idx"] for row in rows}) == 3
    assert rows[0]["solution"] == "Solution 0"
    assert rows[0]["expected_answer"] == "0"


def test_small_source_uses_rough_10_70_20_ratio():
    splits = split_records(normalize_records(_records(20)), sample_size=250, smoke_size=30, holdout_size=50)

    assert {name: len(rows) for name, rows in splits.items()} == {"smoke": 2, "main": 14, "holdout": 4}


def test_known_disputed_problem_is_marked():
    problem = (
        "Find the smallest positive integer K such that every K-element subset of "
        "{1,2,...,50} contains two distinct elements a,b such that a+b divides ab."
    )
    row = normalize_records([{"problem": problem, "answer": "26", "domain": ["Combinatorics"]}])[0]

    assert row["label_status"] == "suspected_incorrect"
    assert "39 rather than 26" in row["review_note"]


def test_prepare_writes_combined_and_three_split_files(tmp_path):
    source = tmp_path / "omni.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in _records()), encoding="utf-8")

    splits, paths = prepare_evaluation(
        source, tmp_path / "omni_math_eval_30.jsonl", sample_size=30, smoke_size=10, holdout_size=5, seed=7
    )

    assert all(path.is_file() for path in paths.values())
    combined = paths["combined"].read_text(encoding="utf-8").splitlines()
    assert len(combined) == len(splits["main"]) + len(splits["holdout"]) == 30

