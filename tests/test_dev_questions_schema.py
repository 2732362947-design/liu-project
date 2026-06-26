import json
from pathlib import Path


def test_dev_questions_schema():
    path = Path("data/dev_questions.json")

    assert path.exists()
    questions = json.loads(path.read_text(encoding="utf-8"))
    assert len(questions) >= 30

    problem_ids = [item.get("problem_id") for item in questions]
    assert len(problem_ids) == len(set(problem_ids))
    for item in questions:
        assert item.get("problem_id")
        assert item.get("problem")
        assert item.get("expected_answer")
        assert item.get("answer_type") in {"number", "expression", "set", "proof", "text"}
