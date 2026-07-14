import json
from pathlib import Path

from agents.classifier_agent import ADVANCED_DOMAINS, classify_problem
from dev_tools.run_user_agent_real_smoke import _load_input_item, _load_input_items
from dev_tools.run_advanced_real_sanity import _classification_from_trace


ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "real_api_sanity_advanced.jsonl"
ANSWER_FIELDS = {
    "answer",
    "expected_answer",
    "gold_answer",
    "reference_answer",
    "solution",
    "gold",
    "reference",
    "ground_truth",
    "expected",
    "expected_solution",
    "official_answer",
    "label",
    "target",
}


def test_advanced_real_sanity_jsonl_has_one_safe_item_per_domain():
    items = _load_input_items(DATA_FILE)

    assert len(items) == len(ADVANCED_DOMAINS)
    assert {item["domain"] for item in items} == set(ADVANCED_DOMAINS)
    for item in items:
        assert item["problem"].strip()
        assert not ANSWER_FIELDS.intersection(item)
        json.dumps(item, ensure_ascii=False)


def test_advanced_real_sanity_items_are_runner_compatible_and_route_correctly():
    for index, expected_domain in enumerate(ADVANCED_DOMAINS):
        problem, idx, metadata = _load_input_item(DATA_FILE, index)
        classification = classify_problem(problem)

        assert idx.startswith("advanced_")
        assert metadata["domain"] == expected_domain
        assert metadata["subject"]
        assert classification["domain"] == expected_domain
        assert classification["solver_key"] == expected_domain


def test_advanced_real_sanity_summary_extracts_routing_without_api_call():
    trace = [{"step": "classify", "content": "domain=statistics, solver_key=statistics"}]

    assert _classification_from_trace(trace) == ("statistics", "statistics")
