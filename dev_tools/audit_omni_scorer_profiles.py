"""Reproduce the frozen non-holdout scorer-profile counterfactual offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev_tools.score_omni_results import score_answer


MAIN_DATASET = ROOT / "evaluation/datasets/omni_math_main_200.jsonl"
MAIN_RESULTS = ROOT / "evaluation/results/omni_math_main_200_results.jsonl"
MAIN_BASELINE = ROOT / "evaluation/results/omni_math_main_200_results_enriched.jsonl"
MAIN_AUDIT = ROOT / "evaluation/reports/omni_math_main_200_scoring_sensitivity.json"

AB_DATASET = ROOT / "evaluation/datasets/omni_math_solver_ablation_60.jsonl"
AB_RESULTS = ROOT / "evaluation/results/omni_math_solver_ablation_results.jsonl"
AB_BASELINE = ROOT / "evaluation/results/omni_math_solver_ablation_enriched.jsonl"
AB_AUDIT = ROOT / "evaluation/reports/omni_math_solver_ablation_sensitivity_report.json"

PNT_DATASET = (
    ROOT / "evaluation/experiments/omni_math_probability_number_theory_expansion_dev_65.jsonl"
)
PNT_RESULTS = (
    ROOT
    / "evaluation/experiments/"
    "omni_pnt_expansion_dev65_paired_baseline_v1_20260722_results.jsonl"
)
PNT_BASELINE = (
    ROOT
    / "evaluation/experiments/"
    "omni_pnt_expansion_dev65_paired_baseline_v1_20260722_enriched.jsonl"
)
PNT_NORMALIZATION_AUDIT = (
    ROOT
    / "evaluation/reports/"
    "omni_pnt_expansion_dev65_scorer_normalization_v2_20260728_comparison.json"
)
PNT_REPLAY = (
    ROOT
    / "evaluation/experiments/"
    "omni_pnt_expansion_dev65_extractor_syntax_v1_20260728_replayed.jsonl"
)
PNT_CLOSED_AUDIT = (
    ROOT
    / "evaluation/reports/"
    "omni_pnt_expansion_dev65_scorer_closed_expr_v3_20260728_comparison.json"
)

INPUT_PATHS = (
    MAIN_DATASET,
    MAIN_RESULTS,
    MAIN_BASELINE,
    MAIN_AUDIT,
    AB_DATASET,
    AB_RESULTS,
    AB_BASELINE,
    AB_AUDIT,
    PNT_DATASET,
    PNT_RESULTS,
    PNT_BASELINE,
    PNT_NORMALIZATION_AUDIT,
    PNT_REPLAY,
    PNT_CLOSED_AUDIT,
)
TASK_FILES = (
    "dev_tools/score_omni_results.py",
    "dev_tools/audit_omni_scorer_profiles.py",
    "tests/test_omni_unified_scorer_v4.py",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if "holdout" in path.name.lower():
        raise RuntimeError(f"holdout input is forbidden: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if "holdout" in path.name.lower():
        raise RuntimeError(f"holdout input is forbidden: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not an object")
    return value


def _arm(value: Any) -> str:
    return str(value or "single").replace("oracle_", "")


def _question_hash(problem: Any) -> str:
    normalized = " ".join(str(problem or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _key(idx: Any, arm: Any = None) -> tuple[str, str]:
    return str(idx or ""), _arm(arm)


def _map(
    rows: list[dict[str, Any]],
    *,
    paired: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    mapped = {
        _key(row.get("idx"), row.get("arm") if paired else None): row
        for row in rows
    }
    if len(mapped) != len(rows):
        raise RuntimeError("duplicate scorer input key")
    return mapped


def _audit_expected_changes() -> dict[str, set[tuple[str, str]]]:
    main = {
        _key(row.get("idx"))
        for row in _load_json(MAIN_AUDIT)["changes"]
        if row.get("raw_score_status") != "correct"
        and row.get("corrected_score_status") == "correct"
    }
    ab_exact: set[tuple[str, str]] = set()
    ab_policy: set[tuple[str, str]] = set()
    for row in _load_json(AB_AUDIT)["changes"]:
        if (
            row.get("raw_score_status") == "correct"
            or row.get("corrected_score_status") != "correct"
        ):
            continue
        target = (
            ab_policy
            if row.get("reason_code") == "open_constraint_validated"
            else ab_exact
        )
        target.add(_key(row.get("idx"), row.get("arm")))
    normalization = {
        _key(row.get("idx"), row.get("arm"))
        for row in _load_json(PNT_NORMALIZATION_AUDIT)["score_changes"]
    }
    closed = {
        _key(row.get("idx"), row.get("arm"))
        for row in _load_json(PNT_CLOSED_AUDIT)["score_changed"]
    }
    return {
        "main": main,
        "ab_exact": ab_exact,
        "ab_policy": ab_policy,
        "normalization": normalization,
        "closed_expression": closed,
    }


def _raw_stages() -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []

    main_items = _map(_load_jsonl(MAIN_DATASET), paired=False)
    main_results = _map(_load_jsonl(MAIN_RESULTS), paired=False)
    main_baseline = _map(_load_jsonl(MAIN_BASELINE), paired=False)
    stages.append(
        {
            "group": "main",
            "dataset": "main_200",
            "rows": [
                (
                    key,
                    item,
                    main_results[key].get("final_response"),
                    main_baseline[key].get("score_status") == "correct",
                    "legacy",
                )
                for key, item in main_items.items()
            ],
        }
    )

    ab_items = _map(_load_jsonl(AB_DATASET), paired=False)
    ab_results = _map(_load_jsonl(AB_RESULTS), paired=True)
    ab_baseline = _map(_load_jsonl(AB_BASELINE), paired=True)
    stages.append(
        {
            "group": "ab",
            "dataset": "solver_ablation_60",
            "rows": [
                (
                    key,
                    ab_items[_key(key[0])],
                    result.get("final_response"),
                    ab_baseline[key].get("correct") is True,
                    "legacy",
                )
                for key, result in ab_results.items()
            ],
        }
    )

    pnt_items = _map(_load_jsonl(PNT_DATASET), paired=False)
    pnt_results = _map(_load_jsonl(PNT_RESULTS), paired=True)
    pnt_baseline = _map(_load_jsonl(PNT_BASELINE), paired=True)
    stages.append(
        {
            "group": "normalization",
            "dataset": "pnt_dev65",
            "rows": [
                (
                    key,
                    pnt_items[_key(key[0])],
                    result.get("final_response"),
                    pnt_baseline[key].get("correct") is True,
                    "legacy",
                )
                for key, result in pnt_results.items()
            ],
        }
    )

    replay = _map(_load_jsonl(PNT_REPLAY), paired=True)
    stages.append(
        {
            "group": "closed_expression",
            "dataset": "pnt_dev65",
            "rows": [
                (
                    key,
                    pnt_items[_key(key[0])],
                    row.get("new_selected_candidate"),
                    row.get("new_score_v2") is True,
                    "normalization_v2",
                )
                for key, row in replay.items()
            ],
        }
    )
    return stages


def _dimension_counts(
    changes: list[dict[str, Any]],
    field: str,
    labels: set[str],
) -> dict[str, dict[str, int]]:
    return {
        label: {
            "exact_rescued": sum(
                row[field] == label and row["classification"] == "exact"
                for row in changes
            ),
            "policy_only_rescued": sum(
                row[field] == label and row["classification"] == "policy"
                for row in changes
            ),
            "harmed": 0,
            "net": sum(row[field] == label for row in changes),
        }
        for label in sorted(labels)
    }


def build_report() -> dict[str, Any]:
    if any("holdout" in path.name.lower() for path in INPUT_PATHS):
        raise RuntimeError("holdout path entered the audit input list")
    expected = _audit_expected_changes()
    changes: list[dict[str, Any]] = []
    harms: list[dict[str, Any]] = []
    legacy_checks = 0
    legacy_mismatches: list[dict[str, Any]] = []
    stage_rows: dict[str, int] = {}
    dimension_labels: dict[str, set[str]] = {
        "dataset": set(),
        "arm": set(),
        "domain": set(),
    }

    for stage in _raw_stages():
        group = stage["group"]
        stage_rows[group] = len(stage["rows"])
        dimension_labels["dataset"].add(stage["dataset"])
        actual_exact: set[tuple[str, str]] = set()
        actual_policy: set[tuple[str, str]] = set()
        for key, item, prediction, baseline_correct, baseline_profile in stage["rows"]:
            dimension_labels["arm"].add(key[1])
            dimension_labels["domain"].add(
                str(item.get("expected_domain") or "unknown")
            )
            legacy = score_answer(prediction, item.get("expected_answer"))
            exact = score_answer(
                prediction,
                item.get("expected_answer"),
                profile="exact_v4",
                problem_text=item.get("problem"),
            )
            policy = score_answer(
                prediction,
                item.get("expected_answer"),
                profile="policy_v4",
                problem_text=item.get("problem"),
            )
            if baseline_profile == "legacy":
                legacy_checks += 1
                if (legacy["status"] == "correct") != baseline_correct:
                    legacy_mismatches.append(
                        {"group": group, "record": list(key)}
                    )
            if baseline_correct and exact["status"] != "correct":
                harms.append(
                    {
                        "group": group,
                        "record": list(key),
                        "profile": "exact_v4",
                        "reason_code": exact["reason_code"],
                    }
                )
            if baseline_correct and policy["status"] != "correct":
                harms.append(
                    {
                        "group": group,
                        "record": list(key),
                        "profile": "policy_v4",
                        "reason_code": policy["reason_code"],
                    }
                )
            classification: str | None = None
            if not baseline_correct and exact["status"] == "correct":
                actual_exact.add(key)
                classification = "exact"
            elif (
                not baseline_correct
                and exact["status"] != "correct"
                and policy["status"] == "correct"
            ):
                actual_policy.add(key)
                classification = "policy"
            if classification is not None:
                changes.append(
                    {
                        "group": group,
                        "dataset": stage["dataset"],
                        "record": list(key),
                        "arm": key[1],
                        "domain": str(item.get("expected_domain") or "unknown"),
                        "question_sha256": _question_hash(item.get("problem")),
                        "classification": classification,
                        "reason_code": (
                            exact["reason_code"]
                            if classification == "exact"
                            else policy["reason_code"]
                        ),
                    }
                )
        expected_exact = (
            expected["ab_exact"] if group == "ab" else expected[group]
        )
        expected_policy = expected["ab_policy"] if group == "ab" else set()
        if actual_exact != expected_exact:
            raise RuntimeError(
                f"{group} exact changes differ: "
                f"actual={sorted(actual_exact)!r}, expected={sorted(expected_exact)!r}"
            )
        if actual_policy != expected_policy:
            raise RuntimeError(
                f"{group} policy changes differ: "
                f"actual={sorted(actual_policy)!r}, expected={sorted(expected_policy)!r}"
            )

    if harms:
        raise RuntimeError(f"scorer harm detected: {harms!r}")
    if legacy_mismatches:
        raise RuntimeError(f"legacy compatibility mismatch: {legacy_mismatches!r}")

    group_hashes = {
        group: {
            row["question_sha256"]
            for row in changes
            if row["group"] == group
        }
        for group in ("main", "ab", "normalization", "closed_expression")
    }
    intersections = {
        f"{first}__{second}": sorted(group_hashes[first] & group_hashes[second])
        for position, first in enumerate(group_hashes)
        for second in list(group_hashes)[position + 1 :]
    }
    exact_changes = [row for row in changes if row["classification"] == "exact"]
    policy_changes = [row for row in changes if row["classification"] == "policy"]
    exact_hashes = {row["question_sha256"] for row in exact_changes}
    policy_hashes = {row["question_sha256"] for row in policy_changes}
    all_hashes = exact_hashes | policy_hashes

    duplicate_evidence = {
        "ab": [
            row
            for row in changes
            if row["group"] == "ab"
            and sum(
                other["group"] == "ab"
                and other["question_sha256"] == row["question_sha256"]
                for other in changes
            )
            > 1
        ],
        "closed_expression": [
            row
            for row in changes
            if row["group"] == "closed_expression"
        ],
    }
    if (
        len(changes) != 15
        or len(all_hashes) != 13
        or len(exact_changes) != 13
        or len(exact_hashes) != 12
        or len(policy_changes) != 2
        or len(policy_hashes) != 1
        or any(intersections.values())
        or len(duplicate_evidence["ab"]) != 2
        or len(duplicate_evidence["closed_expression"]) != 2
    ):
        raise RuntimeError("frozen dual-counting invariants did not reproduce")

    group_arm_counts = Counter(row["group"] for row in changes)
    group_unique_counts = {
        group: len(hashes) for group, hashes in group_hashes.items()
    }
    return {
        "frozen_counts": {
            "arm_change_total": len(changes),
            "unique_question_total": len(all_hashes),
            "group_arm_counts": dict(sorted(group_arm_counts.items())),
            "group_unique_question_counts": group_unique_counts,
            "exact_arm_rescued": len(exact_changes),
            "exact_unique_rescued": len(exact_hashes),
            "policy_only_arm_rescued": len(policy_changes),
            "policy_only_unique_rescued": len(policy_hashes),
            "harmed": 0,
        },
        "stage_rows_scanned": stage_rows,
        "arm_level": {
            "exact_v4": {
                "rescued": len(exact_changes),
                "harmed": 0,
                "net": len(exact_changes),
            },
            "policy_v4": {
                "rescued": len(changes),
                "harmed": 0,
                "net": len(changes),
            },
        },
        "unique_question_level": {
            "exact_v4": {
                "rescued": len(exact_hashes),
                "harmed": 0,
                "net": len(exact_hashes),
            },
            "policy_v4": {
                "rescued": len(all_hashes),
                "harmed": 0,
                "net": len(all_hashes),
            },
        },
        "group_question_intersections": intersections,
        "duplicate_arm_evidence": duplicate_evidence,
        "changes": changes,
        "by_dataset": _dimension_counts(
            changes,
            "dataset",
            dimension_labels["dataset"],
        ),
        "by_arm": _dimension_counts(changes, "arm", dimension_labels["arm"]),
        "by_domain": _dimension_counts(
            changes,
            "domain",
            dimension_labels["domain"],
        ),
        "legacy_compatibility": {
            "raw_non_holdout_rows_checked": legacy_checks,
            "mismatches": len(legacy_mismatches),
            "compatible": not legacy_mismatches,
            "closed_expression_baseline": "normalization_v2",
        },
        "false_positive_audit": {
            "unexpected_rescues": 0,
            "expected_change_sets_matched": True,
            "invalid_results": 0,
            "exceptions": 0,
        },
        "scope": {
            "holdout_read": False,
            "model_or_api_calls": False,
            "runtime_modified": False,
            "committed": False,
            "pushed": False,
            "staged_by_task": False,
            "input_paths": [str(path.relative_to(ROOT)) for path in INPUT_PATHS],
            "task_files": list(TASK_FILES),
        },
    }


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip()


def render_markdown(report: dict[str, Any], validation: dict[str, str]) -> str:
    frozen = report["frozen_counts"]
    lines = [
        "# Omni-MATH unified offline scorer v4 counterfactual",
        "",
        "## Outcome",
        "",
        f"- Arm-level changes: {frozen['arm_change_total']} rescued, 0 harmed, "
        f"net +{frozen['arm_change_total']} under policy_v4.",
        f"- Deduplicated questions: {frozen['unique_question_total']} rescued, "
        f"0 harmed, net +{frozen['unique_question_total']}.",
        f"- exact_v4: {frozen['exact_arm_rescued']} arm rescues, "
        f"{frozen['exact_unique_rescued']} unique-question rescues.",
        f"- policy-only: {frozen['policy_only_arm_rescued']} arm rescues, "
        f"{frozen['policy_only_unique_rescued']} unique-question rescue.",
        "- All 15 frozen arm changes reproduced; all 13 unique questions attributed.",
        "- No scorer harm or unexpected rescue was observed.",
        "",
        "## Dual counting",
        "",
        "| Group | Arm changes | Unique questions |",
        "| --- | ---: | ---: |",
    ]
    for group in ("main", "ab", "normalization", "closed_expression"):
        lines.append(
            f"| {group} | {frozen['group_arm_counts'][group]} | "
            f"{frozen['group_unique_question_counts'][group]} |"
        )
    lines.extend(
        [
            "",
            "A/B is 5→4 because the policy-only question has independent general "
            "and specialized arm rows. closed-expression is 2→1 because the same "
            "question has independent general and specialized arm rows. All six "
            "cross-group question-hash intersections are empty.",
            "",
            "The P/NT counterfactual is staged: normalization v2 compares the "
            "original saved final responses with legacy; closed-expression v3 "
            "compares every extractor-replay candidate with its frozen v2 score. "
            "This prevents an extractor-only change from being counted as a "
            "scorer-only rescue.",
            "",
            "## Arm-level and unique-question totals",
            "",
            "| Profile | Arm rescued | Arm harmed | Arm net | Unique rescued | "
            "Unique harmed | Unique net |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for profile in ("exact_v4", "policy_v4"):
        arm = report["arm_level"][profile]
        unique = report["unique_question_level"][profile]
        lines.append(
            f"| {profile} | {arm['rescued']} | {arm['harmed']} | {arm['net']} | "
            f"{unique['rescued']} | {unique['harmed']} | {unique['net']} |"
        )
    for heading, field in (
        ("Dataset/stage", "by_dataset"),
        ("Arm", "by_arm"),
        ("Domain", "by_domain"),
    ):
        lines.extend(
            [
                "",
                f"## By {heading.lower()}",
                "",
                f"| {heading} | Exact rescued | Policy-only rescued | Harmed | Net |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, counts in report[field].items():
            lines.append(
                f"| {label} | {counts['exact_rescued']} | "
                f"{counts['policy_only_rescued']} | {counts['harmed']} | "
                f"{counts['net']} |"
            )
    compatibility = report["legacy_compatibility"]
    lines.extend(
        [
            "",
            "## Compatibility and safety",
            "",
            f"- legacy compatibility: {compatibility['raw_non_holdout_rows_checked']}"
            f"/{compatibility['raw_non_holdout_rows_checked']} raw non-holdout rows "
            "match their historical correct/not-correct decisions.",
            "- closed-expression replay uses normalization v2 as its sequential "
            "baseline; all pre-existing correct records remain correct.",
            "- policy hits have `equivalence_class=policy` and an independent "
            "reason code; they do not enter exact metrics.",
            "- Candidate extraction accepts only the prediction and never the "
            "expected answer. exact_v4 does not inspect problem text.",
            "- No holdout was read. No model or API was called. No runtime file was "
            "modified. No commit, push, or staging action was performed.",
            "",
            "## Validation",
            "",
        ]
    )
    for label, result in validation.items():
        lines.append(f"- {label}: {result}")
    lines.extend(
        [
            "",
            "## Files and repository state",
            "",
            "Task files:",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in report["scope"]["task_files"])
    lines.extend(
        [
            "",
            "Current `git status --short`:",
            "",
            "```text",
            report["repository"]["git_status_short"],
            "```",
            "",
            f"HEAD: `{report['repository']['head']}`",
            "",
            "Staged changes:",
            "",
            "```text",
            report["repository"]["staged_name_status"] or "(empty)",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default="/tmp/omni_scorer_deduplicated_reproduction.json",
    )
    parser.add_argument(
        "--output-md",
        default="/tmp/omni_scorer_offline_audit.md",
    )
    parser.add_argument("--targeted-tests", default="not yet recorded")
    parser.add_argument("--full-tests", default="not yet recorded")
    parser.add_argument("--submission-check", default="not yet recorded")
    parser.add_argument("--clean-environment", default="not yet recorded")
    parser.add_argument("--compileall", default="not yet recorded")
    parser.add_argument("--diff-check", default="not yet recorded")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = (Path(args.output_json).resolve(), Path(args.output_md).resolve())
    if any(path.parent != Path("/tmp") for path in outputs):
        raise ValueError("audit outputs must be direct children of /tmp")
    if set(outputs) & {path.resolve() for path in INPUT_PATHS}:
        raise ValueError("audit output would overwrite an input")

    report = build_report()
    report["repository"] = {
        "head": _git_output("rev-parse", "HEAD"),
        "tree": _git_output("rev-parse", "HEAD^{tree}"),
        "git_status_short": _git_output("status", "--short"),
        "staged_name_status": _git_output("diff", "--cached", "--name-status"),
    }
    validation = {
        "targeted tests": args.targeted_tests,
        "full tests": args.full_tests,
        "submission readiness": args.submission_check,
        "clean environment": args.clean_environment,
        "compileall dev_tools": args.compileall,
        "git diff --check": args.diff_check,
    }
    report["validation"] = validation
    outputs[0].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs[1].write_text(render_markdown(report, validation), encoding="utf-8")
    print(json.dumps(report["frozen_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
