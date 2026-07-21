"""Prepare reproducible, disjoint Omni-MATH evaluation splits."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev_tools.convert_omni_math import infer_answer_type, simplify_domain


DEFAULT_SOURCE = (
    Path.home()
    / ".cache"
    / "modelscope"
    / "hub"
    / "datasets"
    / "AI-ModelScope"
    / "Omni-MATH"
    / "test.jsonl"
)
DEFAULT_OUTPUT = ROOT / "evaluation" / "datasets" / "omni_math_eval_250.jsonl"
PRESERVED_FIELDS = ("solution", "subject", "category", "difficulty")
PROOF_MARKERS = (
    "prove",
    "proof",
    "show that",
    "demonstrate that",
    "derive",
    "explain why",
    "证明",
    "推导",
    "解释",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def discover_source() -> Path:
    """Return the best local Omni-MATH source without downloading anything."""
    candidates: list[Path] = []
    if DEFAULT_SOURCE.is_file():
        candidates.append(DEFAULT_SOURCE)
    for pattern in ("**/*omni*.jsonl", "**/*omni*.json", "**/*omni*.parquet"):
        candidates.extend(path for path in ROOT.glob(pattern) if path.is_file())
    if not candidates:
        raise FileNotFoundError("no local Omni-MATH source found; pass --source explicitly")
    return max(candidates, key=lambda path: path.stat().st_size)


def load_source(path: str | Path) -> list[dict[str, Any]]:
    source = _resolve(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("items", []))
        rows = payload
    elif suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional local format support
            raise RuntimeError("reading parquet requires pandas and a parquet engine") from exc
        rows = pd.read_parquet(source).to_dict(orient="records")
    else:
        raise ValueError(f"unsupported source format: {source.suffix}")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Omni-MATH source must contain JSON objects")
    return rows


def _source_idx(item: dict[str, Any], position: int) -> str:
    for key in ("source_idx", "id", "idx", "problem_id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(position)


def _is_suspected_incorrect(problem: str) -> bool:
    normalized = problem.lower().replace("\\plus{}", "+")
    normalized = re.sub(r"[\s$\\{}]", "", normalized)
    return (
        "1,2,...,50" in normalized
        and "k-elementsubset" in normalized
        and "a+bdividesab" in normalized
    )


def _response_mode(problem: str, answer_type: str) -> str:
    lower = problem.lower()
    if answer_type == "proof" or any(marker in lower for marker in PROOF_MARKERS):
        return "worked_solution"
    return "short_answer"


def normalize_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize records and remove duplicate source identifiers."""
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, item in enumerate(records):
        source_idx = _source_idx(item, position)
        if source_idx in seen:
            continue
        seen.add(source_idx)
        problem = str(item.get("problem") or item.get("question") or "").strip()
        expected = item.get("expected_answer", item.get("answer", ""))
        expected_answer = str(expected or "").strip()
        if not problem:
            continue
        domain_value = item.get("domain", item.get("subject", item.get("category")))
        expected_domain = simplify_domain(domain_value, problem)
        answer_type = str(item.get("answer_type") or infer_answer_type(expected_answer, problem))
        suspected = _is_suspected_incorrect(problem)
        row: dict[str, Any] = {
            "idx": f"omni_eval_{position + 1:06d}",
            "problem": problem,
            "expected_answer": expected_answer,
            "expected_domain": expected_domain,
            "answer_type": answer_type or "unknown",
            "response_mode": _response_mode(problem, answer_type),
            "source": "omni_math",
            "source_idx": source_idx,
            "label_status": "suspected_incorrect" if suspected else "unreviewed",
            "review_note": (
                "Dataset answer may be inconsistent with the problem; independently computed result was 39 rather than 26."
                if suspected
                else ""
            ),
        }
        for key in PRESERVED_FIELDS:
            if item.get(key) is not None:
                row[key] = item[key]
        if domain_value is not None:
            row["raw_domain"] = domain_value
        if item.get("source") is not None:
            row["original_source"] = item["source"]
        normalized.append(row)
    return normalized


def _allocate_quotas(groups: dict[str, list[dict[str, Any]]], size: int) -> dict[str, int]:
    total = sum(len(rows) for rows in groups.values())
    size = min(size, total)
    if not size or not total:
        return {key: 0 for key in groups}
    exact = {key: size * len(rows) / total for key, rows in groups.items()}
    quotas = {key: min(len(groups[key]), int(value)) for key, value in exact.items()}
    remaining = size - sum(quotas.values())
    order = sorted(groups, key=lambda key: (exact[key] - quotas[key], len(groups[key]), key), reverse=True)
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:  # pragma: no cover - defensive capacity guard
            break
    return quotas


def stratified_take(
    rows: list[dict[str, Any]], size: int, rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("expected_domain") or "unknown")].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)
    quotas = _allocate_quotas(groups, size)
    selected: list[dict[str, Any]] = []
    remainder: list[dict[str, Any]] = []
    for key in sorted(groups):
        quota = quotas[key]
        selected.extend(groups[key][:quota])
        remainder.extend(groups[key][quota:])
    rng.shuffle(selected)
    rng.shuffle(remainder)
    return selected, remainder


def split_records(
    rows: list[dict[str, Any]],
    sample_size: int = 250,
    smoke_size: int = 30,
    holdout_size: int = 50,
    seed: int = 20260720,
) -> dict[str, list[dict[str, Any]]]:
    """Split into smoke plus a sample_size pool of main and holdout rows."""
    if min(sample_size, smoke_size, holdout_size) < 0 or holdout_size > sample_size:
        raise ValueError("sizes must be non-negative and holdout-size cannot exceed sample-size")
    available = len(rows)
    requested_total = sample_size + smoke_size
    if available < requested_total:
        smoke_size = round(available * 0.10)
        main_size = round(available * 0.70)
        holdout_size = available - smoke_size - main_size
    else:
        main_size = sample_size - holdout_size
    rng = random.Random(seed)
    # Keep known label conflicts in smoke so they are visible early and never
    # silently contaminate main-set automatic accuracy.
    priority = [row for row in rows if row.get("label_status") == "suspected_incorrect"][:smoke_size]
    priority_ids = {str(row.get("source_idx")) for row in priority}
    candidates = [row for row in rows if str(row.get("source_idx")) not in priority_ids]
    smoke_fill, remainder = stratified_take(candidates, smoke_size - len(priority), rng)
    smoke = priority + smoke_fill
    rng.shuffle(smoke)
    main, remainder = stratified_take(remainder, main_size, rng)
    holdout, _ = stratified_take(remainder, holdout_size, rng)
    return {"smoke": smoke, "main": main, "holdout": holdout}


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = _resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_evaluation(
    source: str | Path,
    output: str | Path = DEFAULT_OUTPUT,
    sample_size: int = 250,
    seed: int = 20260720,
    smoke_size: int = 30,
    holdout_size: int = 50,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Path]]:
    rows = normalize_records(load_source(source))
    splits = split_records(rows, sample_size, smoke_size, holdout_size, seed)
    output_path = _resolve(output)
    paths = {
        "combined": output_path,
        "smoke": output_path.parent / f"omni_math_smoke_{len(splits['smoke'])}.jsonl",
        "main": output_path.parent / f"omni_math_main_{len(splits['main'])}.jsonl",
        "holdout": output_path.parent / f"omni_math_holdout_{len(splits['holdout'])}.jsonl",
    }
    write_jsonl(paths["combined"], splits["main"] + splits["holdout"])
    for name in ("smoke", "main", "holdout"):
        write_jsonl(paths[name], splits[name])
    return splits, paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="Local Omni-MATH JSON, JSONL, or parquet source")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(ROOT)))
    parser.add_argument("--sample-size", type=int, default=250, help="Main plus holdout size")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--smoke-size", type=int, default=30)
    parser.add_argument("--holdout-size", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = _resolve(args.source) if args.source else discover_source()
    source_rows = load_source(source)
    fields = sorted(set().union(*(row.keys() for row in source_rows))) if source_rows else []
    splits, paths = prepare_evaluation(
        source,
        args.output,
        args.sample_size,
        args.seed,
        args.smoke_size,
        args.holdout_size,
    )
    print(f"source={source}")
    print(f"source_fields={fields}")
    print(f"source_count={len(source_rows)}")
    for name in ("smoke", "main", "holdout"):
        counts = Counter(row["expected_domain"] for row in splits[name])
        print(f"{name}={len(splits[name])} path={paths[name]} domains={dict(sorted(counts.items()))}")
    print(f"combined={paths['combined']} count={len(splits['main']) + len(splits['holdout'])}")


if __name__ == "__main__":
    main()
