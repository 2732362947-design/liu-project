import argparse
import json
import re
import time
from pathlib import Path

try:
    from dev_tools.run_user_agent_real_smoke import _build_client_from_env, _load_input_item, _load_input_items, run_smoke
except ModuleNotFoundError:
    from run_user_agent_real_smoke import _build_client_from_env, _load_input_item, _load_input_items, run_smoke


ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = ROOT / "data" / "real_api_sanity_advanced.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "real_api_sanity_advanced_results.json"


def _classification_from_trace(trace: list[dict]) -> tuple[str, str]:
    for item in trace:
        if not isinstance(item, dict) or item.get("step") != "classify":
            continue
        content = str(item.get("content") or "")
        match = re.search(r"domain=([^,]+),\s*solver_key=([^,\s]+)", content)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return "unknown", "general"


def run_advanced_sanity(
    *,
    input_path: str | Path,
    output_path: str | Path,
    timeout: float,
    sleep_seconds: float,
) -> list[dict]:
    items = _load_input_items(input_path)
    client = _build_client_from_env(timeout)
    seen_domains = set()
    summaries = []
    for index, item in enumerate(items):
        problem, idx, metadata = _load_input_item(input_path, index)
        declared_domain = str(metadata.get("domain") or metadata.get("subject") or "unknown")
        if declared_domain in seen_domains:
            continue
        seen_domains.add(declared_domain)

        result = run_smoke(client, problem=problem, idx=idx, metadata=metadata)
        domain, solver_key = _classification_from_trace(result.get("trace", []))
        summary = {
            "idx": idx,
            "domain": domain,
            "solver_key": solver_key,
            "final_response_nonempty": bool(str(result.get("final_response") or "").strip()),
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
        if sleep_seconds > 0 and index + 1 < len(items):
            time.sleep(sleep_seconds)

    destination = Path(output_path)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential one-question-per-domain advanced real API sanity check.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT.relative_to(ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(ROOT)))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--sleep", type=float, default=0)
    parser.add_argument("--concurrency", type=int, choices=(1,), default=1)
    parser.add_argument("--limit-per-domain", type=int, choices=(1,), default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_advanced_sanity(
        input_path=args.input,
        output_path=args.output,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
