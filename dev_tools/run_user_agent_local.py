import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT_DIR = ROOT / "outputs_user_agent"
DEFAULT_FAKE_ANSWER = "最终答案：2"


class FakeClient:
    def __init__(self, answer: str = DEFAULT_FAKE_ANSWER):
        self.answer = answer
        self.calls = []

    def chat(self, messages, temperature=0.2, max_tokens=4096):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.answer


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with _resolve_path(path).open(encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if isinstance(item, dict):
                items.append(item)
    return items


def safe_error_message(exc: Exception, max_chars: int = 500) -> str:
    message = f"{exc}"
    for marker in ("Authorization", "Bearer", "api_key", "token"):
        message = message.replace(marker, "[redacted]")
    return message[:max_chars]


def normalize_agent_response(response: Any) -> dict:
    if isinstance(response, dict):
        normalized = dict(response)
        normalized["final_response"] = str(normalized.get("final_response", ""))
        trace = normalized.get("trace", [])
        normalized["trace"] = trace if isinstance(trace, list) else []
        return normalized
    return {"final_response": str(response or ""), "trace": []}


def run_one(agent, item: dict, idx: int) -> dict:
    started = time.perf_counter()
    try:
        problem = str(item.get("problem", ""))
        metadata = dict(item)
        metadata.setdefault("idx", idx)
        response = normalize_agent_response(agent.solve(problem, metadata))
        final_response = response.get("final_response", "")
        return {
            "idx": metadata["idx"],
            "status": "success",
            "final_response": str(final_response),
            "trace": response.get("trace", []),
            "time_cost_seconds": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:
        return {
            "idx": item.get("idx", idx),
            "status": "error",
            "final_response": "",
            "error": {
                "type": type(exc).__name__,
                "message": safe_error_message(exc),
            },
            "trace": [],
            "time_cost_seconds": round(time.perf_counter() - started, 6),
        }


def write_result(path: str | Path, result: dict) -> None:
    output_file = _resolve_path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def run_local(
    input_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    overwrite: bool = False,
    fake_answer: str = DEFAULT_FAKE_ANSWER,
) -> list[dict]:
    from user_agent import ReasoningAgent

    output_root = _resolve_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    agent = ReasoningAgent(client=FakeClient(fake_answer))
    results = []
    items = load_jsonl(input_path)
    if limit is not None:
        items = items[:limit]

    for line_index, item in enumerate(items):
        idx = item.get("idx", line_index)
        output_file = output_root / f"{idx}.json"
        if output_file.exists() and output_file.stat().st_size > 0 and not overwrite:
            print(f"skip idx={idx}: existing result")
            continue
        result = run_one(agent, item, line_index)
        write_result(output_file, result)
        results.append(result)
        print(f"done idx={idx} status={result['status']}")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run user_agent.py locally with a fake official-style client.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fake-answer", default=DEFAULT_FAKE_ANSWER)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_local(
        args.input,
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        fake_answer=args.fake_answer,
    )


if __name__ == "__main__":
    main()
