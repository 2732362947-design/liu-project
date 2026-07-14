import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = PROJECT_ROOT
DEFAULT_BASE_URL = "https://chat.intern-ai.org.cn/api/v1"
DEFAULT_MODEL = "intern-latest"
DEFAULT_PROBLEM = "1+1=?"
DEFAULT_IDX = "real_smoke_001"
SENSITIVE_MARKERS = ("Authorization", "Bearer", "api_key", "token")
METADATA_ALLOWLIST = {
    "idx",
    "domain",
    "difficulty",
    "source",
    "answer_type",
    "raw_domain",
    "solver_key",
    "subject",
    "category",
}
METADATA_DENYLIST = {
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


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def safe_text(value: Any, max_chars: int = 500) -> str:
    text = str(value or "")
    for marker in SENSITIVE_MARKERS:
        text = text.replace(marker, "[redacted]")
    return text[:max_chars]


def extract_content(response_json: Any) -> str:
    if isinstance(response_json, str):
        return response_json
    if isinstance(response_json, dict):
        if response_json.get("content") is not None:
            return str(response_json["content"])
        choices = response_json.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    return str(message["content"])
                if choice.get("content") is not None:
                    return str(choice["content"])
    return str(response_json or "")


class RealInternClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120,
        session=None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = session or requests.Session()

    def chat(self, messages, temperature=0.2, max_tokens=4096):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if getattr(response, "status_code", 200) >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {safe_text(getattr(response, 'text', ''))}")
            return extract_content(response.json())
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"request failed: {safe_text(exc)}") from exc
        except Exception as exc:
            raise RuntimeError(f"client failed: {safe_text(exc)}") from exc


def _safe_metadata_from_item(item: dict, idx: str) -> dict:
    metadata = {"idx": idx}
    source = item.get("metadata") if isinstance(item.get("metadata"), dict) else item
    for key in METADATA_ALLOWLIST:
        if key == "idx":
            continue
        if key in source and key.lower() not in METADATA_DENYLIST:
            metadata[key] = source[key]
    return metadata


def _load_input_items(input_json: str | Path) -> list[dict]:
    input_path = _resolve_path(input_json)
    raw_text = input_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("input must be a JSON array or JSONL objects")
    return parsed


def _load_input_item(input_json: str | Path, item_index: int) -> tuple[str, str, dict]:
    items = _load_input_items(input_json)
    if item_index < 0 or item_index >= len(items):
        raise IndexError(f"--item-index {item_index} is out of range for {len(items)} items")
    item = items[item_index]
    if not isinstance(item, dict):
        raise ValueError("selected input item must be a JSON object")
    problem = str(item.get("problem") or "")
    if not problem:
        raise ValueError("selected input item has empty problem")
    idx = str(item.get("problem_id") or item.get("idx") or DEFAULT_IDX)
    metadata = _safe_metadata_from_item(item, idx)
    return problem, idx, metadata


def run_smoke(client, problem: str = DEFAULT_PROBLEM, idx: str = DEFAULT_IDX, metadata: dict | None = None) -> dict:
    from user_agent import ReasoningAgent

    started = time.perf_counter()
    agent = ReasoningAgent(client=client)
    solve_metadata = metadata if isinstance(metadata, dict) else {"idx": idx}
    result = agent.solve(problem, solve_metadata)
    trace = result.get("trace", []) if isinstance(result, dict) else []
    retry_used = any(item.get("step") == "retry_model_call" for item in trace if isinstance(item, dict))
    return {
        "idx": idx,
        "final_response": str(result.get("final_response", "")) if isinstance(result, dict) else "",
        "trace": trace if isinstance(trace, list) else [],
        "retry_used": retry_used,
        "time_cost_seconds": round(time.perf_counter() - started, 6),
    }


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def _build_client_from_env(timeout: float) -> RealInternClient:
    _load_env_file()
    api_key = os.getenv("INTERN_S1_API_KEY", "")
    if not api_key:
        raise RuntimeError("INTERN_S1_API_KEY is not configured")
    return RealInternClient(
        api_key=api_key,
        base_url=os.getenv("INTERN_S1_BASE_URL", DEFAULT_BASE_URL),
        model=os.getenv("INTERN_S1_MODEL", DEFAULT_MODEL),
        timeout=timeout,
    )


def _write_output(path: str | Path, result: dict) -> None:
    output_file = _resolve_path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-question user_agent.py smoke test with a real Intern-S API client.")
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--idx", default=DEFAULT_IDX)
    parser.add_argument("--input-json", default=None, help="Optional JSON array file containing converted question items.")
    parser.add_argument("--item-index", type=int, default=0, help="Question index to use with --input-json.")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        problem = args.problem
        idx = args.idx
        metadata = {"idx": idx}
        if args.input_json:
            problem, idx, metadata = _load_input_item(args.input_json, args.item_index)
        client = _build_client_from_env(args.timeout)
        result = run_smoke(client, problem=problem, idx=idx, metadata=metadata)
        print(f"final_response: {safe_text(result['final_response'])}")
        print(f"trace_steps: {[item.get('step') for item in result['trace'] if isinstance(item, dict)]}")
        print(f"retry_used: {result['retry_used']}")
        print(f"time_cost_seconds: {result['time_cost_seconds']}")
        if args.output:
            _write_output(args.output, result)
    except Exception as exc:
        print(f"real smoke failed: {safe_text(exc)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
