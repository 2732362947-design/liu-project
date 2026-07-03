import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RISK_SK_PREFIX = "sk" + "-"
RISK_AUTH = "Authori" + "zation"
RISK_BEARER = "Bear" + "er"
RISK_HOME = "/" + "home" + "/" + "ubuntu"
RISK_LOAD_DOTENV = "load_" + "dotenv"
RISK_CALL_INTERN = "call_" + "intern_s1"
RISK_ENV_OPEN = "open(" + "\".env\""
RISK_ENV_OPEN_SINGLE = "open(" + "'.env'"
RISK_META_ANSWER_PATTERNS = (
    "metadata" + '["answer"]',
    "metadata" + "['answer']",
    "metadata.get(" + '"answer"',
    "metadata.get(" + "'answer'",
)


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return "推理过程...\n最终答案：2"


def _result() -> dict:
    return {"warnings": [], "failures": []}


def _add(items: list[dict], path: Path | str, code: str, message: str) -> None:
    try:
        display_path = str(Path(path).resolve().relative_to(ROOT))
    except (TypeError, ValueError):
        display_path = str(path)
    items.append({"path": display_path, "code": code, "message": message})


def check_response_schema(response: Any, metadata: dict | None = None) -> dict:
    result = _result()
    metadata = metadata if isinstance(metadata, dict) else {}
    if not isinstance(response, dict):
        _add(result["failures"], "response", "response_not_dict", "solve() must return a dict")
        return result

    final_response = response.get("final_response")
    if not isinstance(final_response, str) or not final_response.strip():
        _add(result["failures"], "response", "missing_final_response", "final_response must be a nonempty string")
    if metadata.get("answer") is not None and final_response == metadata.get("answer"):
        _add(result["failures"], "response", "metadata_answer_used", "final_response must not directly use metadata answer")

    trace = response.get("trace")
    if trace is not None and not isinstance(trace, list):
        _add(result["failures"], "response", "trace_not_list", "trace must be a list when present")

    try:
        json.dumps(response, ensure_ascii=False)
    except TypeError:
        _add(result["failures"], "response", "response_not_json_serializable", "solve() response must be JSON serializable")
    return result


def check_user_agent_entrypoint() -> dict:
    result = _result()
    user_agent_path = ROOT / "user_agent.py"
    if not user_agent_path.exists():
        _add(result["failures"], user_agent_path, "missing_user_agent", "repository root must contain user_agent.py")
        return result

    try:
        from user_agent import ReasoningAgent
    except Exception as exc:
        _add(result["failures"], user_agent_path, "import_failed", f"ReasoningAgent import failed: {type(exc).__name__}")
        return result

    metadata = {"idx": 0, "answer": "999"}
    try:
        client = FakeClient()
        agent = ReasoningAgent(client=client)
        response = agent.solve("1+1=?", metadata)
        if not client.calls:
            _add(result["failures"], user_agent_path, "client_not_called", "ReasoningAgent must call client.chat")
    except Exception as exc:
        _add(result["failures"], user_agent_path, "solve_failed", f"solve() raised {type(exc).__name__}")
        return result

    schema = check_response_schema(response, metadata)
    result["warnings"].extend(schema["warnings"])
    result["failures"].extend(schema["failures"])
    return result


def _default_scan_paths(root: Path = ROOT) -> list[Path]:
    paths = [root / "user_agent.py", root / "runner.py"]
    paths.extend(sorted((root / "agents").glob("*.py")))
    paths.extend(sorted((root / "dev_tools").glob("*.py")))
    return [path for path in paths if path.exists() and path.name != ".env"]


def scan_sensitive_patterns(paths: list[str | Path]) -> dict:
    result = _result()
    for raw_path in paths:
        path = Path(raw_path)
        if path.name == ".env":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")

        if RISK_SK_PREFIX in text:
            _add(result["failures"], path, "hardcoded_api_key", "possible hardcoded API key prefix")
        if RISK_AUTH in text:
            _add(result["warnings"], path, "authorization_marker", "auth marker appears in scanned file")
        if RISK_BEARER in text:
            _add(result["warnings"], path, "bearer_marker", "bearer marker appears in scanned file")
        if RISK_HOME in text:
            _add(result["warnings"], path, "absolute_home_path", "local absolute path appears in scanned file")

        if path.name == "user_agent.py":
            if RISK_LOAD_DOTENV in text:
                _add(result["failures"], path, "load_dotenv_in_user_agent", "user_agent.py must not load .env")
            if RISK_CALL_INTERN in text:
                _add(result["failures"], path, "call_intern_s1_in_user_agent", "user_agent.py must not call local Intern-S1 client")
            if RISK_ENV_OPEN in text or RISK_ENV_OPEN_SINGLE in text:
                _add(result["failures"], path, "open_env_in_user_agent", "user_agent.py must not open .env")
            for pattern in RISK_META_ANSWER_PATTERNS:
                if pattern in text:
                    _add(result["failures"], path, "metadata_answer_used", "user_agent.py must not directly read metadata answer")
                    break
    return result


def check_requirements() -> dict:
    result = _result()
    path = ROOT / "requirements.txt"
    if not path.exists():
        _add(result["failures"], path, "missing_requirements", "requirements.txt must exist")
    return result


def run_checks() -> dict:
    summary = _result()
    for check in (
        check_user_agent_entrypoint(),
        check_requirements(),
        scan_sensitive_patterns(_default_scan_paths()),
    ):
        summary["warnings"].extend(check["warnings"])
        summary["failures"].extend(check["failures"])
    summary["passed"] = not summary["failures"]
    return summary


def main() -> None:
    summary = run_checks()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
